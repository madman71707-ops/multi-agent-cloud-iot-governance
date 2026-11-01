"""Tests for the switchable TAPNA components and their ablation variants."""
import math

import pytest

from negotiation.tapna import (
    compute_adaptive_weight,
    compute_dynamic_weight,
    pareto_filter,
    run_tapna,
    run_tapna_variant,
)
from sim import experiment_runner as er


def _actions(count):
    return [{"id": str(index), "type": "no_op"} for index in range(count)]


# ---------------------------------------------------------------------------
# Full TAPNA behavior is unchanged
# ---------------------------------------------------------------------------

def test_compute_adaptive_weight_with_decay_matches_compute_dynamic_weight():
    for tss in (0.0, 0.1, 0.15, 0.2, 0.5, 0.9, 1.0):
        for prev in (1 / 3, 0.5, 0.9):
            assert compute_adaptive_weight(
                tss, prev_wS=prev, use_decay=True
            ) == compute_dynamic_weight(tss, prev_wS=prev)


def test_run_tapna_equals_full_variant():
    # run_tapna must be exactly run_tapna_variant with every component on.
    actions = _actions(5)
    u_r = [0.9, 0.2, 0.5, 0.4, 0.7]
    u_s = [0.1, 0.8, 0.5, 0.3, 0.9]
    u_c = [0.4, 0.3, 0.6, 0.2, 0.5]

    for tss in (0.0, 0.15, 0.5, 1.0):
        for prev in (1 / 3, 0.7):
            full_action, full_weights = run_tapna(actions, u_r, u_s, u_c, tss, prev)
            variant_action, variant_weights = run_tapna_variant(
                actions, u_r, u_s, u_c, tss, prev,
                use_pareto=True,
                use_adaptive_weight=True,
                use_decay=True,
            )
            assert full_action == variant_action
            assert full_weights == variant_weights


def test_pareto_filtering_matters_when_security_weight_reaches_one():
    """Documented edge case: at wS == 1 the objective weights collapse to
    (0, 1, 0), so candidates that differ only on U_R/U_C tie on the weighted
    sum. The earliest-index tie-break can then land on a dominated action,
    which is exactly what Pareto filtering exists to prevent.

    tss == 1.0 is the only TSS at which wS reaches 1.0.
    """
    actions = _actions(4)
    # Candidate 1 strictly dominates candidate 0 (higher U_R, equal U_S and
    # U_C), so the two tie on the weighted sum when the weights are (0, 1, 0).
    u_r = [0.1, 0.9, 0.5, 0.5]
    u_s = [0.9, 0.9, 0.5, 0.5]
    u_c = [0.5, 0.5, 0.9, 0.5]

    assert compute_adaptive_weight(1.0, use_decay=False) == pytest.approx(1.0)

    # Without Pareto filtering the tie-break favours the earlier candidate 0,
    # which candidate 1 strictly dominates.
    unfiltered, weights = run_tapna_variant(
        actions, u_r, u_s, u_c, tss=1.0, prev_wS=1 / 3,
        use_pareto=False, use_adaptive_weight=True, use_decay=False,
    )
    assert weights["wR"] == pytest.approx(0.0)
    assert weights["wC"] == pytest.approx(0.0)
    assert unfiltered is actions[0]

    # Pareto filtering removes the dominated candidate before selection.
    filtered, _ = run_tapna_variant(
        actions, u_r, u_s, u_c, tss=1.0, prev_wS=1 / 3,
        use_pareto=True, use_adaptive_weight=True, use_decay=False,
    )
    assert filtered is actions[1]


# ---------------------------------------------------------------------------
# Adaptive weight: decay on vs off
# ---------------------------------------------------------------------------

def test_adaptive_weight_ramp_is_identical_with_and_without_decay():
    # Above tau both modes follow the same threat-driven ramp.
    for tss in (0.16, 0.3, 0.575, 0.9, 1.0):
        assert compute_adaptive_weight(tss, use_decay=True) == pytest.approx(
            compute_adaptive_weight(tss, use_decay=False)
        )


def test_adaptive_weight_without_decay_returns_neutral_below_tau():
    for tss in (0.0, 0.05, 0.1, 0.15):
        assert compute_adaptive_weight(tss, use_decay=False) == pytest.approx(1 / 3)


def test_adaptive_weight_with_decay_retains_previous_weight_below_tau():
    # Decay keeps a memory of the previous elevated weight, so the result
    # stays strictly between the neutral 1/3 and the previous weight.
    neutral = 1 / 3
    for tss in (0.0, 0.05, 0.1, 0.15):
        for prev in (0.5, 0.9):
            decayed = compute_adaptive_weight(tss, prev_wS=prev, use_decay=True)
            assert neutral < decayed < prev
            assert decayed == pytest.approx(
                neutral + (prev - neutral) * math.exp(-0.3)
            )


def test_adaptive_weight_without_decay_ignores_previous_weight():
    assert compute_adaptive_weight(
        0.0, prev_wS=0.95, use_decay=False
    ) == compute_adaptive_weight(0.0, prev_wS=1 / 3, use_decay=False)


def test_adaptive_weight_weights_always_sum_to_one():
    for tss in (0.0, 0.15, 0.5, 1.0):
        for use_decay in (True, False):
            wS = compute_adaptive_weight(tss, use_decay=use_decay)
            wR = (1 - wS) / 2
            assert wR + wS + wR == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Pareto switch
# ---------------------------------------------------------------------------

def _dominates(left, right):
    return all(l >= r for l, r in zip(left, right)) and any(
        l > r for l, r in zip(left, right)
    )


def test_pareto_filter_returns_exactly_the_non_dominated_actions():
    actions = _actions(3)
    # Candidate 1 strictly dominates candidate 0 on U_R; candidate 2 is
    # incomparable with both.
    scores = {"U_R": [0.1, 0.9, 0.5], "U_S": [0.9, 0.9, 0.5], "U_C": [0.5, 0.5, 0.9]}

    surviving = pareto_filter(actions, scores)

    assert actions[0] not in surviving
    assert actions[1] in surviving
    assert actions[2] in surviving


def test_no_variant_ever_selects_a_strictly_dominated_candidate():
    """Weights are non-negative, so a dominator always outscores the dominated.

    This holds with Pareto filtering on (the dominated action is excluded) and
    with it off (the dominator simply scores higher). Both are asserted here so
    a future weight change that could break it fails loudly.
    """
    cases = [
        ([0.9, 0.5, 0.4], [0.9, 0.9, 0.5], [0.9, 0.4, 0.95]),
        ([0.3, 0.9, 0.2], [0.3, 0.8, 0.6], [0.3, 0.7, 0.4]),
        ([0.6, 0.6, 0.1, 0.9], [0.6, 0.6, 0.2, 0.2], [0.6, 0.6, 0.3, 0.1]),
    ]

    for u_r, u_s, u_c in cases:
        actions = _actions(len(u_r))
        objectives = list(zip(u_r, u_s, u_c))

        for tss in (0.0, 0.5, 1.0):
            for use_pareto in (True, False):
                for use_adaptive in (True, False):
                    chosen, _ = run_tapna_variant(
                        actions, u_r, u_s, u_c, tss=tss, prev_wS=1 / 3,
                        use_pareto=use_pareto,
                        use_adaptive_weight=use_adaptive,
                        use_decay=False,
                    )
                    chosen_vector = objectives[actions.index(chosen)]
                    assert not any(
                        _dominates(other, chosen_vector)
                        for position, other in enumerate(objectives)
                        if actions[position] is not chosen
                    ), (tss, use_pareto, use_adaptive, chosen_vector)


# ---------------------------------------------------------------------------
# Fixed-weight variants
# ---------------------------------------------------------------------------

def test_fixed_weight_variants_use_equal_thirds():
    actions = _actions(4)
    u_r = [0.1, 0.9, 0.5, 0.4]
    u_s = [0.9, 0.1, 0.5, 0.3]
    u_c = [0.4, 0.9, 0.5, 0.2]

    for tss in (0.0, 0.5, 1.0):
        for use_decay in (True, False):
            _, weights = run_tapna_variant(
                actions, u_r, u_s, u_c, tss=tss, prev_wS=0.9,
                use_pareto=True, use_adaptive_weight=False, use_decay=use_decay,
            )
            assert weights == {"wR": 1 / 3, "wS": 1 / 3, "wC": 1 / 3}


def test_adaptive_variants_do_not_hold_weights_at_neutral():
    actions = _actions(4)
    u_r = [0.1, 0.9, 0.5, 0.4]
    u_s = [0.9, 0.1, 0.5, 0.3]
    u_c = [0.4, 0.9, 0.5, 0.2]

    _, weights = run_tapna_variant(
        actions, u_r, u_s, u_c, tss=0.9, prev_wS=1 / 3,
        use_pareto=True, use_adaptive_weight=True, use_decay=False,
    )
    assert weights["wS"] > 1 / 3


def test_full_variant_still_restricts_to_the_pareto_front():
    actions = _actions(3)
    # Candidate 0 is strictly dominated by candidate 1 on every objective.
    u_r = [0.3, 0.9, 0.5]
    u_s = [0.3, 0.8, 0.5]
    u_c = [0.3, 0.7, 0.5]

    chosen, _ = run_tapna_variant(
        actions, u_r, u_s, u_c, tss=0.5, prev_wS=1 / 3,
        use_pareto=True, use_adaptive_weight=True, use_decay=True,
    )
    assert chosen is actions[1]


def test_run_tapna_variant_rejects_empty_candidate_set():
    with pytest.raises(ValueError):
        run_tapna_variant([], [], [], [], tss=0.5, prev_wS=1 / 3)


# ---------------------------------------------------------------------------
# Runner wiring
# ---------------------------------------------------------------------------

def test_canonical_strategy_list_is_unchanged():
    # The five-strategy experiment and its existing results must not shift.
    assert er.STRATEGIES == [
        "TAPNA",
        "Greedy Cost",
        "Greedy Latency",
        "Static Weighted",
        "NSGA-II",
    ]


def test_four_ablation_variants_are_defined_with_the_intended_switches():
    assert er.ABLATION_STRATEGIES == [
        "TAPNA Pareto-only",
        "TAPNA Adaptive-only",
        "TAPNA Pareto+Adaptive",
        "TAPNA Full",
    ]
    assert er.TAPNA_ABLATION_VARIANTS["TAPNA Pareto-only"] == {
        "use_pareto": True,
        "use_adaptive_weight": False,
        "use_decay": False,
    }
    assert er.TAPNA_ABLATION_VARIANTS["TAPNA Adaptive-only"] == {
        "use_pareto": False,
        "use_adaptive_weight": True,
        "use_decay": False,
    }
    assert er.TAPNA_ABLATION_VARIANTS["TAPNA Pareto+Adaptive"] == {
        "use_pareto": True,
        "use_adaptive_weight": True,
        "use_decay": False,
    }
    assert er.TAPNA_ABLATION_VARIANTS["TAPNA Full"] == {
        "use_pareto": True,
        "use_adaptive_weight": True,
        "use_decay": True,
    }
    assert er.ALL_STRATEGIES == er.STRATEGIES + er.ABLATION_STRATEGIES


def test_ablation_names_do_not_collide_with_canonical_strategy_names():
    assert not set(er.ABLATION_STRATEGIES) & set(er.STRATEGIES)


@pytest.mark.parametrize("strategy", er.ABLATION_STRATEGIES)
def test_ablation_variant_runs_in_the_runner(monkeypatch, strategy):
    monkeypatch.setattr(er, "NUM_TICKS", 2)

    rows = er._run_strategy(strategy, set(), 789)

    assert len(rows) == 2
    for row in rows:
        assert row["strategy"] == strategy
        assert row["seed"] == 789
        for field in ("U_R", "U_S", "U_C", "FairScore", "CIGI", "TSS"):
            assert math.isfinite(row[field])
        # FairScore stays the fixed equal-thirds evaluation for every variant.
        assert row["FairScore"] == pytest.approx(
            (1 / 3) * row["U_R"] + (1 / 3) * row["U_S"] + (1 / 3) * row["U_C"]
        )


def test_tapna_full_ablation_reproduces_tapna_exactly(monkeypatch):
    """The unablated control must match the canonical TAPNA strategy row for row."""
    monkeypatch.setattr(er, "NUM_TICKS", 3)

    identity = ("tick", "TSS", "wR", "wS", "wC", "CIGI",
                "U_R", "U_S", "U_C", "FairScore", "chosen_action_type")

    tapna = er._run_strategy("TAPNA", {2}, 42)
    full = er._run_strategy("TAPNA Full", {2}, 42)

    assert len(tapna) == len(full)
    for left, right in zip(tapna, full):
        assert [left[key] for key in identity] == [right[key] for key in identity]


def test_build_summary_orders_ablation_rows_after_canonical_ones():
    import pandas as pd

    rows = []
    for strategy in er.ALL_STRATEGIES:
        for value in (0.2, 0.4):
            rows.append({
                "strategy": strategy,
                "FairScore": value,
                "CIGI": value,
                "TSS": value,
            })

    summary = er.build_summary(pd.DataFrame(rows))

    assert list(summary["strategy"]) == er.ALL_STRATEGIES


def test_build_summary_ignores_strategies_absent_from_results():
    import pandas as pd

    results = pd.DataFrame([
        {"strategy": "TAPNA", "FairScore": 0.2, "CIGI": 0.4, "TSS": 0.05},
        {"strategy": "TAPNA", "FairScore": 0.4, "CIGI": 0.6, "TSS": 0.15},
    ])

    summary = er.build_summary(results)

    assert list(summary["strategy"]) == ["TAPNA"]


def test_main_defaults_to_the_canonical_five_strategies(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(er, "SEEDS", [42])
    monkeypatch.setattr(er, "NUM_TICKS", 1)

    er.main()

    import pandas as pd

    raw = pd.read_csv(tmp_path / "data" / "experiments" / "experiment_results.csv")
    assert len(raw) == 1 * 5 * 1
    assert set(raw["strategy"]) == set(er.STRATEGIES)

    summary = pd.read_csv(tmp_path / "data" / "experiments" / "experiment_summary.csv")
    assert list(summary["strategy"]) == er.STRATEGIES


def test_main_can_include_the_ablation_variants(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(er, "SEEDS", [42])
    monkeypatch.setattr(er, "NUM_TICKS", 1)

    er.main(include_ablations=True)

    import pandas as pd

    raw = pd.read_csv(tmp_path / "data" / "experiments" / "experiment_results.csv")
    assert len(raw) == 1 * 9 * 1
    assert set(raw["strategy"]) == set(er.ALL_STRATEGIES)

    summary = pd.read_csv(tmp_path / "data" / "experiments" / "experiment_summary.csv")
    assert list(summary["strategy"]) == er.ALL_STRATEGIES
    assert len(summary) == 9