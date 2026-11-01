"""Tests for the TAPNA tau / decay_lambda sensitivity experiments."""
import math

import pandas as pd
import pytest

from negotiation.tapna import (
    compute_adaptive_weight,
    compute_dynamic_weight,
    run_tapna,
    run_tapna_variant,
)
from sim import experiment_runner as er
from sim import sensitivity_runner as sr


# ---------------------------------------------------------------------------
# Defaults must reproduce canonical Full TAPNA
# ---------------------------------------------------------------------------

def test_compute_adaptive_weight_defaults_match_canonical_dynamic_weight():
    for tss in (0.0, 0.1, 0.15, 0.2, 0.5, 0.9, 1.0):
        for prev in (1 / 3, 0.5, 0.9):
            assert compute_adaptive_weight(tss, prev_wS=prev) == pytest.approx(
                compute_dynamic_weight(tss, prev_wS=prev)
            )


def test_run_tapna_variant_defaults_match_run_tapna():
    actions = [{"id": str(i), "type": "no_op"} for i in range(5)]
    u_r = [0.9, 0.2, 0.5, 0.4, 0.7]
    u_s = [0.1, 0.8, 0.5, 0.3, 0.9]
    u_c = [0.4, 0.3, 0.6, 0.2, 0.5]

    for tss in (0.0, 0.15, 0.5, 1.0):
        for prev in (1 / 3, 0.7):
            left = run_tapna(actions, u_r, u_s, u_c, tss, prev)
            right = run_tapna_variant(actions, u_r, u_s, u_c, tss, prev)
            assert left == right


def test_sensitivity_at_canonical_parameters_matches_tapna_strategy(monkeypatch):
    """Running the sweep at tau=0.15, lambda=0.3 must reproduce TAPNA."""
    monkeypatch.setattr(er, "NUM_TICKS", 3)

    sweep_rows = sr._run_tapna_sensitivity(
        set(), 42, sr.DEFAULT_TAU, sr.DEFAULT_DECAY_LAMBDA, num_ticks=3
    )
    tapna_rows = er._run_strategy("TAPNA", set(), 42)

    identity = ("tick", "TSS", "wR", "wS", "wC", "CIGI",
                "U_R", "U_S", "U_C", "FairScore", "chosen_action_type")

    assert len(sweep_rows) == len(tapna_rows)
    for sweep, tapna in zip(sweep_rows, tapna_rows):
        assert [sweep[key] for key in identity] == [tapna[key] for key in identity]


# ---------------------------------------------------------------------------
# Parameter plumbing
# ---------------------------------------------------------------------------

def test_tau_shifts_the_ramp_threshold():
    # A TSS below the higher tau decays, but ramps for the lower tau.
    tss = 0.25
    assert compute_adaptive_weight(
        tss, tau=0.10, prev_wS=1 / 3, use_decay=True
    ) > 1 / 3
    assert compute_adaptive_weight(
        tss, tau=0.30, prev_wS=1 / 3, use_decay=True
    ) == pytest.approx(1 / 3)


def test_decay_lambda_controls_the_decay_rate():
    # Below tau, a larger lambda decays further toward the neutral weight.
    tss, prev = 0.0, 0.9
    slow = compute_adaptive_weight(
        tss, tau=0.15, prev_wS=prev, use_decay=True, decay_lambda=0.10
    )
    fast = compute_adaptive_weight(
        tss, tau=0.15, prev_wS=prev, use_decay=True, decay_lambda=0.70
    )
    assert fast < slow < prev
    assert slow == pytest.approx(1 / 3 + (prev - 1 / 3) * math.exp(-0.10))
    assert fast == pytest.approx(1 / 3 + (prev - 1 / 3) * math.exp(-0.70))


def test_decay_lambda_ignored_when_decay_disabled():
    for decay_lambda in (0.10, 0.30, 0.50, 0.70):
        for tss in (0.0, 0.2, 0.5):
            assert compute_adaptive_weight(
                tss, tau=0.15, prev_wS=0.9,
                use_decay=False, decay_lambda=decay_lambda,
            ) == pytest.approx(
                compute_adaptive_weight(
                    tss, tau=0.15, prev_wS=1 / 3, use_decay=False,
                )
            )


def test_run_tapna_variant_accepts_tau_and_lambda():
    actions = [{"id": str(i), "type": "no_op"} for i in range(3)]
    u_r = [0.5, 0.6, 0.7]
    u_s = [0.5, 0.6, 0.7]
    u_c = [0.5, 0.6, 0.7]

    _, weights = run_tapna_variant(
        actions, u_r, u_s, u_c, tss=0.25, prev_wS=0.9,
        use_decay=True, tau=0.10, decay_lambda=0.70,
    )
    assert weights["wS"] == pytest.approx(
        compute_dynamic_weight(0.25, tau=0.10, prev_wS=0.9, decay_lambda=0.70)
    )


def test_weights_sum_to_one_for_every_swept_parameter():
    actions = [{"id": str(i), "type": "no_op"} for i in range(3)]
    u_r = [0.5, 0.6, 0.7]
    u_s = [0.2, 0.4, 0.6]
    u_c = [0.9, 0.3, 0.1]

    for tau in sr.TAU_VALUES:
        for decay_lambda in sr.DECAY_LAMBDA_VALUES:
            for tss in (0.0, 0.2, 0.5, 1.0):
                _, weights = run_tapna_variant(
                    actions, u_r, u_s, u_c, tss=tss, prev_wS=0.8,
                    use_decay=True, tau=tau, decay_lambda=decay_lambda,
                )
                total = weights["wR"] + weights["wS"] + weights["wC"]
                assert total == pytest.approx(1.0)
                assert 0.0 <= weights["wS"] <= 1.0


# ---------------------------------------------------------------------------
# Sweep construction
# ---------------------------------------------------------------------------

def test_default_sweep_values_match_the_specification():
    assert sr.TAU_VALUES == [0.10, 0.15, 0.20, 0.40]
    assert sr.DECAY_LAMBDA_VALUES == [0.10, 0.30, 0.50, 0.70]


def test_canonical_parameters_are_the_documented_defaults():
    assert sr.DEFAULT_TAU == 0.15
    assert sr.DEFAULT_DECAY_LAMBDA == 0.3


def test_configurations_cover_both_sweeps():
    configurations = sr.build_sensitivity_configurations()

    taus = {c["value"] for c in configurations if c["parameter"] == "tau"}
    lambdas = {
        c["value"] for c in configurations if c["parameter"] == "decay_lambda"
    }

    assert taus == set(sr.TAU_VALUES)
    assert lambdas == set(sr.DECAY_LAMBDA_VALUES)


def test_tau_sweep_holds_lambda_at_the_canonical_value():
    for config in sr.build_sensitivity_configurations():
        if config["parameter"] == "tau":
            assert config["decay_lambda"] == sr.DEFAULT_DECAY_LAMBDA


def test_lambda_sweep_holds_tau_at_the_canonical_value():
    for config in sr.build_sensitivity_configurations():
        if config["parameter"] == "decay_lambda":
            assert config["tau"] == sr.DEFAULT_TAU


def test_baseline_configuration_is_present_and_flagged():
    configurations = sr.build_sensitivity_configurations()

    baselines = [c for c in configurations if c["is_baseline"]]
    assert baselines, "no baseline configuration produced"

    # Exactly one configuration may sit at the canonical pair per sweep axis,
    # and the dedicated baseline entry is never duplicated.
    canonical = [
        c for c in configurations
        if c["tau"] == sr.DEFAULT_TAU
        and c["decay_lambda"] == sr.DEFAULT_DECAY_LAMBDA
    ]
    assert len(canonical) == len({
        (c["parameter"], c["value"]) for c in canonical
    })
    assert all(c["tau"] == sr.DEFAULT_TAU for c in canonical)


def test_baseline_can_be_omitted():
    configurations = sr.build_sensitivity_configurations(include_baseline=False)

    assert not any(c["parameter"] == "baseline" for c in configurations)


def test_configurations_have_no_duplicate_identities():
    configurations = sr.build_sensitivity_configurations()

    identities = [
        (c["parameter"], c["value"], c["tau"], c["decay_lambda"])
        for c in configurations
    ]
    assert len(identities) == len(set(identities))


def test_custom_sweep_values_are_honored():
    configurations = sr.build_sensitivity_configurations(
        tau_values=[0.05], lambda_values=[0.9], include_baseline=False
    )

    assert len(configurations) == 2
    assert {c["value"] for c in configurations} == {0.05, 0.9}


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def test_summary_reports_mean_and_std_for_all_three_metrics():
    rows = []
    for parameter, value, tau, decay_lambda in [
        ("tau", 0.10, 0.10, 0.30),
        ("tau", 0.30, 0.30, 0.30),
    ]:
        for fair, cigi, w_s in [(0.2, 0.4, 0.34), (0.4, 0.6, 0.36)]:
            rows.append({
                "parameter": parameter,
                "value": value,
                "tau": tau,
                "decay_lambda": decay_lambda,
                "FairScore": fair,
                "CIGI": cigi,
                "wS": w_s,
            })

    summary = sr.build_sensitivity_summary(pd.DataFrame(rows))

    assert list(summary["parameter"]) == ["tau", "tau"]
    assert list(summary["value"]) == [0.10, 0.30]
    for column in ("FairScore_mean", "FairScore_std",
                   "CIGI_mean", "CIGI_std", "wS_mean", "wS_std"):
        assert column in summary.columns

    first = summary.iloc[0]
    assert first["FairScore_mean"] == pytest.approx(0.3)
    assert first["FairScore_std"] == pytest.approx(abs(0.4 - 0.2) / math.sqrt(2))
    assert first["wS_mean"] == pytest.approx(0.35)
    assert first["wS_std"] == pytest.approx(abs(0.36 - 0.34) / math.sqrt(2))
    assert first["n_rows"] == 2


def test_summary_keeps_the_two_sweeps_separate():
    rows = []
    # tau=0.30 appears in both sweeps as different sweep axes.
    for parameter, value, tau, decay_lambda in [
        ("tau", 0.30, 0.30, 0.30),
        ("decay_lambda", 0.30, 0.15, 0.30),
    ]:
        rows.append({
            "parameter": parameter,
            "value": value,
            "tau": tau,
            "decay_lambda": decay_lambda,
            "FairScore": 0.5,
            "CIGI": 0.5,
            "wS": 0.5,
        })

    summary = sr.build_sensitivity_summary(pd.DataFrame(rows))

    assert len(summary) == 2
    assert set(summary["parameter"]) == {"tau", "decay_lambda"}


def test_summary_matches_an_independent_aggregation():
    import numpy as np

    rows = []
    for value in (0.10, 0.20):
        for index in range(6):
            rows.append({
                "parameter": "tau",
                "value": value,
                "tau": value,
                "decay_lambda": 0.3,
                "FairScore": index / 10,
                "CIGI": index / 5,
                "wS": 1 / 3 + index / 100,
            })
    frame = pd.DataFrame(rows)

    summary = sr.build_sensitivity_summary(frame)

    for _, row in summary.iterrows():
        subset = frame[frame["value"] == row["value"]]
        for metric in ("FairScore", "CIGI", "wS"):
            values = subset[metric].to_numpy(dtype=float)
            assert row[f"{metric}_mean"] == pytest.approx(values.mean())
            assert row[f"{metric}_std"] == pytest.approx(values.std(ddof=1))


# ---------------------------------------------------------------------------
# End-to-end smoke behavior
# ---------------------------------------------------------------------------

def test_sensitivity_run_writes_both_csvs_and_covers_configurations(tmp_path):
    results = sr.run_sensitivity(
        seeds=[42, 123],
        tau_values=[0.10, 0.30],
        lambda_values=[0.50],
        num_ticks=2,
        include_baseline=True,
        output_dir=tmp_path / "sens",
    )

    raw_path = tmp_path / "sens" / "sensitivity_results.csv"
    summary_path = tmp_path / "sens" / "sensitivity_summary.csv"
    assert raw_path.is_file()
    assert summary_path.is_file()

    raw = pd.read_csv(raw_path)
    summary = pd.read_csv(summary_path)

    expected_configs = 4  # 2 tau + 1 lambda + 1 baseline
    assert len(summary) == expected_configs
    assert len(raw) == expected_configs * 2 * 2
    assert sorted(raw["seed"].unique()) == [42, 123]
    assert set(raw["tau"]) <= set([0.10, 0.30, sr.DEFAULT_TAU])
    assert set(raw["decay_lambda"]) <= set([0.50, sr.DEFAULT_DECAY_LAMBDA])

    for column in ("FairScore_mean", "FairScore_std", "CIGI_mean",
                   "CIGI_std", "wS_mean", "wS_std",
                   "parameter", "value", "tau", "decay_lambda"):
        assert column in summary.columns

    # Every configuration must have the full expected row count.
    assert (summary["n_rows"] == 4).all()

    # wS must stay a valid weight in the raw output.
    assert raw["wS"].between(0.0, 1.0).all()

    # FairScore stays the fixed equal-thirds evaluation.
    fair = (raw["U_R"] + raw["U_S"] + raw["U_C"]) / 3.0
    assert (fair - raw["FairScore"]).abs().max() < 1e-12


def test_sensitivity_run_is_reproducible(tmp_path):
    kwargs = dict(
        seeds=[42],
        tau_values=[0.10],
        lambda_values=[0.50],
        num_ticks=2,
        include_baseline=False,
    )

    first = sr.run_sensitivity(output_dir=tmp_path / "a", **kwargs)
    second = sr.run_sensitivity(output_dir=tmp_path / "b", **kwargs)

    pd.testing.assert_frame_equal(first, second)


def test_cached_replay_matches_uncached_execution():
    """Caching the environment per seed must not change any result.

    ``run_sensitivity`` captures each seed's environment once and replays
    the negotiation per configuration; ``_run_tapna_sensitivity``
    re-simulates the environment for every configuration. Both paths must
    produce identical rows, including float-exact wS/CIGI/FairScore values
    and the same chosen action each tick.
    """
    seed = 42
    # Threat late enough that the pre-threat ticks stay in the low-TSS
    # regime (trust only random-walks back, so early threats would pin the
    # whole trace in the ramp branch).
    threat_schedule = {3}
    configs = [
        (0.10, 0.10),
        (0.10, 0.70),
        (0.15, 0.30),  # canonical baseline
        (0.20, 0.50),
        (0.40, 0.30),
    ]
    num_ticks = 6

    # Uncached path: environment re-simulated for every configuration.
    uncached_rows = [
        row
        for tau, decay_lambda in configs
        for row in sr._run_tapna_sensitivity(
            threat_schedule, seed, tau, decay_lambda, num_ticks=num_ticks
        )
    ]

    # Cached path: environment captured once, replayed per configuration.
    trace = list(sr._iterate_environment_ticks(
        threat_schedule, seed, num_ticks=num_ticks
    ))
    cached_rows = [
        row
        for tau, decay_lambda in configs
        for row in sr._run_tapna_on_ticks(trace, seed, tau, decay_lambda)
    ]

    assert len(uncached_rows) == len(cached_rows) == len(configs) * num_ticks
    for uncached, cached in zip(uncached_rows, cached_rows):
        assert uncached == cached

    # The comparison above is only meaningful if the threat schedule and
    # the parameters actually exercise both weight branches: at least one
    # tick must ramp (wS above the neutral 1/3) and at least one must sit
    # in the decay branch (wS exactly 1/3, since the weight starts neutral
    # and the ramp never pushes wS below 1/3).
    assert any(row["wS"] > 1 / 3 + 1e-9 for row in cached_rows)
    assert any(abs(row["wS"] - 1 / 3) < 1e-12 for row in cached_rows)


def test_higher_lambda_produces_a_lower_mean_security_weight(tmp_path):
    """The decay rate must actually move wS, otherwise the sweep is inert.

    tau is pinned above the observed TSS range so every tick takes the decay
    branch and the comparison isolates decay_lambda.
    """
    low = sr.run_sensitivity(
        seeds=[42], tau_values=[0.30], lambda_values=[0.10],
        num_ticks=20, include_baseline=False,
        output_dir=tmp_path / "low",
    )
    high = sr.run_sensitivity(
        seeds=[42], tau_values=[0.30], lambda_values=[0.70],
        num_ticks=20, include_baseline=False,
        output_dir=tmp_path / "high",
    )

    # Every tick must be in the decay branch for this test to isolate lambda.
    assert (low["TSS"] <= 0.30).all(), (
        "a TSS exceeded tau=0.30; the ramp branch would mask decay_lambda"
    )

    def w_s_mean_for_lambda(frame, value):
        """Select the lambda-sweep row explicitly.

        The summary is sorted by parameter, so a positional index would pick
        the tau-sweep row instead.
        """
        summary = sr.build_sensitivity_summary(frame)
        row = summary[
            (summary["parameter"] == "decay_lambda")
            & (summary["value"] == value)
        ]
        assert len(row) == 1, f"expected one decay_lambda={value} row"
        return row["wS_mean"].iloc[0]

    low_mean = w_s_mean_for_lambda(low, 0.10)
    high_mean = w_s_mean_for_lambda(high, 0.70)

    # A higher decay rate relaxes the weight toward 1/3 faster, so the
    # average security weight must be lower.
    assert high_mean < low_mean


def test_higher_tau_produces_a_lower_mean_security_weight(tmp_path):
    """Raising tau keeps more ticks in the decay branch, lowering mean wS.

    Needs enough ticks for some TSS values to fall between the two taus: with
    only a handful of ticks no TSS may land in (0.10, 0.30] and the two
    configurations would be indistinguishable.
    """
    low = sr.run_sensitivity(
        seeds=[42], tau_values=[0.10], lambda_values=[0.30],
        num_ticks=100, include_baseline=False,
        output_dir=tmp_path / "low",
    )
    high = sr.run_sensitivity(
        seeds=[42], tau_values=[0.30], lambda_values=[0.30],
        num_ticks=100, include_baseline=False,
        output_dir=tmp_path / "high",
    )

    # Guard the premise: the two taus must actually separate some ticks,
    # otherwise the comparison below would be vacuous.
    low_tss = low["TSS"]
    assert ((low_tss > 0.10) & (low_tss <= 0.30)).any(), (
        "no TSS in (0.10, 0.30]; increase num_ticks for this test"
    )

    def w_s_mean_for_tau(frame, value):
        """Select the tau-sweep row explicitly.

        The summary is sorted by parameter, so a positional index would pick
        the decay_lambda-sweep row instead.
        """
        summary = sr.build_sensitivity_summary(frame)
        row = summary[
            (summary["parameter"] == "tau") & (summary["value"] == value)
        ]
        assert len(row) == 1, f"expected one tau={value} row"
        return row["wS_mean"].iloc[0]

    low_mean = w_s_mean_for_tau(low, 0.10)
    high_mean = w_s_mean_for_tau(high, 0.30)

    assert high_mean < low_mean


def test_sensitivity_does_not_touch_the_canonical_experiment_outputs():
    """The baseline experiment CSVs must be left alone by a sweep."""
    import inspect

    source = inspect.getsource(sr)
    assert "experiment_results.csv" not in source
    assert "experiment_summary.csv" not in source
    assert sr.DEFAULT_OUTPUT_DIR != er.Path("data/experiments")