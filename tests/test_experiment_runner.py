import math

import pandas as pd
import pytest

from sim import experiment_runner as er


STRATEGIES = [
    "TAPNA",
    "Greedy Cost",
    "Greedy Latency",
    "Static Weighted",
    "NSGA-II",
]

EXPECTED_SEEDS = [
    42, 123, 456, 789, 1001, 2024, 3141, 5000, 7777, 8888,
    9999, 11111, 22222, 33333, 44444, 55555, 66666, 77777, 88888, 99999,
]


def test_runner_is_configured_for_20_seeds_100_ticks_and_5_strategies():
    assert er.SEEDS == EXPECTED_SEEDS
    assert len(er.SEEDS) == 20
    assert er.NUM_TICKS == 100
    assert er.STRATEGIES == STRATEGIES


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_seed_is_propagated_to_every_row(monkeypatch, strategy):
    monkeypatch.setattr(er, "NUM_TICKS", 1)

    rows = er._run_strategy(strategy, set(), 789)

    assert len(rows) == 1
    assert rows[0]["seed"] == 789


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_runner_logs_chosen_objectives_and_fixed_weight_fairscore(
    monkeypatch, strategy
):
    monkeypatch.setattr(er, "NUM_TICKS", 1)

    rows = er._run_strategy(strategy, set(), 42)

    assert len(rows) == 1
    row = rows[0]

    for field in ("U_R", "U_S", "U_C", "FairScore"):
        assert field in row
        assert math.isfinite(row[field])
        assert 0.0 <= row[field] <= 1.0

    # FairScore always uses the fixed 1/3 weights, independent of the
    # strategy's own weights (TAPNA's dynamic wS affects only CIGI).
    expected = (
        (1 / 3) * row["U_R"] + (1 / 3) * row["U_S"] + (1 / 3) * row["U_C"]
    )
    assert row["FairScore"] == pytest.approx(expected)


def test_build_summary_aggregates_mean_and_std_per_strategy():
    results = pd.DataFrame(
        [
            {"strategy": "TAPNA", "FairScore": 0.2, "CIGI": 0.4, "TSS": 0.05},
            {"strategy": "TAPNA", "FairScore": 0.4, "CIGI": 0.6, "TSS": 0.15},
            {"strategy": "NSGA-II", "FairScore": 0.5, "CIGI": 0.5, "TSS": 0.1},
            {"strategy": "NSGA-II", "FairScore": 0.7, "CIGI": 0.7, "TSS": 0.3},
        ]
    )

    summary = er.build_summary(results)

    assert list(summary.columns) == [
        "strategy",
        "FairScore_mean",
        "FairScore_std",
        "CIGI_mean",
        "CIGI_std",
        "TSS_mean",
        "TSS_std",
    ]
    # Row order follows the run order in er.STRATEGIES.
    assert list(summary["strategy"]) == ["TAPNA", "NSGA-II"]

    # Hand-computed sample statistics (ddof=1) for two values a, b:
    #   mean = (a + b) / 2,  std = |a - b| / sqrt(2)
    tapna = summary[summary["strategy"] == "TAPNA"].iloc[0]
    assert tapna["FairScore_mean"] == pytest.approx((0.2 + 0.4) / 2)
    assert tapna["FairScore_std"] == pytest.approx(abs(0.4 - 0.2) / math.sqrt(2))
    assert tapna["CIGI_mean"] == pytest.approx((0.4 + 0.6) / 2)
    assert tapna["CIGI_std"] == pytest.approx(abs(0.6 - 0.4) / math.sqrt(2))
    assert tapna["TSS_mean"] == pytest.approx((0.05 + 0.15) / 2)
    assert tapna["TSS_std"] == pytest.approx(abs(0.15 - 0.05) / math.sqrt(2))

    nsga2 = summary[summary["strategy"] == "NSGA-II"].iloc[0]
    assert nsga2["FairScore_mean"] == pytest.approx(0.6)
    assert nsga2["FairScore_std"] == pytest.approx(abs(0.7 - 0.5) / math.sqrt(2))


def test_main_writes_raw_and_summary_csvs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(er, "SEEDS", [42, 123])
    monkeypatch.setattr(er, "NUM_TICKS", 1)

    er.main()

    raw_path = tmp_path / "data" / "experiments" / "experiment_results.csv"
    summary_path = tmp_path / "data" / "experiments" / "experiment_summary.csv"
    assert raw_path.is_file()
    assert summary_path.is_file()

    raw = pd.read_csv(raw_path)
    for field in ("strategy", "seed", "tick", "CIGI", "TSS", "U_R", "U_S", "U_C", "FairScore"):
        assert field in raw.columns
    # One row per seed x strategy x tick, with the seed values propagated.
    assert len(raw) == 2 * 5 * 1
    assert set(raw["seed"]) == {42, 123}

    summary = pd.read_csv(summary_path)
    assert list(summary.columns) == [
        "strategy",
        "FairScore_mean",
        "FairScore_std",
        "CIGI_mean",
        "CIGI_std",
        "TSS_mean",
        "TSS_std",
    ]
    assert list(summary["strategy"]) == STRATEGIES
    # Two seeds => two rows per strategy => all statistics are defined.
    for column in summary.columns[1:]:
        assert summary[column].notna().all()

    # Summary statistics must match an independent aggregation of the raw CSV.
    expected = (
        raw.groupby("strategy")[["FairScore", "CIGI", "TSS"]]
        .agg(["mean", "std"])
        .reindex(STRATEGIES)
    )
    for column in summary.columns[1:]:
        metric, stat = column.rsplit("_", 1)
        assert summary[column].tolist() == pytest.approx(
            expected[(metric, stat)].tolist()
        )
