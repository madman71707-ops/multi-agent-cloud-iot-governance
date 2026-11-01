import numpy as np

from agents.security_agent import compute_threat_severity
from sim import experiment_runner as er


STATE = {
    "avg_trust_score": 0.9,
    "avg_network_load": 0.35,
    "cpu_load": 0.4,
    "memory_load": 0.5,
    "network_load": 0.3,
}

# Columns excluded from value comparisons: `chosen_action` is a uuid4 and
# `target_device_id` comes from an unseeded Faker instance - pure identity
# strings that carry no experimental values.
IDENTITY_COLUMNS = ("chosen_action", "target_device_id")


def _severity_values(count=5):
    return [compute_threat_severity(STATE) for _ in range(count)]


def test_same_seed_produces_identical_threat_severity():
    np.random.seed(42)
    first = _severity_values()

    np.random.seed(42)
    second = _severity_values()

    assert first == second


def test_different_seeds_can_produce_different_threat_severity():
    np.random.seed(42)
    first = _severity_values()

    np.random.seed(12345)
    second = _severity_values()

    assert first != second


def _metric_rows(seed, ticks=3, strategy="TAPNA"):
    rows = er._run_strategy(strategy, set(), seed)
    return [
        {key: value for key, value in row.items() if key not in IDENTITY_COLUMNS}
        for row in rows
    ]


def test_same_experiment_seed_gives_identical_metric_results(monkeypatch):
    monkeypatch.setattr(er, "NUM_TICKS", 3)

    first = _metric_rows(seed=42)
    second = _metric_rows(seed=42)

    assert first == second
    # Sanity: the compared rows really contain the seeded metrics.
    assert any("TSS" in row and "U_S" in row for row in first)


def test_different_experiment_seeds_can_give_different_metric_results(
    monkeypatch,
):
    monkeypatch.setattr(er, "NUM_TICKS", 3)

    first = _metric_rows(seed=42)
    second = _metric_rows(seed=12345)

    assert first != second


def test_same_seed_same_result_across_strategies(monkeypatch):
    """Every strategy is independently reproducible from its experiment seed."""
    monkeypatch.setattr(er, "NUM_TICKS", 2)

    for strategy in er.STRATEGIES:
        assert _metric_rows(seed=7, strategy=strategy) == _metric_rows(
            seed=7, strategy=strategy
        )
