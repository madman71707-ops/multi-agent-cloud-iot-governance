"""Attack-pattern experiments: TAPNA vs the canonical baselines.

Four deterministic threat schedules probe how each strategy responds to
different attack temporal patterns:

* ``Gradual attack``     - one threat per tick, severity ramping 0.1 -> 0.5.
* ``Short attack spike`` - a single short high-severity burst (default two
                           ticks around the midpoint).
* ``Repeated attacks``   - a high-severity threat every ``interval`` ticks
                           (default 10).
* ``Sustained attack``   - a high-severity threat on every tick.

Each scenario runs the same five strategies as the canonical experiment --
TAPNA, Static Weighted, Greedy Cost, Greedy Latency and NSGA-II -- under the
identical simulation loop as ``sim.experiment_runner``: same seeding, twin
construction, threat-injection arithmetic, candidate generation, action
scoring, TSS, CIGI and FairScore. The only differences are the threat
schedule (severity-valued instead of a set of ticks) and the extra
security-response columns.

With an empty threat schedule the per-tick rows are identical to
``experiment_runner._run_strategy`` apart from the schedule/response columns
(pinned by ``test_empty_schedule_matches_canonical_runner``).

Strategies never influence the environment: the twin dynamics, threat
injection, candidate generation and action scoring depend only on the seed
and the scenario schedule, and the negotiation itself is RNG-free. Each
(seed, scenario) environment is therefore simulated once, captured as a
trace, and replayed for all five strategies -- bit-for-bit identical to the
uncached per-strategy simulation (pinned by
``test_cached_replay_matches_uncached_execution``).

The canonical experiment, its code paths and its already-generated results
are untouched: this module only reads its public constants and helpers, and
``run_attack_patterns`` writes to its own output directory.
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.security_agent import compute_threat_severity
from digital_twin.action_generator import generate_candidate_actions
from digital_twin.telemetry_generator import generate_telemetry
from digital_twin.threat_injector import inject_threat
from digital_twin.twin import DigitalTwin
from negotiation.baselines import (
    greedy_cost_baseline,
    greedy_latency_baseline,
    nsga2_baseline,
    static_weighted_baseline,
)
from negotiation.cigi import compute_cigi
from negotiation.tapna import run_tapna
from sim.experiment_runner import (
    NUM_TICKS,
    SEEDS,
    STRATEGIES,
    _score_actions,
)

DEFENSIVE_ACTION_TYPES = ("isolate",)

DEFAULT_OUTPUT_DIR = Path("data/experiments_attack_patterns")

# Column order of the raw CSV; also the contract consumed by
# build_attack_response_summary.
RAW_COLUMNS = [
    "scenario",
    "strategy",
    "seed",
    "tick",
    "TSS",
    "wR",
    "wS",
    "wC",
    "CIGI",
    "U_R",
    "U_S",
    "U_C",
    "FairScore",
    "under_attack",
    "is_defensive_response",
    "security_response_gain",
    "response_lag",
    "chosen_action",
    "chosen_action_type",
    "target_device_id",
]


# ---------------------------------------------------------------------------
# Deterministic threat schedules
# ---------------------------------------------------------------------------

def gradual_attack_schedule(num_ticks, start_severity=0.1, end_severity=0.5):
    """A threat on every tick, severity ramping linearly up."""
    if num_ticks <= 0:
        return {}
    if num_ticks == 1:
        return {0: start_severity}
    return {
        tick: start_severity
        + (end_severity - start_severity) * (tick / (num_ticks - 1))
        for tick in range(num_ticks)
    }


def short_spike_schedule(num_ticks, spike_tick=None, spike_length=2,
                         spike_severity=0.5):
    """A single short high-severity burst (default: two ticks at the middle)."""
    if num_ticks <= 0:
        return {}
    if spike_tick is None:
        spike_tick = max(0, num_ticks // 2 - spike_length // 2)
    return {
        tick: spike_severity
        for tick in range(spike_tick, min(num_ticks, spike_tick + spike_length))
    }


def repeated_attack_schedule(num_ticks, interval=10, severity=0.5):
    """A high-severity threat every ``interval`` ticks (default 10)."""
    if num_ticks <= 0:
        return {}
    return {tick: severity for tick in range(0, num_ticks, interval)}


def sustained_attack_schedule(num_ticks, severity=0.5):
    """A high-severity threat on every tick."""
    if num_ticks <= 0:
        return {}
    return {tick: severity for tick in range(num_ticks)}


ATTACK_SCENARIOS = {
    "Gradual attack": gradual_attack_schedule,
    "Short attack spike": short_spike_schedule,
    "Repeated attacks": repeated_attack_schedule,
    "Sustained attack": sustained_attack_schedule,
}


# ---------------------------------------------------------------------------
# Strategy dispatch (mirrors experiment_runner._run_strategy)
# ---------------------------------------------------------------------------

def _run_strategy_tick(strategy, actions, U_R_list, U_S_list, U_C_list,
                       tss, prev_wS):
    """One negotiation step for any canonical strategy.

    Mirrors the dispatch inside ``experiment_runner._run_strategy``'s tick
    loop, including each baseline's reporting weights and TAPNA's prev_wS
    chaining. Returns ``(chosen_action, weights, next_prev_wS)``.
    """
    if strategy == "TAPNA":
        chosen_action, weights = run_tapna(
            actions, U_R_list, U_S_list, U_C_list, tss, prev_wS,
        )
        return chosen_action, weights, weights["wS"]
    if strategy == "Greedy Cost":
        chosen_action = greedy_cost_baseline(actions, U_C_list)
        return chosen_action, {"wR": 0.0, "wS": 0.0, "wC": 1.0}, prev_wS
    if strategy == "Greedy Latency":
        chosen_action = greedy_latency_baseline(actions, U_R_list)
        return chosen_action, {"wR": 1.0, "wS": 0.0, "wC": 0.0}, prev_wS
    if strategy == "NSGA-II":
        # Selection-only discrete NSGA-II over the same candidates and the
        # same objective values; equal weights only for reporting CIGI.
        chosen_action = nsga2_baseline(actions, U_R_list, U_S_list, U_C_list)
        return chosen_action, {"wR": 1 / 3, "wS": 1 / 3, "wC": 1 / 3}, prev_wS
    if strategy == "Static Weighted":
        chosen_action = static_weighted_baseline(
            actions, U_R_list, U_S_list, U_C_list,
        )
        return chosen_action, {"wR": 1 / 3, "wS": 1 / 3, "wC": 1 / 3}, prev_wS
    raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Environment capture + strategy replay
# ---------------------------------------------------------------------------

def _iterate_scenario_environment(threat_schedule, seed, num_ticks=NUM_TICKS):
    """Yield the strategy-independent environment for each tick.

    Everything up to (but excluding) the negotiation step is identical for
    all five strategies: twin dynamics, threat injection, candidate
    generation, action scoring and TSS depend only on the seed and the
    scenario schedule, never on the strategy. The negotiation is RNG-free,
    so capturing this environment once and replaying each strategy's
    negotiation on the trace reproduces the uncached per-strategy
    simulation bit for bit. Tick data must be treated as read-only by
    consumers: the same action dictionaries are shared across strategies.
    """
    random.seed(seed)
    np.random.seed(seed)

    twin = DigitalTwin(100)
    attack_started = None  # first tick carrying a nonzero-severity threat

    for tick in range(num_ticks):
        twin.tick()

        severity = threat_schedule.get(tick, 0.0)
        if severity > 0:
            if attack_started is None:
                attack_started = tick
            # Same shape as the canonical runner: one injection per device
            # plus one extra, each at the scenario severity.
            for device in twin.devices:
                inject_threat(twin, device.id, severity)
            inject_threat(twin, device.id, severity)

        twin_state = generate_telemetry(twin)
        resource_state = twin.get_state()
        actions = generate_candidate_actions(twin)
        U_R_list, U_S_list, U_C_list = _score_actions(
            actions, twin_state, resource_state
        )
        tss = compute_threat_severity(twin_state)

        yield {
            "actions": actions,
            "U_R_list": U_R_list,
            "U_S_list": U_S_list,
            "U_C_list": U_C_list,
            "tss": tss,
            "severity": severity,
            "is_attack_onset": attack_started == tick,
        }


def _negotiation_row(tick_data, chosen_action, weights, strategy, seed, tick):
    """Build one raw result row from one negotiated tick.

    Reads only the captured tick data and the negotiation outcome; must not
    mutate the tick data, which is shared across strategies.
    """
    actions = tick_data["actions"]
    U_R_list = tick_data["U_R_list"]
    U_S_list = tick_data["U_S_list"]
    U_C_list = tick_data["U_C_list"]

    chosen_index = actions.index(chosen_action)
    cigi = compute_cigi(
        U_R_list[chosen_index],
        U_S_list[chosen_index],
        U_C_list[chosen_index],
        weights,
    )

    chosen_u_r = U_R_list[chosen_index]
    chosen_u_s = U_S_list[chosen_index]
    chosen_u_c = U_C_list[chosen_index]
    # FairScore uses the same fixed 1/3 weights as the baseline
    # experiment, independent of the strategy's internal weights.
    fair_score = (
        (1 / 3) * chosen_u_r + (1 / 3) * chosen_u_s + (1 / 3) * chosen_u_c
    )

    # Security-response behavior: the chosen action's security utility
    # gain over the do-nothing candidate in the same tick.
    no_op_index = next(
        (i for i, action in enumerate(actions) if action["type"] == "no_op"),
        None,
    )
    if no_op_index is not None:
        security_response_gain = (
            U_S_list[chosen_index] - U_S_list[no_op_index]
        )
    else:
        security_response_gain = 0.0

    return {
        "scenario": tick_data["scenario"],
        "strategy": strategy,
        "seed": seed,
        "tick": tick,
        "TSS": tick_data["tss"],
        "wR": weights["wR"],
        "wS": weights["wS"],
        "wC": weights["wC"],
        "CIGI": cigi,
        "U_R": chosen_u_r,
        "U_S": chosen_u_s,
        "U_C": chosen_u_c,
        "FairScore": fair_score,
        "under_attack": tick_data["severity"] > 0,
        "is_defensive_response": chosen_action["type"] in DEFENSIVE_ACTION_TYPES,
        "security_response_gain": security_response_gain,
        "response_lag": 0,
        "chosen_action": chosen_action["id"],
        "chosen_action_type": chosen_action["type"],
        "target_device_id": chosen_action["target_device_id"],
    }


def _replay_strategy_on_trace(trace, strategy, scenario, seed):
    """Negotiate one strategy tick by tick over a captured environment trace.

    Both execution paths share this loop: the uncached path passes a live
    generator (environment re-simulated for the single strategy), the cached
    path passes a materialized trace (environment simulated once per seed
    and scenario, reused across strategies).
    """
    prev_wS = 1 / 3
    rows = []
    for tick, tick_data in enumerate(trace):
        data = dict(tick_data, scenario=scenario)
        chosen_action, weights, prev_wS = _run_strategy_tick(
            strategy,
            data["actions"],
            data["U_R_list"],
            data["U_S_list"],
            data["U_C_list"],
            data["tss"],
            prev_wS,
        )
        rows.append(_negotiation_row(
            data, chosen_action, weights, strategy, seed, tick,
        ))

    # Response lag: ticks from the first attack tick until the first
    # defensive response at or after it (scalar per seed run; recorded on
    # the attack-onset row, -1 when the run never responds defensively).
    attack_started = next(
        (row["tick"] for row in rows if row["under_attack"]), None,
    )
    if attack_started is not None:
        lag = -1
        for row in rows:
            if row["tick"] >= attack_started and row["is_defensive_response"]:
                lag = row["tick"] - attack_started
                break
        rows[attack_started]["response_lag"] = lag

    return rows


def _run_attack_scenario(strategy, scenario, threat_schedule, seed,
                         num_ticks=NUM_TICKS):
    """Simulate one seed under one attack pattern for one strategy.

    Uncached reference path: mirrors ``experiment_runner._run_strategy``
    exactly (identical seeding, twin construction, threat-injection
    arithmetic, scoring, CIGI and FairScore) with the environment
    re-simulated for this single strategy. ``run_attack_patterns`` instead
    captures the environment once per (seed, scenario) via
    ``_iterate_scenario_environment`` and replays every strategy on the
    trace; ``test_cached_replay_matches_uncached_execution`` pins the two
    paths to identical rows.
    """
    return _replay_strategy_on_trace(
        _iterate_scenario_environment(threat_schedule, seed, num_ticks),
        strategy,
        scenario,
        seed,
    )


def run_attack_patterns(seeds=None, scenarios=None, strategies=None,
                        num_ticks=NUM_TICKS, output_dir=DEFAULT_OUTPUT_DIR):
    """Run every scenario x strategy x seed and write raw + summary CSVs.

    The environment is simulated once per (seed, scenario) and shared
    across all five strategies; only the negotiation is re-run per
    strategy. Raw rows: one per (scenario, strategy, seed, tick). Summary:
    per (scenario, strategy) aggregates of FairScore / CIGI / TSS / wS plus
    the security-response metrics.
    """
    seeds = list(SEEDS if seeds is None else seeds)
    scenarios = list(ATTACK_SCENARIOS if scenarios is None else scenarios)
    strategies = list(STRATEGIES if strategies is None else strategies)

    unknown = set(strategies) - set(STRATEGIES)
    if unknown:
        raise ValueError(f"Unknown strategies: {sorted(unknown)}")
    unknown_scenarios = set(scenarios) - set(ATTACK_SCENARIOS)
    if unknown_scenarios:
        raise ValueError(f"Unknown scenarios: {sorted(unknown_scenarios)}")

    # Build all schedules up front; deterministic given num_ticks.
    schedules = {
        scenario: ATTACK_SCENARIOS[scenario](num_ticks)
        for scenario in scenarios
    }

    # The environment is identical for every strategy, so simulate each
    # (seed, scenario) once and replay the RNG-free negotiation per
    # strategy.
    traces = {
        (scenario, seed): list(_iterate_scenario_environment(
            schedules[scenario], seed, num_ticks,
        ))
        for scenario in scenarios
        for seed in seeds
    }

    rows = []
    for scenario in scenarios:
        for strategy in strategies:
            for seed in seeds:
                rows.extend(_replay_strategy_on_trace(
                    traces[(scenario, seed)], strategy, scenario, seed,
                ))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(rows, columns=RAW_COLUMNS)
    raw_path = output_dir / "attack_pattern_results.csv"
    results.to_csv(raw_path, index=False)
    print(f"Saved attack-pattern results to {raw_path}")

    summary_path = output_dir / "attack_pattern_summary.csv"
    build_attack_response_summary(results).to_csv(summary_path, index=False)
    print(f"Saved attack-pattern summary to {summary_path}")

    return results


def build_attack_response_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Per (scenario, strategy) aggregates of performance + response metrics.

    FairScore/CIGI/TSS/wS mean and std (ddof=1) over all ticks and seeds,
    plus the response-behavior metrics over attack ticks only:

    * ``defensive_response_rate``       - share of attack ticks answered
      with a defensive action type.
    * ``mean_security_response_gain``   - mean security-utility gain of the
      chosen action over the do-nothing candidate.
    * ``mean_response_lag``             - mean per-run response lag over runs
      that responded at all (never-defended runs carry lag -1 and are
      excluded here; ``never_responded_runs`` counts them).
    """
    group_columns = ["scenario", "strategy"]
    grouped = results.groupby(group_columns)

    perf = grouped[["FairScore", "CIGI", "TSS", "wS"]].agg(["mean", "std"])
    perf.columns = [f"{metric}_{stat}" for metric, stat in perf.columns]

    attacked = results[results["under_attack"]]
    attacked_grouped = attacked.groupby(group_columns)

    response = pd.DataFrame(index=perf.index)
    response["defensive_response_rate"] = (
        attacked_grouped["is_defensive_response"].mean()
    )
    response["mean_security_response_gain"] = (
        attacked_grouped["security_response_gain"].mean()
    )

    def _mean_lag(series):
        responded = series[series >= 0]
        return responded.mean() if len(responded) else float("nan")

    response["mean_response_lag"] = (
        attacked_grouped["response_lag"].agg(_mean_lag)
    )
    response["never_responded_runs"] = (
        attacked_grouped["response_lag"].apply(lambda s: int((s < 0).sum()))
    )

    summary = perf.join(response, how="outer").reset_index()

    scenario_order = list(ATTACK_SCENARIOS)
    strategy_order = [s for s in STRATEGIES if s in set(summary["strategy"])]
    summary["scenario"] = pd.Categorical(
        summary["scenario"], categories=scenario_order, ordered=True
    )
    summary["strategy"] = pd.Categorical(
        summary["strategy"], categories=strategy_order, ordered=True
    )
    summary = summary.sort_values(group_columns).reset_index(drop=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to run (default: the canonical 20 seeds).",
    )
    parser.add_argument(
        "--num-ticks",
        type=int,
        default=NUM_TICKS,
        help=f"Ticks per run (default: {NUM_TICKS}).",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        choices=sorted(ATTACK_SCENARIOS),
        help="Scenarios to run (default: all four).",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        choices=STRATEGIES,
        help="Strategies to run (default: all five canonical strategies).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Directory for the generated CSVs "
            f"(default: {DEFAULT_OUTPUT_DIR})."
        ),
    )
    args = parser.parse_args()

    run_attack_patterns(
        seeds=args.seeds,
        scenarios=args.scenarios,
        strategies=args.strategies,
        num_ticks=args.num_ticks,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
