"""TAPNA parameter sensitivity experiments.

Sweeps the two adaptive-weight parameters of Full TAPNA:

* ``tau``          - the TSS threshold separating the threat ramp from the
                     decay branch (``compute_dynamic_weight``'s ``tau``).
* ``decay_lambda`` - the exponential decay rate applied to the previous
                     security weight in the low-threat branch.

Full TAPNA (tau=0.15, decay_lambda=0.3) is left untouched: this module only
re-runs it at other parameter values, and the canonical configuration is
included in the sweep so the baseline sits in the same table.

Every configuration writes both a raw per-tick CSV and a summary CSV. The
summary reports FairScore/CIGI/wS mean and std per configuration, plus the
parameter value used, so each sweep can be read directly.

The sweep parameters only affect the negotiation step
(``run_tapna_variant``), which is RNG-free; the twin dynamics, threat
injection, candidate generation and action scoring are identical across
configurations. Each seed's environment is therefore simulated once,
captured as a trace, and replayed for every configuration -- bit-for-bit
identical to re-simulating per configuration (pinned by
``test_cached_replay_matches_uncached_execution``).
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
from negotiation.cigi import compute_cigi
from negotiation.tapna import run_tapna_variant
from sim.experiment_runner import NUM_TICKS, SEEDS, _score_actions


# The canonical Full TAPNA parameter values. Kept as explicit constants so the
# default configuration is identifiable inside a sweep of other values.
DEFAULT_TAU = 0.15
DEFAULT_DECAY_LAMBDA = 0.3

TAU_VALUES = [0.10, 0.15, 0.20, 0.40]
DECAY_LAMBDA_VALUES = [0.10, 0.30, 0.50, 0.70]

DEFAULT_OUTPUT_DIR = Path("data/experiments_sensitivity")

# Metrics aggregated per configuration. ``wS`` is included because the
# adaptive weight is the quantity these parameters actually move.
SUMMARY_METRICS = ["FairScore", "CIGI", "wS"]


def _iterate_environment_ticks(threat_schedule, seed, num_ticks=NUM_TICKS):
    """Yield the config-independent environment for each tick of one seed.

    Everything up to (but excluding) the negotiation step is identical
    across all tau / decay_lambda configurations: twin dynamics, threat
    injection, candidate generation, action scoring and TSS depend only on
    the seed and its RNG stream, never on the sweep parameters. The
    negotiation (``run_tapna_variant``) is RNG-free, so capturing this
    environment once and replaying the negotiation per configuration
    reproduces the uncached per-configuration simulation bit for bit.
    """
    random.seed(seed)
    np.random.seed(seed)

    twin = DigitalTwin(100)

    for tick in range(num_ticks):
        twin.tick()

        if tick in threat_schedule:
            for device in twin.devices:
                inject_threat(twin, device.id, 0.5)
            inject_threat(twin, device.id, 0.5)

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
        }


def _replay_negotiation(tick_data, tau, decay_lambda, prev_wS):
    """Run one Full TAPNA negotiation step on a captured environment tick."""
    return run_tapna_variant(
        tick_data["actions"],
        tick_data["U_R_list"],
        tick_data["U_S_list"],
        tick_data["U_C_list"],
        tick_data["tss"],
        prev_wS,
        use_pareto=True,
        use_adaptive_weight=True,
        use_decay=True,
        tau=tau,
        decay_lambda=decay_lambda,
    )


def _tapna_result_row(tick_data, chosen_action, weights, tau, decay_lambda,
                      seed, tick):
    """Build one raw result row from a negotiated tick."""
    actions = tick_data["actions"]
    chosen_index = actions.index(chosen_action)
    cigi = compute_cigi(
        tick_data["U_R_list"][chosen_index],
        tick_data["U_S_list"][chosen_index],
        tick_data["U_C_list"][chosen_index],
        weights,
    )

    chosen_u_r = tick_data["U_R_list"][chosen_index]
    chosen_u_s = tick_data["U_S_list"][chosen_index]
    chosen_u_c = tick_data["U_C_list"][chosen_index]
    # FairScore uses the same fixed 1/3 weights as the baseline
    # experiment, independent of the adaptive weights used for CIGI.
    fair_score = (
        (1 / 3) * chosen_u_r + (1 / 3) * chosen_u_s + (1 / 3) * chosen_u_c
    )

    return {
        # `parameter` and `value` identify the sweep this row belongs to and
        # are stamped by the caller, which knows the sweep axis.
        "parameter": None,
        "value": None,
        "tau": tau,
        "decay_lambda": decay_lambda,
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
        "chosen_action_type": chosen_action["type"],
    }


def _run_tapna_on_ticks(ticks, seed, tau, decay_lambda):
    """Negotiate Full TAPNA tick by tick over a tick iterable.

    Both execution paths share this loop: the uncached path passes a live
    generator (environment re-simulated for the single configuration), the
    cached path passes a materialized environment trace (environment
    simulated once per seed, reused across configurations).
    """
    prev_wS = 1 / 3
    rows = []
    for tick, tick_data in enumerate(ticks):
        chosen_action, weights = _replay_negotiation(
            tick_data, tau, decay_lambda, prev_wS
        )
        prev_wS = weights["wS"]
        rows.append(_tapna_result_row(
            tick_data, chosen_action, weights, tau, decay_lambda, seed, tick
        ))
    return rows


def _run_tapna_sensitivity(threat_schedule, seed, tau, decay_lambda,
                           num_ticks=NUM_TICKS):
    """Run Full TAPNA at one (tau, decay_lambda) pair, returning raw rows.

    Mirrors ``experiment_runner._run_strategy`` exactly for Full TAPNA: same
    seeding, same twin construction, same threat injection, same candidate
    generation and scoring, same CIGI and FairScore definitions. The only
    difference is that tau and decay_lambda are passed through instead of
    taking their defaults.

    This is the uncached reference path: the environment is re-simulated
    for this single configuration. ``run_sensitivity`` instead captures
    the environment once per seed via ``_iterate_environment_ticks`` and
    replays the negotiation for every configuration;
    ``test_cached_replay_matches_uncached_execution`` pins the two paths
    to identical rows.
    """
    return _run_tapna_on_ticks(
        _iterate_environment_ticks(threat_schedule, seed, num_ticks),
        seed,
        tau,
        decay_lambda,
    )


def build_sensitivity_configurations(tau_values=None, lambda_values=None,
                                    include_baseline=True):
    """Build the list of (parameter, value, tau, decay_lambda) configurations.

    ``include_baseline`` adds the canonical (tau=0.15, decay_lambda=0.3)
    configuration once, marked as the baseline, so the sweep can be read
    against it. It is deduplicated against any sweep value that coincides with
    the canonical parameters.
    """
    tau_values = list(TAU_VALUES if tau_values is None else tau_values)
    lambda_values = list(
        DECAY_LAMBDA_VALUES if lambda_values is None else lambda_values
    )

    configurations = []
    seen = set()

    for tau in tau_values:
        key = ("tau", tau, DEFAULT_DECAY_LAMBDA)
        if key not in seen:
            seen.add(key)
            configurations.append({
                "parameter": "tau",
                "value": tau,
                "tau": tau,
                "decay_lambda": DEFAULT_DECAY_LAMBDA,
                "is_baseline": tau == DEFAULT_TAU,
            })

    for decay_lambda in lambda_values:
        key = ("decay_lambda", decay_lambda, DEFAULT_TAU)
        if key not in seen:
            seen.add(key)
            configurations.append({
                "parameter": "decay_lambda",
                "value": decay_lambda,
                "tau": DEFAULT_TAU,
                "decay_lambda": decay_lambda,
                "is_baseline": decay_lambda == DEFAULT_DECAY_LAMBDA,
            })

    if include_baseline:
        key = ("baseline", DEFAULT_TAU, DEFAULT_DECAY_LAMBDA)
        if not any(
            config["tau"] == DEFAULT_TAU
            and config["decay_lambda"] == DEFAULT_DECAY_LAMBDA
            and config["parameter"] != "baseline"
            for config in configurations
        ):
            configurations.append({
                "parameter": "baseline",
                "value": DEFAULT_TAU,
                "tau": DEFAULT_TAU,
                "decay_lambda": DEFAULT_DECAY_LAMBDA,
                "is_baseline": True,
            })

    return configurations


def build_sensitivity_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw per-tick rows into one row per configuration.

    Groups by the full (parameter, value, tau, decay_lambda) identity so the
    tau sweep and the decay_lambda sweep stay separate even where they share
    a configuration. Uses pandas mean/std (ddof=1).
    """
    group_columns = ["parameter", "value", "tau", "decay_lambda"]
    summary = (
        results.groupby(group_columns)[SUMMARY_METRICS]
        .agg(["mean", "std"])
    )
    summary.columns = [
        f"{metric}_{stat}" for metric, stat in summary.columns
    ]
    summary = (
        summary.reset_index()
        .sort_values(["parameter", "value"])
        .reset_index(drop=True)
    )
    # Row counts per configuration, so an incomplete run is visible.
    counts = (
        results.groupby(group_columns).size().rename("n_rows").reset_index()
    )
    return summary.merge(counts, on=group_columns, how="left")


def run_sensitivity(seeds=None, tau_values=None, lambda_values=None,
                    num_ticks=NUM_TICKS, threat_schedule=None,
                    include_baseline=True, output_dir=DEFAULT_OUTPUT_DIR):
    """Run the tau and decay_lambda sweeps and write raw + summary CSVs.

    The environment is simulated once per seed and shared across all
    configurations; only the negotiation is re-run per configuration.
    """
    seeds = list(SEEDS if seeds is None else seeds)
    if threat_schedule is None:
        threat_schedule = {50}

    configurations = build_sensitivity_configurations(
        tau_values=tau_values,
        lambda_values=lambda_values,
        include_baseline=include_baseline,
    )

    # The environment is identical for every configuration, so simulate
    # each seed once and replay the RNG-free negotiation per configuration.
    traces = {
        seed: list(_iterate_environment_ticks(threat_schedule, seed, num_ticks))
        for seed in seeds
    }

    rows = []
    for config in configurations:
        for seed in seeds:
            config_rows = _run_tapna_on_ticks(
                traces[seed],
                seed,
                config["tau"],
                config["decay_lambda"],
            )
            for row in config_rows:
                # Stamp the sweep identity onto every row.
                row["parameter"] = config["parameter"]
                row["value"] = config["value"]
                row["is_baseline"] = config["is_baseline"]
            rows.extend(config_rows)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(rows)
    raw_path = output_dir / "sensitivity_results.csv"
    results.to_csv(raw_path, index=False)
    print(f"Saved sensitivity results to {raw_path}")

    summary_path = output_dir / "sensitivity_summary.csv"
    build_sensitivity_summary(results).to_csv(summary_path, index=False)
    print(f"Saved sensitivity summary to {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--taus",
        type=float,
        nargs="+",
        default=None,
        help="Values of tau to sweep (default: 0.10 0.15 0.20 0.40).",
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Values of decay_lambda to sweep "
            "(default: 0.10 0.30 0.50 0.70)."
        ),
    )
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
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Directory for the generated CSVs "
            f"(default: {DEFAULT_OUTPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Omit the canonical tau=0.15, decay_lambda=0.3 configuration.",
    )
    args = parser.parse_args()

    run_sensitivity(
        seeds=args.seeds,
        tau_values=args.taus,
        lambda_values=args.lambdas,
        num_ticks=args.num_ticks,
        include_baseline=not args.no_baseline,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()