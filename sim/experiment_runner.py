import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.resource_agent import score_resource
from agents.security_agent import compute_threat_severity, score_security
from agents.sustainability_agent import score_sustainability
from digital_twin.action_generator import generate_candidate_actions
from digital_twin.telemetry_generator import generate_telemetry
from digital_twin.threat_injector import inject_threat, schedule_random_threats
from digital_twin.twin import DigitalTwin
from negotiation.baselines import (
    greedy_cost_baseline,
    greedy_latency_baseline,
    nsga2_baseline,
    static_weighted_baseline,
)
from negotiation.cigi import compute_cigi
from negotiation.tapna import run_tapna, run_tapna_variant


SEEDS = [
    42, 123, 456, 789, 1001, 2024, 3141, 5000, 7777, 8888,
    9999, 11111, 22222, 33333, 44444, 55555, 66666, 77777, 88888, 99999,
]
NUM_TICKS = 100
STRATEGIES = [
    "TAPNA",
    "Greedy Cost",
    "Greedy Latency",
    "Static Weighted",
    "NSGA-II",
]

# ---------------------------------------------------------------------------
# TAPNA ablation variants.
#
# Each variant switches off a subset of TAPNA's three components so their
# individual contributions can be measured:
#
#   Pareto filtering    - restrict candidates to the non-dominated front.
#   Adaptive weight     - TSS-driven security weight.
#   Exponential decay   - relax the previous security weight in low-threat
#                         ticks.
#
# Kept out of STRATEGIES so the canonical five-strategy experiment, its row
# counts and its already-generated results stay exactly as they were. Pass
# include_ablations=True to main() (or set er.ALL_STRATEGIES) to run them.
# "TAPNA Full" is the unablated control and must reproduce the "TAPNA" rows.
# ---------------------------------------------------------------------------
TAPNA_ABLATION_VARIANTS = {
    "TAPNA Pareto-only": {
        "use_pareto": True,
        "use_adaptive_weight": False,
        "use_decay": False,
    },
    "TAPNA Adaptive-only": {
        "use_pareto": False,
        "use_adaptive_weight": True,
        "use_decay": False,
    },
    "TAPNA Pareto+Adaptive": {
        "use_pareto": True,
        "use_adaptive_weight": True,
        "use_decay": False,
    },
    "TAPNA Full": {
        "use_pareto": True,
        "use_adaptive_weight": True,
        "use_decay": True,
    },
}

ABLATION_STRATEGIES = list(TAPNA_ABLATION_VARIANTS)

ALL_STRATEGIES = STRATEGIES + ABLATION_STRATEGIES


def _run_tapna_ablation(strategy, actions, U_R_list, U_S_list, U_C_list,
                        tss, prev_wS):
    """Run one TAPNA ablation variant; returns (action, weights, prev_wS)."""
    flags = TAPNA_ABLATION_VARIANTS[strategy]
    chosen_action, weights = run_tapna_variant(
        actions,
        U_R_list,
        U_S_list,
        U_C_list,
        tss,
        prev_wS,
        **flags,
    )
    return chosen_action, weights, weights["wS"]


def _score_actions(actions, twin_state, resource_state):
    U_R_list = [
        score_resource(action, resource_state)
        for action in actions
    ]
    U_S_list = [
        score_security(action, twin_state)
        for action in actions
    ]
    U_C_list = [
        score_sustainability(action, twin_state)
        for action in actions
    ]
    return U_R_list, U_S_list, U_C_list


def _run_strategy(strategy, threat_schedule, seed):
    random.seed(seed)
    np.random.seed(seed)

    twin = DigitalTwin(100)
    prev_wS = 1 / 3
    rows = []

    for tick in range(NUM_TICKS):
        twin.tick()

        if tick in threat_schedule:
            for device in twin.devices:
                inject_threat(twin, device.id, 0.5)
            inject_threat(twin, device.id, 0.5)

        twin_state = generate_telemetry(twin)
        resource_state = twin.get_state()
        actions = generate_candidate_actions(twin)
        U_R_list, U_S_list, U_C_list = _score_actions(actions, twin_state, resource_state)
        tss = compute_threat_severity(twin_state)

        if strategy in TAPNA_ABLATION_VARIANTS:
            chosen_action, weights, prev_wS = _run_tapna_ablation(
                strategy,
                actions,
                U_R_list,
                U_S_list,
                U_C_list,
                tss,
                prev_wS,
            )
        elif strategy == "TAPNA":
            chosen_action, weights = run_tapna(
                actions,
                U_R_list,
                U_S_list,
                U_C_list,
                tss,
                prev_wS,
            )
            prev_wS = weights["wS"]
        elif strategy == "Greedy Cost":
            chosen_action = greedy_cost_baseline(actions, U_C_list)
            weights = {"wR": 0.0, "wS": 0.0, "wC": 1.0}
        elif strategy == "Greedy Latency":
            chosen_action = greedy_latency_baseline(actions, U_R_list)
            weights = {"wR": 1.0, "wS": 0.0, "wC": 0.0}
        elif strategy == "NSGA-II":
            # Selection-only discrete NSGA-II over the same candidates and
            # the same objective values. NSGA-II is preference-free; the
            # equal weights below are used only to report CIGI.
            chosen_action = nsga2_baseline(actions, U_R_list, U_S_list, U_C_list)
            weights = {"wR": 1 / 3, "wS": 1 / 3, "wC": 1 / 3}
        else:
            chosen_action = static_weighted_baseline(
                actions,
                U_R_list,
                U_S_list,
                U_C_list,
            )
            weights = {"wR": 1 / 3, "wS": 1 / 3, "wC": 1 / 3}

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
        # FairScore: fixed, strategy-independent evaluation of the chosen
        # action. Uses the same 1/3 weights for every strategy, regardless
        # of the (possibly dynamic) weights used for CIGI.
        fair_score = (
            (1 / 3) * chosen_u_r + (1 / 3) * chosen_u_s + (1 / 3) * chosen_u_c
        )

        rows.append(
            {
                "strategy": strategy,
                "seed": seed,
                "tick": tick,
                "TSS": tss,
                "wR": weights["wR"],
                "wS": weights["wS"],
                "wC": weights["wC"],
                "CIGI": cigi,
                "U_R": chosen_u_r,
                "U_S": chosen_u_s,
                "U_C": chosen_u_c,
                "FairScore": fair_score,
                "response_lag": 0,
                "chosen_action": chosen_action["id"],
                "chosen_action_type": chosen_action["type"],
                "target_device_id": chosen_action["target_device_id"],
            }
        )

    return rows


def build_summary(
    results: pd.DataFrame,
    order: list | None = None,
) -> pd.DataFrame:
    """Aggregate per-tick results into per-strategy summary statistics.

    Uses pandas mean/std (sample standard deviation, ddof=1) over every tick
    of every seed. Rows follow the run order given by ``order``, defaulting to
    STRATEGIES followed by ABLATION_STRATEGIES. Strategies absent from
    ``results`` are skipped, so this is unchanged for the canonical run.
    """
    if order is None:
        order = ALL_STRATEGIES

    available = set(results["strategy"])
    present = [strategy for strategy in order if strategy in available]
    summary = (
        results.groupby("strategy")[["FairScore", "CIGI", "TSS"]]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return (
        summary.reset_index()
        .set_index("strategy")
        .reindex(present)
        .reset_index()
    )


def main(include_ablations: bool = False, output_dir: str | Path | None = None):
    """Run the baseline experiment and write the raw and summary CSVs.

    ``include_ablations`` adds the four TAPNA ablation variants after the
    canonical five strategies. It defaults to False so the standard run, its
    row counts and its previously generated CSVs are unchanged.
    """
    strategies = ALL_STRATEGIES if include_ablations else STRATEGIES
    threat_schedule = {50}
    rows = []

    for run_number, seed in enumerate(SEEDS, start=1):
        random.seed(seed)
        np.random.seed(seed)

        for strategy in strategies:
            rows.extend(_run_strategy(strategy, threat_schedule, seed))

    if output_dir is None:
        output_dir = Path("data/experiments")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(rows)
    output_path = output_dir / "experiment_results.csv"
    results.to_csv(output_path, index=False)
    print(f"Saved experiment results to {output_path}")

    summary_path = output_dir / "experiment_summary.csv"
    build_summary(results, order=strategies).to_csv(summary_path, index=False)
    print(f"Saved experiment summary to {summary_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-ablations",
        action="store_true",
        help="Also run the four TAPNA ablation variants.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the generated CSVs (default: data/experiments).",
    )
    args = parser.parse_args()
    main(
        include_ablations=args.include_ablations,
        output_dir=args.output_dir,
    )
