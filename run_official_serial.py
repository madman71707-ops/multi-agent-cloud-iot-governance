"""Serial runner for the official 4 x 8 x 20 x 100 v3 closed-loop grid.



Uses sim.attack_patterns_v3 (the frozen calibration config) and writes

experiment_results_v3.csv, outcome_metrics_v3.csv and outcome_summary_v3.csv

to data/experiments_v3/closed_loop/. It honours per-seed checkpoints so an

interrupted run resumes rather than re-starts.

"""

import os

import sys

import time

from pathlib import Path



sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



import pandas as pd



import sim.attack_patterns_v3 as ap3

from sim.attack_patterns_v2 import build_outcome_summary as _bos



OUTDIR = Path("data/experiments_v3/closed_loop")

RAW_PATH = OUTDIR / "experiment_results_v3.csv"

OUTCOME_PATH = OUTDIR / "outcome_metrics_v3.csv"

SUMMARY_PATH = OUTDIR / "outcome_summary_v3.csv"



SEEDS = ap3._official_seeds()

SCENARIOS = ap3.SCENARIOS_V3

STRATS = ap3.STRATEGIES_V3

KAPPA = 2.0

NUM_TICKS = ap3._NUM_TICKS





def _load_done():

    if not (RAW_PATH.exists() and OUTCOME_PATH.exists()):

        return set()

    try:

        prev = pd.read_csv(OUTCOME_PATH)

    except Exception:

        return set()

    if prev.empty:

        return set()

    wanted = set(STRATS)

    done = {

        (str(s), int(d))

        for (s, d), grp in prev.groupby(["scenario", "seed"])

        if set(grp["strategy"]) == wanted

    }

    return done





def _checkpoint(rows, outcomes, done):

    results = pd.DataFrame(rows)

    outcome_df = pd.DataFrame(outcomes)

    summary_df = _bos(outcome_df, metric_columns=ap3.OUTCOME_METRIC_COLUMNS_V3)

    for frame, path in (

        (results, RAW_PATH),

        (outcome_df, OUTCOME_PATH),

        (summary_df, SUMMARY_PATH),

    ):

        tmp = path.with_suffix(path.suffix + ".tmp")

        frame.to_csv(tmp, index=False)

        os.replace(tmp, path)





def main():

    OUTDIR.mkdir(parents=True, exist_ok=True)

    done = _load_done()

    if done:

        print(f"[grid] resume: {len(done)} (scenario, seed) pairs done")



    rows, outcomes, done_pairs = [], [], done

    tasks = [

        (scenario, seed)

        for scenario in SCENARIOS

        for seed in SEEDS

        if (scenario, seed) not in done_pairs

    ]

    total = len(tasks)

    print(f"official v3 grid: {total} runs left "

          f"({len(SCENARIOS)} scenarios x {len(SEEDS)} seeds x {len(STRATS)} strategies)")

    print(f"frozen kappa={KAPPA}, seeds={SEEDS}")

    started = time.perf_counter()



    for index, (scenario, seed) in enumerate(tasks, start=1):

        for strategy in STATS:
            _, outcome = ap3.run_v3_closed_loop(
                strategy, scenario, seed, kappa=KAPPA,
                num_ticks=NUM_TICKS, us_mode="v2", ur_mode="v3",
            )