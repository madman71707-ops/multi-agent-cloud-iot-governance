"""Phase 2 experiment runner: v2 twin dynamics + v2 utilities + closed loop.

Grid (spec 2e): 4 scenarios x 6 strategies x 20 official seeds x 100 ticks,
run twice - once with ``closed_loop=True`` (actions change the environment)
and once with ``closed_loop=False`` (open loop, the legacy causal structure)
so the closed-loop effect can be shown separately. Output goes to
data/experiments_v2/{closed_loop,open_loop}/; the legacy
data/experiments/ CSVs are never touched.

Strategies (v2 labels):
    TAPNA            canonical frozen negotiation (negotiation/tapna.run_tapna)
    Greedy Cost      max U_C
    Greedy Latency   max U_R
    Static Weighted  fixed 1/3-1/3-1/3 weighted sum
    ParetoCrowding   NSGA-II-style selection rule, renamed (see
                     negotiation/baselines_v2.py - same frozen helpers)
    ThreatRuleOnly   isolate while TSS > tau, else Static Weighted

Environment protocol per run: ``random.seed(seed)`` and ``np.random.seed``
are set exactly once, then DigitalTwinV2 (device draws) and
build_v2_schedule (onset/intensity jitter) consume the seeded numpy stream
in that fixed order, followed by per-tick walk/propagation draws.

Closed loop (per tick): advance -> telemetry -> candidates -> TSS -> score
(us_mode="v2") -> negotiate -> apply the chosen action, so its effects land
on the NEXT tick. Because actions change the state, the environment trace is
strategy-dependent and is simulated per strategy; after the first differing
action the noise streams diverge as well (initial devices, schedule and
onset stay seed-identical).

Open loop: no action is ever applied, so the per-(scenario, seed)
environment trace (candidates + utilities + TSS + telemetry) is
strategy-INDEPENDENT: it is computed once and replayed for all six
strategies, which therefore face literally identical conditions.

Outcome metrics (spec 2d) per strategy x scenario x seed - see
compute_outcome_metrics. The legacy "defensive response rate" survives as
``isolate_share`` (share of ALL ticks whose chosen action type was isolate);
it is a behaviour descriptor, NOT an outcome metric, and must not be read as
one in the paper.

CLI:
    python -m sim.attack_patterns_v2 --smoke           # calibration checks
    python -m sim.attack_patterns_v2 --grid closed_loop
    python -m sim.attack_patterns_v2 --grid open_loop
    python -m sim.attack_patterns_v2 --grid both
    python -m sim.attack_patterns_v2 --sensitivity
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.security_agent import compute_threat_severity
from digital_twin.twin_v2 import (
    DigitalTwinV2,
    build_v2_schedule,
    deep_update,
    generate_candidate_actions_v2,
    get_closed_loop_config,
    get_twin_v2_config,
)
from negotiation.baselines import (
    greedy_cost_baseline,
    greedy_latency_baseline,
    static_weighted_baseline,
)
from negotiation.baselines_v2 import (
    paretocrowding_baseline,
    threat_rule_only_baseline,
)
from negotiation.cigi import compute_cigi
from negotiation.tapna import run_tapna
from sim.scoring_v2 import (
    score_actions_v2,
    identical_us_fraction,
    _tss_batch,
)
import sim.scoring_v2 as sim_scoring_v2  # batched-TSS helper for trace-level TSS

# Official evaluation seeds - imported so the v2 grid provably uses the same
# list as every legacy experiment. Imported late to keep the module docstring
# and shared helpers free of a project-root sys.path dependency.
from sim.experiment_runner import SEEDS as SEEDS_OFFICIAL  # noqa: E402

SCENARIOS_V2 = [
    "Gradual attack",
    "Short attack spike",
    "Repeated attacks",
    "Sustained attack",
]
STRATEGIES_V2 = [
    "TAPNA",
    "Greedy Cost",
    "Greedy Latency",
    "Static Weighted",
    "ParetoCrowding",
    "ThreatRuleOnly",
]
NUM_TICKS_V2 = 100

# Calibration / tuning seeds ONLY (never used for the reported grid).
TUNING_SEEDS = list(range(1000, 1010))
SENSITIVITY_SEEDS = list(range(1000, 1005))

_THIRDS = {"wR": 1 / 3, "wS": 1 / 3, "wC": 1 / 3}

ACTION_TYPES = ["isolate", "migrate", "scale_down", "no_op"]

# Consecutive ticks required to declare an attack bite or a containment
# (see compute_outcome_metrics): baseline TSS noise straddles tau=0.15, so
# single-tick crossings are not evidence of anything.
_CONTAINMENT_RUN = 3

OUTCOME_METRIC_COLUMNS = [
    "time_to_containment",
    "contained",
    "cumulative_compromised_device_ticks",
    "peak_TSS",
    "area_under_TSS",
    "trust_recovery_time",
    "trust_recovered",
    "cumulative_energy",
    "cumulative_cost",
    "FairScore_mean",
    "mean_U_S_attack",
    "attack_isolate_share",
    "attack_migrate_share",
    "attack_scale_down_share",
    "attack_no_op_share",
    "normal_isolate_share",
    "normal_migrate_share",
    "normal_scale_down_share",
    "normal_no_op_share",
    "isolate_share",
    "us_identical_fraction",
]


# ---------------------------------------------------------------------------
# Negotiation dispatch (pure: no RNG, no environment access)
# ---------------------------------------------------------------------------
def _negotiate(strategy, actions, U_R_list, U_S_list, U_C_list, tss, prev_wS, tau):
    """One negotiation; returns ``(chosen_action, weights)``."""
    if strategy == "TAPNA":
        chosen_action, weights = run_tapna(
            actions, U_R_list, U_S_list, U_C_list, tss, prev_wS
        )
    elif strategy == "Greedy Cost":
        chosen_action = greedy_cost_baseline(actions, U_C_list)
        weights = {"wR": 0.0, "wS": 0.0, "wC": 1.0}
    elif strategy == "Greedy Latency":
        chosen_action = greedy_latency_baseline(actions, U_R_list)
        weights = {"wR": 1.0, "wS": 0.0, "wC": 0.0}
    elif strategy == "Static Weighted":
        chosen_action = static_weighted_baseline(
            actions, U_R_list, U_S_list, U_C_list
        )
        weights = dict(_THIRDS)
    elif strategy == "ParetoCrowding":
        # Selection-only NSGA-II rule; preference-free, so equal thirds are
        # used only to report CIGI (same convention as the legacy NSGA-II).
        chosen_action = paretocrowding_baseline(
            actions, U_R_list, U_S_list, U_C_list
        )
        weights = dict(_THIRDS)
    elif strategy == "ThreatRuleOnly":
        chosen_action = threat_rule_only_baseline(
            actions, U_R_list, U_S_list, U_C_list, tss, tau=tau
        )
        weights = dict(_THIRDS)
    else:
        raise ValueError(f"Unknown v2 strategy: {strategy}")
    return chosen_action, weights


# ---------------------------------------------------------------------------
# Environment step (shared by the closed-loop and open-loop paths)
# ---------------------------------------------------------------------------
def _environment_step(twin, intensity, cl_cfg, tick, us_mode, config):
    """Advance one tick and observe; returns the per-tick negotiation context.

    The per-tick TSS is computed by :func:`sim.scoring_v2._tss_batch` (one
    batched IsolationForest ``decision_function`` call) instead of
    ``compute_threat_severity`` (one call per tick) so the 100-tick open-loop
    trace is fast enough to run on this machine. The batched helper builds one
    ``_telemetry_frame`` per input state and stacks them (that advances the
    rng exactly as the per-tick sequential path would), then calls the
    estimator once; the resulting TSS values match
    ``compute_threat_severity`` within float tolerance.
    """
    twin.advance(intensity, closed_loop_cfg=cl_cfg)
    twin_state = twin.get_telemetry()
    actions = generate_candidate_actions_v2(twin, tick)
    tss = float(_tss_batch([twin_state]).ravel()[0])
    U_R_list, U_S_list, U_C_list = score_actions_v2(
        actions,
        twin,
        twin_state,
        twin_state,  # score_resource reads only the avg_* keys of this dict
        tss,
        us_mode=us_mode,
        config=config,
    )
    return {
        "tick": tick,
        "intensity": intensity,
        "twin_state": twin_state,
        "actions": actions,
        "tss": tss,
        "U_R": U_R_list,
        "U_S": U_S_list,
        "U_C": U_C_list,
        "compromised": twin.compromised_count,
        "isolated": twin.isolated_count,
    }


def _tick_row(strategy, scenario, seed, step, chosen, weights, chosen_index,
              closed_loop, us_mode, info):
    u_r = step["U_R"][chosen_index]
    u_s = step["U_S"][chosen_index]
    u_c = step["U_C"][chosen_index]
    telemetry = step["twin_state"]
    return {
        "strategy": strategy,
        "scenario": scenario,
        "seed": seed,
        "tick": step["tick"],
        "attack_intensity": step["intensity"],
        "TSS": step["tss"],
        "wR": weights["wR"],
        "wS": weights["wS"],
        "wC": weights["wC"],
        "CIGI": compute_cigi(u_r, u_s, u_c, weights),
        "U_R": u_r,
        "U_S": u_s,
        "U_C": u_c,
        # FairScore: fixed, strategy-independent 1/3-1/3-1/3 evaluation.
        "FairScore": (1 / 3) * u_r + (1 / 3) * u_s + (1 / 3) * u_c,
        "chosen_action": chosen["id"],
        "chosen_action_type": chosen["type"],
        "target_device_id": chosen["target_device_id"],
        "compromised_devices": step["compromised"],
        "isolated_devices": step["isolated"],
        "avg_cpu_load": telemetry["avg_cpu_load"],
        "avg_memory_load": telemetry["avg_memory_load"],
        "avg_network_load": telemetry["avg_network_load"],
        "avg_trust_score": telemetry["avg_trust_score"],
        "total_energy_kwh": telemetry["total_energy_kwh"],
        "total_cost_per_hour": telemetry["total_cost_per_hour"],
        "onset": info["onset"],
        "targeted_fraction": info["targeted_fraction"],
        "closed_loop": closed_loop,
        "us_mode": us_mode,
    }


def _run_info(twin, schedule_info):
    info = dict(schedule_info)
    info["targeted_fraction"] = twin.targeted_fraction
    info["mean_baseline_trust"] = float(
        np.mean([device.baseline_trust for device in twin.devices])
    )
    if hasattr(twin, "peak_intensity"):
        info["peak_intensity"] = float(twin.peak_intensity)
    elif "peak_intensity" in info:
        info["peak_intensity"] = float(info["peak_intensity"])
    else:
        info["peak_intensity"] = 0.0
    return info


# ---------------------------------------------------------------------------
# Closed-loop path: the environment is simulated per strategy
# ---------------------------------------------------------------------------
def run_v2_closed_loop(
    strategy,
    scenario,
    seed,
    num_ticks=NUM_TICKS_V2,
    us_mode="v2",
    twin_config=None,
    cl_config=None,
):
    """One closed-loop run; returns ``(rows, outcome_metrics_dict)``."""
    random.seed(seed)
    np.random.seed(seed)

    twin_cfg = twin_config or get_twin_v2_config()
    cl_cfg = cl_config or get_closed_loop_config()
    tau = twin_cfg["containment_tau"]

    twin = DigitalTwinV2(100, config=twin_cfg)
    schedule, schedule_info = build_v2_schedule(
        scenario, num_ticks, config=twin_cfg
    )
    info = _run_info(twin, schedule_info)

    prev_wS = 1 / 3
    rows = []
    flat_ticks = 0
    for tick in range(num_ticks):
        step = _environment_step(
            twin,
            schedule.get(tick, 0.0),
            cl_cfg,
            tick,
            us_mode,
            twin_cfg,
        )
        chosen, weights = _negotiate(
            strategy,
            step["actions"],
            step["U_R"],
            step["U_S"],
            step["U_C"],
            step["tss"],
            prev_wS,
            tau,
        )
        prev_wS = weights["wS"]
        if max(step["U_S"]) == min(step["U_S"]):
            flat_ticks += 1
        chosen_index = step["actions"].index(chosen)
        rows.append(
            _tick_row(
                strategy, scenario, seed, step, chosen, weights, chosen_index,
                True, us_mode, info,
            )
        )
        # Closed loop: the chosen action changes the state the NEXT tick sees.
        twin.apply_action(chosen, cl_cfg)

    outcome = compute_outcome_metrics(
        rows, info, num_ticks=num_ticks, tau=tau,
        recovery_tolerance=twin_cfg["recovery_tolerance"],
        us_identical_fraction=flat_ticks / num_ticks,
    )
    return rows, outcome


# ---------------------------------------------------------------------------
# Open-loop path: one strategy-independent trace, replayed per strategy
# ---------------------------------------------------------------------------
def run_v2_open_loop_env(
    scenario,
    seed,
    num_ticks=NUM_TICKS_V2,
    us_mode="v2",
    twin_config=None,
):
    """Strategy-independent open-loop trace for one (scenario, seed).

    No action is ever applied, so this trace is identical for every
    strategy and is computed once per (scenario, seed).

    It advances the twin through the seeded schedule so the trace reflects
    real CICIoT2023-derived noise per tick; TSS is computed in one batched
    IsolationForest decision_function call at the end so replay
    (:func:`run_v2_open_loop_replay`) can reuse the same per-tick TSS without
    re-advancing the rng.
    """
    random.seed(seed)
    np.random.seed(seed)

    twin_cfg = twin_config or get_twin_v2_config()
    twin = DigitalTwinV2(100, config=twin_cfg)
    schedule, schedule_info = build_v2_schedule(
        scenario, num_ticks, config=twin_cfg
    )
    info = _run_info(twin, schedule_info)

    trace = []
    states_for_tss = []
    for tick in range(num_ticks):
        step = _environment_step(twin, schedule.get(tick, 0.0), None, tick, us_mode, twin_cfg)
        trace.append(step)
        states_for_tss.append(step["twin_state"])
    tss_batch = _tss_batch(states_for_tss).ravel()
    for step, tss_val in zip(trace, tss_batch):
        step["tss"] = float(tss_val)
    return trace, info


def run_v2_open_loop_replay(
    strategy,
    scenario,
    seed,
    trace,
    info,
    num_ticks=NUM_TICKS_V2,
    us_mode="v2",
    twin_config=None,
):
    """Replay one strategy over a cached open-loop trace."""
    twin_cfg = twin_config or get_twin_v2_config()
    tau = twin_cfg["containment_tau"]

    prev_wS = 1 / 3
    rows = []
    flat_ticks = 0
    for step in trace:
        chosen, weights = _negotiate(
            strategy,
            step["actions"],
            step["U_R"],
            step["U_S"],
            step["U_C"],
            step["tss"],
            prev_wS,
            tau,
        )
        prev_wS = weights["wS"]
        if max(step["U_S"]) == min(step["U_S"]):
            flat_ticks += 1
        chosen_index = step["actions"].index(chosen)
        rows.append(
            _tick_row(
                strategy, scenario, seed, step, chosen, weights, chosen_index,
                False, us_mode, info,
            )
        )

    outcome = compute_outcome_metrics(
        rows, info, num_ticks=num_ticks, tau=tau,
        recovery_tolerance=twin_cfg["recovery_tolerance"],
        us_identical_fraction=flat_ticks / num_ticks,
    )
    return rows, outcome


# ---------------------------------------------------------------------------
# Outcome metrics (spec 2d)
# ---------------------------------------------------------------------------
def compute_outcome_metrics(
    rows,
    info,
    num_ticks=NUM_TICKS_V2,
    tau=0.15,
    recovery_tolerance=0.02,
    us_identical_fraction=None,
):
    """Aggregate one run's per-tick rows into the spec-2d outcome metrics.

    Definitions (documented in docs/phase2_summary.md as well):
      * time_to_containment - ticks from onset until TSS falls below tau.
        Both the bite and the containment must be SUSTAINED for
        ``_CONTAINMENT_RUN`` (3) consecutive ticks: the baseline TSS
        straddles tau (0.3 * (1 - trust) alone is ~0.045 plus anomaly
        jitter), so a single tick >= tau cannot mark an attack and a single
        dip < tau cannot mark containment (a lone dip at tick 7 would
        otherwise "contain" the gradual attack while TSS then rises to 0.69
        for the rest of the horizon). Concretely: bite = first tick >=
        onset with 3 consecutive ticks >= tau; containment = the first tick
        from the bite with 3 consecutive ticks < tau, measured from onset.
        If the attack never bites, containment is 0/True; if it never
        sustains below tau, the value is capped at the remaining horizon
        (num_ticks - onset) and contained is False.
      * cumulative_compromised_device_ticks - sum over ticks of the number
        of compromised non-isolated devices.
      * peak_TSS / area_under_TSS - max and discrete area (sum) of TSS over
        the horizon.
      * trust_recovery_time - ticks after the LAST attack tick until the
        fleet-average trust is back within recovery_tolerance of the fleet
        mean baseline trust; capped at the remaining horizon, with
        trust_recovered=False when that never happens (the horizon for a
        sustained attack that runs to the last tick is empty, giving
        trust_recovery_time=0 with trust_recovered=False).
      * cumulative_energy / cumulative_cost - sums over ticks of the fleet
        total_energy_kwh / total_cost_per_hour (energy-rate / cost-rate
        integrals over the horizon).
      * mean_U_S_attack - mean U_S of the CHOSEN action over the attack
        phase (ticks >= first onset).
      * attack_*_share / normal_*_share - chosen-action-type shares per
        phase (normal phase NaN when onset == 0).
      * isolate_share - share of ALL ticks whose chosen action type was
        isolate. Relabelled legacy "defensive response rate"; a behaviour
        descriptor, NOT an outcome metric.
    """
    df = pd.DataFrame(rows)
    onset = int(info["onset"])
    tss = df["TSS"].to_numpy()
    attack_mask = df["tick"].to_numpy() >= onset

    # --- containment (sustained run rule, see docstring) ------------------
    run = _CONTAINMENT_RUN

    def _first_sustained(start, predicate):
        for t in range(start, len(tss) - run + 1):
            if all(predicate(tss[t + i]) for i in range(run)):
                return t
        return None

    bite = _first_sustained(onset, lambda v: v >= tau)
    if bite is None:
        time_to_containment, contained = 0, True  # attack never crossed tau
    else:
        below = _first_sustained(bite, lambda v: v < tau)
        if below is not None:
            time_to_containment = int(below - onset)
            contained = True
        else:
            time_to_containment = num_ticks - onset
            contained = False

    # --- trust recovery after the last attack tick --------------------------
    intensity = df["attack_intensity"].to_numpy()
    attack_ticks = np.nonzero(intensity > 0)[0]
    reference = float(info["mean_baseline_trust"])
    post_trust = df["avg_trust_score"].to_numpy()
    if len(attack_ticks) == 0:
        trust_recovery_time, trust_recovered = 0, True
    else:
        attack_end = int(attack_ticks[-1])
        window = range(attack_end + 1, num_ticks)
        hit = next(
            (t for t in window if post_trust[t] >= reference - recovery_tolerance),
            None,
        )
        if hit is not None:
            trust_recovery_time, trust_recovered = hit - attack_end, True
        else:
            trust_recovery_time = max(0, num_ticks - 1 - attack_end)
            trust_recovered = False

    outcome = {
        "strategy": df["strategy"].iloc[0],
        "scenario": df["scenario"].iloc[0],
        "seed": df["seed"].iloc[0],
        "closed_loop": bool(df["closed_loop"].iloc[0]),
        "us_mode": df["us_mode"].iloc[0],
        "num_ticks": num_ticks,
        "onset": onset,
        "targeted_fraction": float(info["targeted_fraction"]),
        # 'peak_intensity' is populated by build_v2_schedule; if an older trace
    # doesn't carry it, fall back to 0.0 so the smoke closed-loop path
    # (which composes its own info dicts) doesn't crash.        "peak_intensity": float((info.get("peak_intensity") if isinstance(info.get("peak_intensity"), (int, float)) else (info.get("peak_intensity") if isinstance(info.get("peak_intensity"), (int, float)) else 0.0))),

        "mean_baseline_trust": reference,
        "time_to_containment": time_to_containment,
        "contained": bool(contained),
        "cumulative_compromised_device_ticks": int(
            df["compromised_devices"].sum()
        ),
        "peak_TSS": float(tss.max()),
        "area_under_TSS": float(tss.sum()),
        "trust_recovery_time": int(trust_recovery_time),
        "trust_recovered": bool(trust_recovered),
        "cumulative_energy": float(df["total_energy_kwh"].sum()),
        "cumulative_cost": float(df["total_cost_per_hour"].sum()),
        "FairScore_mean": float(df["FairScore"].mean()),
        "mean_U_S_attack": float(df.loc[attack_mask, "U_S"].mean()),
        "isolate_share": float((df["chosen_action_type"] == "isolate").mean()),
        "us_identical_fraction": (
            float(us_identical_fraction)
            if us_identical_fraction is not None
            else float("nan")
        ),
    }
    for phase_name, mask in (("attack", attack_mask), ("normal", ~attack_mask)):
        count = int(mask.sum())
        for action_type in ACTION_TYPES:
            key = f"{phase_name}_{action_type}_share"
            if count:
                outcome[key] = float(
                    (df.loc[mask, "chosen_action_type"] == action_type).mean()
                )
            else:
                outcome[key] = float("nan")
    return outcome


def build_outcome_summary(outcome_df, metric_columns=None):
    """Mean +/- SD across PER-SEED MEANS, per (strategy, scenario).

    This follows the Phase 0 audit recommendation: the SD is taken across
    the 20 per-seed values of each metric (ddof=1), not pooled over
    tick x seed, so it measures how much seeds really differ.
    """
    metrics = list(metric_columns or OUTCOME_METRIC_COLUMNS)
    metrics = [m for m in metrics if m in outcome_df.columns]
    per_seed = (
        outcome_df.groupby(["strategy", "scenario", "seed"], sort=False)[metrics]
        .mean()
        .reset_index()
    )
    agg = per_seed.groupby(["strategy", "scenario"], sort=False)[metrics].agg(
        ["mean", "std"]
    )
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    return agg.reset_index()


# ---------------------------------------------------------------------------
# Grids (spec 2e)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Grid tasks: one (scenario, seed) unit of work. Module-level so the tasks
# can be pickled to ProcessPoolExecutor workers (spawn) on Windows.
# Each run seeds random/np at entry, so a task is a pure function of its
# arguments and worker scheduling cannot change any result.
# ---------------------------------------------------------------------------
def _closed_loop_task(task):
    scenario, seed, strategies, num_ticks, us_mode, twin_config, cl_config = task
    rows, outcomes = [], []
    for strategy in strategies:
        run_rows, outcome = run_v2_closed_loop(
            strategy, scenario, seed,
            num_ticks=num_ticks, us_mode=us_mode,
            twin_config=twin_config, cl_config=cl_config,
        )
        rows.extend(run_rows)
        outcomes.append(outcome)
    return rows, outcomes


def _open_loop_task(task):
    scenario, seed, strategies, num_ticks, us_mode, twin_config = task
    trace, info = run_v2_open_loop_env(
        scenario, seed, num_ticks=num_ticks, us_mode=us_mode,
        twin_config=twin_config,
    )
    rows, outcomes = [], []
    for strategy in strategies:
        run_rows, outcome = run_v2_open_loop_replay(
            strategy, scenario, seed, trace, info,
            num_ticks=num_ticks, us_mode=us_mode,
            twin_config=twin_config,
        )
        rows.extend(run_rows)
        outcomes.append(outcome)
    return rows, outcomes


def _load_grid_progress(raw_path, outcome_path, strategies, num_ticks, us_mode,
                        closed_loop):
    """Resume state from the CSVs of an interrupted grid run.

    Returns ``(rows, outcomes, done_pairs)`` where ``done_pairs`` holds the
    (scenario, seed) pairs that already contain every requested strategy for
    the same us_mode / num_ticks / closed_loop. Empty inputs when there is
    nothing to resume.
    """
    if not (raw_path.exists() and outcome_path.exists()):
        return [], [], set()
    prev_out = pd.read_csv(outcome_path)
    prev_raw = pd.read_csv(raw_path)
    if not len(prev_out) or not len(prev_raw):
        return [], [], set()
    seed_match = (
        (prev_out["us_mode"] == us_mode)
        & (prev_out["num_ticks"].astype(int) == int(num_ticks))
        & (prev_out["closed_loop"] == closed_loop)
    )
    prev_out = prev_out[seed_match]
    wanted = set(strategies)
    done = {
        (str(scenario), int(seed))
        for (scenario, seed), grp in prev_out.groupby(["scenario", "seed"])
        if set(grp["strategy"]) == wanted
    }
    if not done:
        return [], [], set()
    raw_match = (
        (prev_raw["us_mode"] == us_mode)
        & (prev_raw["closed_loop"] == closed_loop)
        & np.array(
            [
                (str(s), int(d)) in done
                for s, d in zip(prev_raw["scenario"], prev_raw["seed"])
            ]
        )
    )
    prev_raw = prev_raw[raw_match]
    prev_out = prev_out[
        np.array(
            [
                (str(s), int(d)) in done
                for s, d in zip(prev_out["scenario"], prev_out["seed"])
            ]
        )
    ]
    return prev_raw.to_dict("records"), prev_out.to_dict("records"), done


def run_grid(
    closed_loop,
    seeds=None,
    scenarios=None,
    strategies=None,
    num_ticks=NUM_TICKS_V2,
    us_mode="v2",
    output_dir=None,
    twin_config=None,
    cl_config=None,
    workers=1,
    resume=False,
):
    """Run one full grid and write raw / outcome / summary CSVs.

    One task = one (scenario, seed): it runs all requested strategies for
    that pair (closed loop) or computes the shared open-loop trace once and
    replays every strategy over it. Tasks run in canonical scenario -> seed
    order; with ``workers > 1`` a process pool executes them in parallel and
    results are collected IN ORDER, so the CSVs are identical to a serial
    run. After every completed task the three CSVs are re-written via a
    temp file + ``os.replace`` (atomic), and ``resume=True`` skips the
    (scenario, seed) pairs already present in them, so an interrupted grid
    continues instead of restarting.
    """
    seeds = list(seeds or SEEDS_OFFICIAL)
    scenarios = list(scenarios or SCENARIOS_V2)
    strategies = list(strategies or STRATEGIES_V2)

    if output_dir is None:
        output_dir = Path("data/experiments_v2") / (
            "closed_loop" if closed_loop else "open_loop"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "experiment_results_v2.csv"
    outcome_path = output_dir / "outcome_metrics_v2.csv"
    summary_path = output_dir / "outcome_summary_v2.csv"

    rows, outcomes, done = [], [], set()
    if resume:
        rows, outcomes, done = _load_grid_progress(
            raw_path, outcome_path, strategies, num_ticks, us_mode, closed_loop
        )
        if done:
            print(f"[grid] resume: {len(done)} (scenario, seed) pairs done")

    tasks = [
        (scenario, seed, strategies, num_ticks, us_mode, twin_config, cl_config)
        if closed_loop
        else (scenario, seed, strategies, num_ticks, us_mode, twin_config)
        for scenario in scenarios
        for seed in seeds
        if (scenario, seed) not in done
    ]
    label = "closed_loop" if closed_loop else "open_loop"
    fn = _closed_loop_task if closed_loop else _open_loop_task
    total = len(tasks)
    started = time.perf_counter()

    def _checkpoint():
        results = pd.DataFrame(rows)
        outcome_df = pd.DataFrame(outcomes)
        summary_df = build_outcome_summary(outcome_df)
        for frame, path in (
            (results, raw_path),
            (outcome_df, outcome_path),
            (summary_df, summary_path),
        ):
            tmp = path.with_suffix(path.suffix + ".tmp")
            frame.to_csv(tmp, index=False)
            os.replace(tmp, path)
        return results, outcome_df, summary_df

    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            mapped = pool.map(fn, tasks)
            for index, (task, (task_rows, task_outcomes)) in enumerate(
                zip(tasks, mapped), start=1
            ):
                rows.extend(task_rows)
                outcomes.extend(task_outcomes)
                print(
                    f"[grid {label}] {task[0]} seed={task[1]} done "
                    f"({index}/{total}, {time.perf_counter() - started:.1f}s "
                    f"elapsed)",
                    flush=True,
                )
                _checkpoint()
    else:
        for index, task in enumerate(tasks, start=1):
            task_rows, task_outcomes = fn(task)
            rows.extend(task_rows)
            outcomes.extend(task_outcomes)
            print(
                f"[grid {label}] {task[0]} seed={task[1]} done "
                f"({index}/{total}, {time.perf_counter() - started:.1f}s "
                f"elapsed)",
                flush=True,
            )
            _checkpoint()

    results, outcome_df, summary_df = _checkpoint()
    print(f"Saved raw results to {raw_path}")
    print(f"Saved outcome metrics to {outcome_path}")
    print(f"Saved outcome summary to {summary_path}")
    return results, outcome_df, summary_df


# ---------------------------------------------------------------------------
# Smoke / calibration checks (tuning seeds ONLY)
# ---------------------------------------------------------------------------
def _episodes(tss, tau):
    """Contiguous runs of TSS >= tau; returns list of (start, end) tuples."""
    episodes = []
    start = None
    for tick, value in enumerate(tss):
        if value >= tau and start is None:
            start = tick
        elif value < tau and start is not None:
            episodes.append((start, tick - 1))
            start = None
    if start is not None:
        episodes.append((start, len(tss) - 1))
    return episodes


def run_smoke(num_ticks=NUM_TICKS_V2, out_root="data/experiments_v2"):
    """Calibration + timing checks on tuning seeds 1000-1009 ONLY.

    Prints, per scenario: the per-tick TSS curve (seed 1000, open loop),
    shape checks, the std of per-seed mean TSS across the 10 tuning seeds
    (the Phase 0 legacy value was 0.0018), and closed-loop TAPNA timing with
    the U_S flatness fraction; then a runtime estimate for both grids.
    """
    tcfg = get_twin_v2_config()
    tau = tcfg["containment_tau"]
    print("=" * 78)
    print("Phase 2 smoke - tuning seeds 1000-1009 ONLY (never used for grids)")
    print("=" * 78)

    # --- 1) per-tick TSS curves, one seed per scenario (open loop) ----------
    traces = {}
    for scenario in SCENARIOS_V2:
        trace, info = run_v2_open_loop_env(scenario, 1000, num_ticks=num_ticks)
        traces[scenario] = trace
        tss = [step["tss"] for step in trace]
        episodes = _episodes(tss, tau)
        print(f"\n--- {scenario} (seed 1000, open loop, onset={info['onset']}, "
              f"targeted_fraction={info['targeted_fraction']:.2f}) ---")
        print("TSS:", " ".join(f"{v:.3f}" for v in tss))
        print(
            f"episodes(TSS>=tau): {len(episodes)} {episodes[:6]} | "
            f"max TSS={max(tss):.3f} | final TSS={tss[-1]:.3f} | "
            f"min avg_trust={min(s['twin_state']['avg_trust_score'] for s in trace):.3f} | "
            f"max avg_network={max(s['twin_state']['avg_network_load'] for s in trace):.3f} | "
            f"max compromised={max(s['compromised'] for s in trace)}"
        )

    # --- 2) seed heterogeneity: std of per-seed mean TSS --------------------
    print("\n--- std of per-seed mean TSS across tuning seeds (legacy: 0.0018) ---")
    for scenario in SCENARIOS_V2:
        per_seed_means = []
        for seed in TUNING_SEEDS:
            trace, _ = run_v2_open_loop_env(scenario, seed, num_ticks=num_ticks)
            per_seed_means.append(float(np.mean([s["tss"] for s in trace])))
        spread = float(np.std(per_seed_means, ddof=1))
        verdict = "MEANINGFUL" if spread > 0.01 else "too small - recalibrate"
        print(
            f"{scenario}: per-seed means "
            f"[{min(per_seed_means):.3f}, {max(per_seed_means):.3f}] "
            f"std={spread:.4f} -> {verdict}"
        )

    # --- 3) closed-loop timing + U_S flatness (TAPNA, seed 1000) ------------
    print("\n--- closed-loop TAPNA timing + U_S flatness (seed 1000) ---")
    per_tick_closed = []
    for scenario in SCENARIOS_V2:
        started = time.perf_counter()
        try:
            _, outcome = run_v2_closed_loop(
                "TAPNA", scenario, 1000, num_ticks=num_ticks
            )
            elapsed = time.perf_counter() - started
            per_tick_closed.append(elapsed / num_ticks)
            print(
                f"{scenario}: {elapsed:.2f}s for {num_ticks} ticks "
                f"({elapsed / num_ticks * 1000:.1f} ms/tick) | "
                f"us_identical_fraction={outcome['us_identical_fraction']:.3f} | "
                f"mean_U_S_attack={outcome['mean_U_S_attack']:.3f} | "
                f"isolate_share={outcome['isolate_share']:.2f}"
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            print(
                f"{scenario}: FAILED after {elapsed:.2f}s -> {type(exc).__name__}: {exc}"
            )
            raise SystemExit(f"smoke failed on {scenario}; fix before running grids")

    # --- 4) open-loop replay timing -----------------------------------------
    # Use the real per-(scenario, seed) info captured above so the timing call
    # exercises compute_outcome_metrics with realistic values.
    _gradual_info = {
        "onset": traces["Gradual attack"][0].get("onset", 10),
        "targeted_fraction": 0.5,
        "mean_baseline_trust": 0.85,
    }
    start = time.perf_counter()
    run_v2_open_loop_replay(
        "TAPNA", "Gradual attack", 1000, traces["Gradual attack"],
        _gradual_info, num_ticks=num_ticks,
    )
    per_tick_replay = (time.perf_counter() - start) / num_ticks
    print(f"\nopen-loop replay: {per_tick_replay * 1000:.2f} ms/tick (negotiation only)")

    # --- 5) runtime estimate for both grids ---------------------------------
    closed_tick = float(np.mean(per_tick_closed))
    closed_est = closed_tick * len(STRATEGIES_V2) * len(SEEDS_OFFICIAL) * 4 * num_ticks
    open_est = (
        closed_tick * len(SEEDS_OFFICIAL) * 4 * num_ticks
        + per_tick_replay * len(STRATEGIES_V2) * len(SEEDS_OFFICIAL) * 4 * num_ticks
    )
    print(
        f"estimated closed_loop grid: {closed_est / 60:.1f} min | "
        f"open_loop grid: {open_est / 60:.1f} min"
    )
    print("(twin_config overrides from the grid never touch these defaults)")
    return traces


# ---------------------------------------------------------------------------
# Sensitivity check (spec 2c): 3 parameters x 3 values x 5 seeds, OFAT
# ---------------------------------------------------------------------------
SENSITIVITY_PARAMETERS = {
    "trust_recovery_rate": (
        lambda value: ({"trust_recovery_rate": value}, {}),
        [0.05, 0.10, 0.20],
    ),
    "propagation_rate": (
        lambda value: ({}, {"propagation": {"rate": value}}),
        [0.05, 0.10, 0.20],
    ),
    "migrate_threat_leak": (
        lambda value: ({}, {"actions": {"migrate": {"threat_leak": value}}}),
        [0.25, 0.50, 0.75],
    ),
}

RANKING_METRICS = {
    # metric: (outcome column, higher_is_better)
    "FairScore": ("FairScore_mean", True),
    "AUC_TSS": ("area_under_TSS", False),
    "time_to_containment": ("time_to_containment", False),
}


def kendall_tau(rank_x, rank_y):
    """Kendall tau-b between two best-first rankings of the same items."""
    pos_x = {item: pos for pos, item in enumerate(rank_x)}
    pos_y = {item: pos for pos, item in enumerate(rank_y)}
    items = list(rank_x)
    n = len(items)
    concordant = discordant = 0
    ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = np.sign(pos_x[items[i]] - pos_x[items[j]])
            dy = np.sign(pos_y[items[i]] - pos_y[items[j]])
            if dx == 0 and dy == 0:
                ties_x += 1
                ties_y += 1
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    n0 = n * (n - 1) / 2
    denom_x = n0 - ties_x
    denom_y = n0 - ties_y
    if denom_x == 0 or denom_y == 0:
        return float("nan")
    return (concordant - discordant) / float(np.sqrt(denom_x * denom_y))


def _rank_strategies(per_value, metric, higher_is_better):
    """Best-first strategy ranking by per-strategy mean of one metric."""
    means = per_value.groupby("strategy")[metric].mean()
    return list(means.sort_values(ascending=not higher_is_better).index)


def _sensitivity_task(task):
    """Run all six strategies for one (parameter, value, seed) cell."""
    (param_name, value, twin_cfg, cl_cfg, scenario, strategies, num_ticks,
     seed) = task
    records = []
    for strategy in strategies:
        _, outcome = run_v2_closed_loop(
            strategy, scenario, seed,
            num_ticks=num_ticks,
            twin_config=twin_cfg,
            cl_config=cl_cfg,
        )
        outcome["parameter"] = param_name
        outcome["value"] = value
        records.append(outcome)
    return records


def run_sensitivity(
    seeds=None,
    scenario="Gradual attack",
    num_ticks=NUM_TICKS_V2,
    out_dir=None,
    workers=1,
    resume=False,
):
    """OFAT sensitivity of strategy ranking to three closed-loop parameters.

    One parameter at a time around the baseline (trust_recovery_rate,
    propagation_rate, migrate_threat_leak; 3 values each incl. the
    baseline), 5 tuning seeds, closed loop, the canonical gradual scenario.
    Ranking stability is reported as pairwise Kendall tau-b between the
    strategy rankings at the different values of each parameter.

    Cells are executed in canonical (parameter, value, seed) order, in
    parallel when ``workers > 1`` (ordered collection keeps the output
    deterministic), with the outcomes CSV re-written after every cell and
    ``resume=True`` skipping the cells already present in it.
    """
    seeds = list(seeds or SENSITIVITY_SEEDS)
    if out_dir is None:
        out_dir = Path("data/experiments_v2/sensitivity")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = out_dir / "sensitivity_outcomes.csv"

    records = []
    done = set()
    if resume and outcomes_path.exists():
        prev = pd.read_csv(outcomes_path)
        if len(prev):
            wanted = set(STRATEGIES_V2)
            done = {
                (str(param), float(value), int(seed))
                for (param, value, seed), grp in prev.groupby(
                    ["parameter", "value", "seed"]
                )
                if set(grp["strategy"]) == wanted
            }
            if done:
                keep = [
                    (str(p), float(v), int(s)) in done
                    for p, v, s in zip(
                        prev["parameter"], prev["value"], prev["seed"]
                    )
                ]
                records = prev[keep].to_dict("records")
                print(f"[sensitivity] resume: {len(done)} cells done")

    tasks = []
    for param_name, (override_of, values) in SENSITIVITY_PARAMETERS.items():
        for value in values:
            twin_override, cl_override = override_of(value)
            twin_cfg = deep_update(get_twin_v2_config(), twin_override)
            cl_cfg = deep_update(get_closed_loop_config(), cl_override)
            for seed in seeds:
                if (param_name, float(value), seed) in done:
                    continue
                tasks.append(
                    (param_name, value, twin_cfg, cl_cfg, scenario,
                     STRATEGIES_V2, num_ticks, seed)
                )

    started = time.perf_counter()
    total = len(tasks)

    def _checkpoint():
        frame = pd.DataFrame(records)
        tmp = outcomes_path.with_suffix(outcomes_path.suffix + ".tmp")
        frame.to_csv(tmp, index=False)
        os.replace(tmp, outcomes_path)
        return frame

    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            for index, (task, cell) in enumerate(
                zip(tasks, pool.map(_sensitivity_task, tasks)), start=1
            ):
                records.extend(cell)
                print(
                    f"[sensitivity] {task[0]}={task[1]} seed={task[7]} done "
                    f"({index}/{total}, {time.perf_counter() - started:.1f}s "
                    f"elapsed)",
                    flush=True,
                )
                _checkpoint()
    else:
        for index, task in enumerate(tasks, start=1):
            records.extend(_sensitivity_task(task))
            print(
                f"[sensitivity] {task[0]}={task[1]} seed={task[7]} done "
                f"({index}/{total}, {time.perf_counter() - started:.1f}s "
                f"elapsed)",
                flush=True,
            )
            _checkpoint()

    outcome_df = _checkpoint()

    ranking_rows = []
    tau_rows = []
    for param_name, (_, values) in SENSITIVITY_PARAMETERS.items():
        subset = outcome_df[outcome_df["parameter"] == param_name]
        rankings = {}
        for value in values:
            per_value = subset[subset["value"] == value]
            for metric, (column, higher_better) in RANKING_METRICS.items():
                ranking = _rank_strategies(per_value, column, higher_better)
                rankings[(value, metric)] = ranking
                ranking_rows.append(
                    {
                        "parameter": param_name,
                        "value": value,
                        "metric": metric,
                        "ranking": " > ".join(ranking),
                    }
                )
        for metric in RANKING_METRICS:
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    tau = kendall_tau(
                        rankings[(values[i], metric)],
                        rankings[(values[j], metric)],
                    )
                    tau_rows.append(
                        {
                            "parameter": param_name,
                            "metric": metric,
                            "value_a": values[i],
                            "value_b": values[j],
                            "kendall_tau": tau,
                        }
                    )

    # sensitivity_outcomes.csv was already checkpointed by _checkpoint()
    pd.DataFrame(ranking_rows).to_csv(
        out_dir / "sensitivity_rankings.csv", index=False
    )
    tau_df = pd.DataFrame(tau_rows)
    tau_df.to_csv(out_dir / "sensitivity_kendall_tau.csv", index=False)

    print(f"\nSaved sensitivity outcomes to {out_dir / 'sensitivity_outcomes.csv'}")
    print("\nKendall tau-b between strategy rankings across parameter values:")
    print(tau_df.to_string(index=False))
    return outcome_df, tau_df



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--grid",
        choices=["closed_loop", "open_loop", "both"],
        help="Run a full 4x6x20x100 grid (official seeds).",
    )
    group.add_argument(
        "--smoke", action="store_true",
        help="Calibration/timing checks on tuning seeds 1000-1009.",
    )
    group.add_argument(
        "--sensitivity", action="store_true",
        help="OFAT sensitivity check (3 params x 3 values x 5 tuning seeds).",
    )
    parser.add_argument("--num-ticks", type=int, default=NUM_TICKS_V2)
    parser.add_argument(
        "--seeds", default=None,
        help="Comma-separated seed override (default: official SEEDS for "
             "grids, tuning seeds for smoke/sensitivity).",
    )
    parser.add_argument("--out-root", default="data/experiments_v2")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel grid/sensitivity processes (default: min(8, cpus-1)). "
             "Results are identical to a serial run; tasks are collected "
             "in canonical order.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip (scenario, seed) / sensitivity cells already present in "
             "the output CSVs and continue an interrupted run.",
    )
    args = parser.parse_args()
    workers = args.workers or max(1, min(8, (os.cpu_count() or 2) - 1))

    if args.smoke:
        run_smoke(num_ticks=args.num_ticks, out_root=args.out_root)
        return
    if args.sensitivity:
        run_sensitivity(
            num_ticks=args.num_ticks,
            out_dir=Path(args.out_root) / "sensitivity",
            workers=workers,
            resume=args.resume,
        )
        return

    seeds = (
        [int(s) for s in args.seeds.split(",") if s.strip()]
        if args.seeds else None
    )
    grids = (
        ["closed_loop", "open_loop"] if args.grid == "both" else [args.grid]
    )
    for grid in grids:
        run_grid(
            closed_loop=(grid == "closed_loop"),
            seeds=seeds,
            num_ticks=args.num_ticks,
            output_dir=Path(args.out_root) / grid,
            workers=workers,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
