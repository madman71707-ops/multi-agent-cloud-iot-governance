"""Phase 2b v3 environment runner.

New environment with finite-cost isolation (isolation_cost="v3") and the
v3 resource utility (ur_mode="v3"), on top of the existing v2 mechanics
(dynamics="v2", us_mode="v2"). The v2 environment and its
data/experiments_v2/ outputs are untouched and remain reproducible.

New strategies added (do NOT rename existing ones):
  NoOp           always no_op (do-nothing reference)
  RandomAction   uniform random candidate each tick, seeded by the run seed

Existing strategies kept:
  TAPNA, Greedy Cost, Greedy Latency, Static Weighted, ParetoCrowding,
  ThreatRuleOnly

Official seed list: seed 1001 is replaced by 12345 (it lies inside the
tuning range 1000-1009). The full 20 official seeds are printed at run time.

Official run: 4 scenarios x 8 strategies x 20 seeds x 100 ticks, closed loop,
isolation_cost="v3", us_mode="v2", ur_mode="v3", output to
data/experiments_v3/closed_loop/.

Calibration sandbox: tuning seeds 1000-1009 ONLY, picks kappa,
max_isolated_fraction, isolation_duration using TAPNA-agnostic criteria and
prints the frozen config + tuning table + a small kappa & max_isolated_fraction
sensitivity with ranking stability.

Frozen parameters (from calibration on seeds 1000-1009):
  kappa = 2.0
  max_isolated_fraction = 0.15
  isolation_duration = 11
  energy_saving (isolate) = 0.0
  propagate_service_loss_exponent = 0.5
  migrate_service_loss_fraction = 0.2
  scale_down_capacity_reduction = 0.1
  scale_down_load_shift = 0.1
  propagate_neighbours = 1
  propagate_rate = 0.10
  propagate_impact = 0.05
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.resource_agent import _metric, score_resource
from digital_twin.twin_v3 import (
    DigitalTwinV3,
    get_closed_loop_v3_config as _get_v3_config,
)
from digital_twin.twin_v2 import (
    build_v2_schedule,
    deep_update,
    generate_candidate_actions_v2,
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
from sim.scoring_v2 import score_actions_v2
import sim.scoring_v2 as sim_scoring_v2
from sim.scoring_v3 import ur_v3_list

from sim.experiment_runner import SEEDS as _SEEDS_OFFICIAL
from sim.attack_patterns_v2 import (
    _CONTAINMENT_RUN,
    _rank_strategies,
    build_outcome_summary,
    kendall_tau,
)


SCENARIOS_V3 = [
    "Gradual attack",
    "Short attack spike",
    "Repeated attacks",
    "Sustained attack",
]

STRATEGIES_V3 = [
    "TAPNA",
    "Greedy Cost",
    "Greedy Latency",
    "Static Weighted",
    "ParetoCrowding",
    "ThreatRuleOnly",
    "NoOp",
    "RandomAction",
]

ACTION_TYPES = ["isolate", "migrate", "scale_down", "no_op"]

_NUM_TICKS = 100
_TUNING_SEEDS = list(range(1000, 1010))
_TUNING_SEEDS_SMALL = list(range(1000, 1005))

# Official seed list = the canonical SEEDS from sim/experiment_runner.py with
# seed 1001 replaced by 12345 (1001 lies inside the tuning range
# 1000-1009, so it may not be used for the reported grid). 20 seeds.
def _official_seeds():
    seeds = list(_SEEDS_OFFICIAL)
    return [12345 if s == 1001 else s for s in seeds]


def _iso_duration():
    v3 = yaml.safe_load(open("config/closed_loop_v3.yaml", encoding="utf-8"))
    return v3["actions"]["isolate"]["isolation_duration"]


def _max_isolated():
    v3 = yaml.safe_load(open("config/closed_loop_v3.yaml", encoding="utf-8"))
    return v3["isolation_cap"]["max_isolated_fraction"]


def _tau():
    return get_twin_v2_config()["containment_tau"]


def _negotiate(strategy, actions, U_R_list, U_S_list, U_C_list, tss, prev_wS, tau):
    thirds = {"wR": 1/3, "wS": 1/3, "wC": 1/3}
    if strategy == "TAPNA":
        chosen, weights = run_tapna(actions, U_R_list, U_S_list, U_C_list, tss, prev_wS)
    elif strategy == "Greedy Cost":
        # Same dispatch and reporting weights as sim/attack_patterns_v2._negotiate.
        chosen = greedy_cost_baseline(actions, U_C_list)
        weights = {"wR": 0.0, "wS": 0.0, "wC": 1.0}
    elif strategy == "Greedy Latency":
        chosen = greedy_latency_baseline(actions, U_R_list)
        weights = {"wR": 1.0, "wS": 0.0, "wC": 0.0}
    elif strategy == "Static Weighted":
        chosen = static_weighted_baseline(actions, U_R_list, U_S_list, U_C_list)
        weights = dict(thirds)
    elif strategy == "ParetoCrowding":
        chosen = paretocrowding_baseline(actions, U_R_list, U_S_list, U_C_list)
        weights = dict(thirds)
    elif strategy == "ThreatRuleOnly":
        chosen = threat_rule_only_baseline(actions, U_R_list, U_S_list,
                                           U_C_list, tss, tau=tau)
        weights = dict(thirds)
    elif strategy == "NoOp":
        chosen = next(a for a in actions if a["type"] == "no_op")
        weights = dict(thirds)
    elif strategy == "RandomAction":
        # Seeded by the run seed: run_v3_closed_loop calls random.seed(seed)
        # on entry, so the module-level stream is per-run reproducible.
        chosen = random.choice(actions)
        weights = dict(thirds)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return chosen, weights


def _score_tick_v3(twin, intensity, tick, us_mode, ur_mode, twin_cfg, v3_cfg, kappa):
    twin.advance(intensity, closed_loop_cfg=v3_cfg)
    twin_state = twin.get_telemetry()
    # Cap-aware candidate generation (isolate candidates are dropped when the
    # isolation cap is reached) - see DigitalTwinV3.generate_candidates.
    actions = twin.generate_candidates(tick)
    tss = float(sim_scoring_v2._tss_batch([twin_state]).ravel()[0])
    U_R_list, U_S_list, U_C_list = score_actions_v2(
        actions, twin, twin_state, twin_state, tss,
        us_mode=us_mode, config=twin_cfg,
    )
    if ur_mode == "v3":
        U_R_list = ur_v3_list(actions, twin, v3_cfg, kappa=kappa,
                               legacy=U_R_list)
    elif ur_mode != "legacy":
        raise ValueError(f"Unknown ur_mode: {ur_mode!r}")
    return {
        "tick": tick,
        "intensity": intensity,
        "twin_state": twin_state,
        "actions": actions,
        "tss": tss,
        "U_R": U_R_list,
        "U_S": U_S_list,
        "U_C": U_C_list,
    }


def _print_official_header(twin_cfg, base_v3, frozen_kappa, frozen_cap, frozen_duration):
    print("=" * 78)
    print("FROZEN config (after calibration on seeds 1000-1009):")
    print("=" * 78)
    print(f"  kappa: {frozen_kappa}")
    print(f"  max_isolated_fraction: {frozen_cap}")
    print(f"  isolation_duration: {frozen_duration}")
    isolate = base_v3["actions"]["isolate"]
    migrate = base_v3["actions"]["migrate"]
    scale_down = base_v3["actions"]["scale_down"]
    prop = base_v3["propagation"]
    print(f"  energy_saving (isolate): {isolate.get('energy_saving', 0.0)}")
    print(f"  isolate capacity_penalty: {isolate.get('capacity_penalty')}, "
          f"cost_penalty: {isolate.get('cost_penalty')}")
    print(f"  migrate_service_loss_fraction: "
          f"{migrate.get('migrate_service_loss_fraction')}")
    print(f"  scale_down capacity_reduction: {scale_down.get('capacity_reduction')}, "
          f"energy_reduction: {scale_down.get('energy_reduction')}")
    print(f"  propagation: rate={prop.get('rate')}, impact={prop.get('impact')}, "
          f"neighbours={prop.get('neighbours')}")
    print()


def run_v3_closed_loop(strategy, scenario, seed, kappa,
                       twin_config=None, v3_config=None, num_ticks=_NUM_TICKS,
                       us_mode="v2", ur_mode="v3", tau=_tau()):
    random.seed(seed)
    np.random.seed(seed)
    twin_cfg = twin_config or get_twin_v2_config()
    v3_cfg = v3_config or _get_v3_config()
    twin = DigitalTwinV3(100, config=twin_cfg, cl_config=v3_cfg)

    schedule, schedule_info = build_v2_schedule(scenario, num_ticks, config=twin_cfg)
    info = dict(schedule_info)
    info["targeted_fraction"] = twin.targeted_fraction
    info["mean_baseline_trust"] = float(np.mean([d.baseline_trust for d in twin.devices]))
    info["peak_intensity"] = float(max(schedule.values())) if schedule else 0.0

    rows = []
    flat_ticks = 0
    prev_wS = 1/3
    for tick in range(num_ticks):
        step = _score_tick_v3(twin, schedule.get(tick, 0.0), tick,
                              us_mode, ur_mode, twin_cfg, v3_cfg, kappa)
        chosen, weights = _negotiate(strategy, step["actions"], step["U_R"],
                                     step["U_S"], step["U_C"], step["tss"],
                                     prev_wS, tau)
        prev_wS = weights["wS"]
        if max(step["U_S"]) == min(step["U_S"]):
            flat_ticks += 1
        ci = step["actions"].index(chosen)
        row = {
            "tick": step["tick"],
            "strategy": strategy,
            "scenario": scenario,
            "seed": seed,
            "isolation_cost": "v3",
            "us_mode": us_mode,
            "ur_mode": ur_mode,
            "kappa": kappa,
            "closed_loop": True,
            "attack_intensity": float(step["intensity"]),
            "onset": int(info["onset"]),
            "TSS": float(step["tss"]),
            "wR": float(weights["wR"]),
            "wS": float(weights["wS"]),
            "wC": float(weights["wC"]),
            "CIGI": float(compute_cigi(step["U_R"][ci], step["U_S"][ci],
                                       step["U_C"][ci], weights)),
            "avg_trust_score": float(step["twin_state"]["avg_trust_score"]),
            "avg_network_load": float(step["twin_state"]["avg_network_load"]),
            "total_energy_kwh": float(step["twin_state"]["total_energy_kwh"]),
            "total_cost_per_hour": float(step["twin_state"]["total_cost_per_hour"]),
            "FairScore": float(weights["wR"]*step["U_R"][ci]
                               + weights["wS"]*step["U_S"][ci]
                               + weights["wC"]*step["U_C"][ci]),
            "U_R": float(step["U_R"][ci]),
            "U_S": float(step["U_S"][ci]),
            "U_C": float(step["U_C"][ci]),
            "chosen_action": chosen["id"],
            "chosen_action_type": chosen["type"],
            "target_device_id": chosen.get("target_device_id"),
            "isolated_devices": int(twin.isolated_count),
            "compromised_devices": int(twin.compromised_count),
            # Per-tick v3 metrics, read from the SAME telemetry snapshot the
            # scorers saw (state after advance, before this tick's action).
            "false_isolated_device_ticks": int(
                step["twin_state"]["false_isolated_device_ticks"]),
            "offline_workload_share": float(
                step["twin_state"]["offline_workload_share"]),
            "service_availability": float(
                step["twin_state"]["service_availability"]),
            "max_isolated_possible": int(twin.max_isolated_count),
            "us_identical_fraction": float(max(step["U_S"]) == min(step["U_S"])),
        }
        rows.append(row)
        twin.apply_action(chosen, v3_cfg)
    outcome = _compute_outcome_v3(rows, info, num_ticks=num_ticks, tau=tau,
                                  recovery_tolerance=twin_cfg["recovery_tolerance"])
    outcome["us_identical_fraction"] = flat_ticks / num_ticks
    return rows, outcome


def _compute_outcome_v3(rows, info, num_ticks=_NUM_TICKS, tau=_tau(),
                        recovery_tolerance=0.02):
    df = pd.DataFrame(rows)
    onset = int(info["onset"])
    tss = df["TSS"].to_numpy()
    attack_mask = df["tick"].to_numpy() >= onset

    def _first_sustained(start, pred):
        n = len(tss)
        for t in range(start, n - _CONTAINMENT_RUN + 1):
            if all(pred(tss[t+i]) for i in range(_CONTAINMENT_RUN)):
                return t
        return None
    bite = _first_sustained(onset, lambda v: v >= tau)
    if bite is None:
        time_to_containment, contained = 0, True
    else:
        below = _first_sustained(bite, lambda v: v < tau)
        if below is not None:
            time_to_containment = int(below - onset)
            contained = True
        else:
            time_to_containment = num_ticks - onset
            contained = False

    intensity = df["attack_intensity"].to_numpy()
    attack_ticks = np.nonzero(intensity > 0)[0]
    ref = float(info.get("mean_baseline_trust", 0.85))
    post_trust = df["avg_trust_score"].to_numpy()
    if len(attack_ticks) == 0:
        trust_recovery_time, trust_recovered = 0, True
    else:
        attack_end = int(attack_ticks[-1])
        window = range(attack_end + 1, num_ticks)
        hit = next((t for t in window if post_trust[t] >= ref - recovery_tolerance), None)
        if hit is not None:
            trust_recovery_time, trust_recovered = hit - attack_end, True
        else:
            trust_recovery_time = max(0, num_ticks - 1 - attack_end)
            trust_recovered = False

    isolated_tick_total = int(df["isolated_devices"].sum())
    compromised_tick_total = int(df["compromised_devices"].sum())
    offline_workload_per_tick = df["offline_workload_share"].to_numpy(dtype=float) \
        if "offline_workload_share" in df else np.zeros(len(df))
    net_disruption = float(np.nansum(offline_workload_per_tick))
    service_availability_per_tick = 1.0 - offline_workload_per_tick
    service_availability_mean = float(np.mean(service_availability_per_tick))
    service_availability_min = float(np.min(service_availability_per_tick))
    max_isolated = int(df["isolated_devices"].max())
    false_isolated_device_ticks = int(df.get(
        "false_isolated_device_ticks",
        pd.Series([0]*len(df))).sum())

    outcome = {
        "strategy": df["strategy"].iloc[0],
        "scenario": df["scenario"].iloc[0],
        "seed": df["seed"].iloc[0],
        "isolation_cost": df["isolation_cost"].iloc[0],
        "us_mode": df["us_mode"].iloc[0],
        "ur_mode": df["ur_mode"].iloc[0],
        "num_ticks": num_ticks,
        "onset": onset,
        "targeted_fraction": float(info["targeted_fraction"]),
        "mean_baseline_trust": ref,
        "time_to_containment": time_to_containment,
        "contained": bool(contained),
        "cumulative_compromised_device_ticks": compromised_tick_total,
        "peak_TSS": float(tss.max()),
        "area_under_TSS": float(tss.sum()),
        "trust_recovery_time": int(trust_recovery_time),
        "trust_recovered": bool(trust_recovered),
        "cumulative_energy": float(df["total_energy_kwh"].sum()),
        "cumulative_cost": float(df["total_cost_per_hour"].sum()),
        "mean_U_S_attack": float(df.loc[attack_mask, "U_S"].mean()) if attack_mask.any() else float("nan"),
        "FairScore_mean": float(df["FairScore"].mean()),
        "isolate_share": float((df["chosen_action_type"] == "isolate").mean()),
        # --- Phase 2b new metrics (2b-3), all non-negative by construction ---
        "service_availability_mean": service_availability_mean,
        "service_availability_min": service_availability_min,
        "isolated_device_ticks_total": isolated_tick_total,
        "false_isolation_ticks": false_isolated_device_ticks,
        "max_isolated_devices": max_isolated,
        "net_disruption_cost": net_disruption,
    }
    for phase_name, mask in (("attack", attack_mask), ("normal", ~attack_mask)):
        count = int(mask.sum())
        for at in ACTION_TYPES:
            key = f"{phase_name}_{at}_share"
            outcome[key] = (float((df.loc[mask, "chosen_action_type"] == at).mean())
                            if count else float("nan"))
        # Per-phase (pre-onset vs attack) reporting of the v3 cost metrics.
        if count:
            phase_offline = float(np.mean(offline_workload_per_tick[mask]))
            outcome[f"{phase_name}_service_availability_mean"] = 1.0 - phase_offline
            outcome[f"{phase_name}_net_disruption_cost"] = float(
                np.nansum(offline_workload_per_tick[mask]))
            outcome[f"{phase_name}_isolated_device_ticks"] = int(
                df.loc[mask, "isolated_devices"].sum())
            outcome[f"{phase_name}_false_isolation_ticks"] = int(
                np.asarray(df.get("false_isolated_device_ticks",
                                  pd.Series([0]*len(df))))[mask].sum())
        else:
            outcome[f"{phase_name}_service_availability_mean"] = float("nan")
            outcome[f"{phase_name}_net_disruption_cost"] = float("nan")
            outcome[f"{phase_name}_isolated_device_ticks"] = 0
            outcome[f"{phase_name}_false_isolation_ticks"] = 0
    normal_count = int((~attack_mask).sum())
    outcome["pre_onset_isolate_share"] = (
        float((df.loc[~attack_mask, "chosen_action_type"] == "isolate").mean())
        if normal_count else float("nan"))
    return outcome


def _kappa_candidates_default():
    return [1.0, 2.0, 3.0]


def _cap_candidates_default():
    return [0.10, 0.15, 0.20]


def _frozen_kappa():
    return 2.0


def _frozen_cap():
    return 0.15


# ---------------------------------------------------------------------------
# Calibration grid (2b-5): parallel, checkpointed, resumable
# ---------------------------------------------------------------------------
CALIB_DIR = Path("data/experiments_v3/calibration")
CALIB_CSV = CALIB_DIR / "calib_grid.csv"

_CALIB_CACHE = {}
_CALIB_LOADED = False


def _calib_config(cap, duration=None):
    """Frozen-shape v3 config for one calibration cell (only the cap varies)."""
    return deep_update(_get_v3_config(), {
        "actions": {
            "isolate": {
                "energy_saving": 0.0,
                "isolation_duration": (_iso_duration() if duration is None
                                       else duration),
            },
            "migrate": {"migrate_service_loss_fraction": 0.2},
            "scale_down": {"capacity_reduction": 0.1},
            "no_op": {},
        },
        "isolation_cap": {"max_isolated_fraction": cap},
    })


def _calib_key(kappa, cap, scenario, seed, strategy):
    return (round(float(kappa), 6), round(float(cap), 6),
            str(scenario), int(seed), str(strategy))


def _load_calib_cache():
    global _CALIB_LOADED
    if _CALIB_LOADED:
        return
    _CALIB_LOADED = True
    if not CALIB_CSV.exists():
        return
    try:
        prev = pd.read_csv(CALIB_CSV)
    except Exception:
        return
    for rec in prev.to_dict("records"):
        try:
            key = _calib_key(rec["kappa"], rec["max_isolated_fraction"],
                             rec["scenario"], rec["seed"], rec["strategy"])
        except (KeyError, TypeError, ValueError):
            continue
        _CALIB_CACHE[key] = rec
    if _CALIB_CACHE:
        print(f"[calib] resume: {len(_CALIB_CACHE)} cached runs from {CALIB_CSV}")


def _save_calib_cache():
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(_CALIB_CACHE.values()))
    frame.to_csv(CALIB_CSV, index=False)


def _calib_task(task):
    """One (kappa, cap, scenario, seed) cell: every strategy. Picklable."""
    kappa, cap, scenario, seed, strategies, num_ticks, twin_config = task
    v3_cfg = _calib_config(cap)
    records = []
    for strategy in strategies:
        _, outcome = run_v3_closed_loop(
            strategy, scenario, seed, kappa=kappa,
            twin_config=twin_config, v3_config=v3_cfg,
            num_ticks=num_ticks, tau=_tau())
        outcome = dict(outcome)
        outcome["kappa"] = kappa
        outcome["max_isolated_fraction"] = cap
        records.append(outcome)
    return records


def run_calibration_grid(workers=1):
    """Run the kappa x cap x scenario x seed calibration grid.

    Cells already present in data/experiments_v3/calibration/calib_grid.csv
    are skipped, and the CSV is re-written atomically after every completed
    cell, so an interrupted calibration resumes where it stopped.
    """
    _load_calib_cache()
    twin_cfg = get_twin_v2_config()
    tasks = []
    for kappa in _kappa_candidates_default():
        for cap in _cap_candidates_default():
            for scenario in SCENARIOS_V3:
                for seed in _TUNING_SEEDS:
                    missing = [s for s in STRATEGIES_V3
                               if _calib_key(kappa, cap, scenario, seed, s)
                               not in _CALIB_CACHE]
                    if missing:
                        tasks.append((kappa, cap, scenario, seed, missing,
                                      _NUM_TICKS, twin_cfg))
    total = len(tasks)
    print(f"[calib] {total} cells to run (each: all {len(STRATEGIES_V3)} "
          f"strategies), workers={workers}", flush=True)
    if not tasks:
        return
    started = time.perf_counter()

    def _absorb(records):
        for outcome in records:
            key = _calib_key(outcome["kappa"], outcome["max_isolated_fraction"],
                             outcome["scenario"], outcome["seed"],
                             outcome["strategy"])
            _CALIB_CACHE[key] = outcome
        _save_calib_cache()

    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for index, records in enumerate(pool.map(_calib_task, tasks), start=1):
                _absorb(records)
                if index % 10 == 0 or index == total:
                    print(f"[calib] {index}/{total} cells "
                          f"({time.perf_counter()-started:.0f}s elapsed)",
                          flush=True)
    else:
        for index, task in enumerate(tasks, start=1):
            _absorb(_calib_task(task))
            if index % 10 == 0 or index == total:
                print(f"[calib] {index}/{total} cells "
                      f"({time.perf_counter()-started:.0f}s elapsed)",
                      flush=True)


def _calib_cached(kappa, cap, scenario, seed, strategy, twin_config):
    """Outcome of one grid cell from the cache (runs it on a miss)."""
    _load_calib_cache()
    key = _calib_key(kappa, cap, scenario, seed, strategy)
    if key in _CALIB_CACHE:
        return _CALIB_CACHE[key]
    _, outcome = run_v3_closed_loop(
        strategy, scenario, seed, kappa=kappa, twin_config=twin_config,
        v3_config=_calib_config(cap), tau=_tau())
    outcome = dict(outcome)
    outcome["kappa"] = kappa
    outcome["max_isolated_fraction"] = cap
    _CALIB_CACHE[key] = outcome
    _save_calib_cache()
    return outcome


def calibration(workers=1):
    from sim.attack_patterns_v2 import _rank_strategies, kendall_tau
    twin_cfg = get_twin_v2_config()

    kappa_candidates = _kappa_candidates_default()
    cap_candidates = _cap_candidates_default()

    print("=" * 78)
    print("Phase 2b calibration sandbox (tuning seeds 1000-1009 ONLY)")
    print("=" * 78)
    print("\nCandidate config (before tuning):")
    print(f"  kappa candidates: {kappa_candidates}")
    print(f"  max_isolated_fraction candidates: {cap_candidates}")
    print(f"  isolation_duration (frozen from YAML): {_iso_duration()}")
    print()

    # Phase 1: run (or resume) the full calibration grid in parallel.
    run_calibration_grid(workers=workers)

    results = []
    for kappa in kappa_candidates:
        for cap in cap_candidates:
            aggregate = {}
            for scenario in SCENARIOS_V3:
                per_strategy = {}
                for strategy in STRATEGIES_V3:
                    per_seed_outcomes = [
                        _calib_cached(kappa, cap, scenario, seed, strategy,
                                      twin_cfg)
                        for seed in _TUNING_SEEDS
                    ]
                    mean_outcome = {k: float(np.nanmean([o[k] for o in per_seed_outcomes]))
                                    for k in [
                        "time_to_containment", "area_under_TSS",
                        "cumulative_compromised_device_ticks", "isolate_share",
                        "pre_onset_isolate_share", "service_availability_mean",
                        "service_availability_min", "isolated_device_ticks_total",
                        "false_isolation_ticks", "max_isolated_devices",
                        "net_disruption_cost", "FairScore_mean", "peak_TSS",
                        "mean_U_S_attack", "us_identical_fraction",
                    ]}
                    per_strategy[strategy] = mean_outcome
                aggregate[scenario] = per_strategy

            sw = aggregate["Gradual attack"]["Static Weighted"]
            noop = aggregate["Gradual attack"]["NoOp"]
            max_isolated_over_scenarios = max(
                aggregate[s][st].get("max_isolated_devices", 0)
                for s in SCENARIOS_V3 for st in STRATEGIES_V3) / 100.0
            fail_reasons = []
            if max_isolated_over_scenarios > 0.30:
                fail_reasons.append(f"max_isolated fraction {max_isolated_over_scenarios:.2%} > 30%")
            if sw["pre_onset_isolate_share"] >= 0.50:
                fail_reasons.append("Static Weighted pre-onset isolate share >= 50%")
            if not (noop["cumulative_compromised_device_ticks"] > sw["cumulative_compromised_device_ticks"]
                    and noop["area_under_TSS"] > sw["area_under_TSS"]):
                fail_reasons.append("NoOp not clearly worse than Static Weighted (acting not worthwhile)")
            ok = not fail_reasons
            record = {
                "kappa": kappa, "max_isolated_fraction": cap, "ok": ok,
                "reason": "; ".join(fail_reasons),
                "max_isolated_fraction_used": max_isolated_over_scenarios,
                "sw_pre_onset_isolate_share": sw["pre_onset_isolate_share"],
                "noop_cum_compromised": noop["cumulative_compromised_device_ticks"],
                "sw_cum_compromised": sw["cumulative_compromised_device_ticks"],
                "noop_auc": noop["area_under_TSS"],
                "sw_auc": sw["area_under_TSS"],
                "noop_isolate_share": noop["isolate_share"],
                "gradual": {st: aggregate["Gradual attack"][st]
                            for st in ["TAPNA","Static Weighted","ThreatRuleOnly","NoOp","RandomAction"]},
            }
            results.append(record)
            print(f"  kappa={kappa}, cap={cap}: {'OK' if ok else 'FAIL ' + record['reason']}")

    def failure_count(r):
        n = 0
        if r["max_isolated_fraction_used"] > 0.30: n += 1
        if r["sw_pre_onset_isolate_share"] >= 0.50: n += 1
        if not (r["noop_cum_compromised"] > r["sw_cum_compromised"]
                and r["noop_auc"] > r["sw_auc"]): n += 1
        return n

    ranked = sorted(results, key=failure_count)
    chosen = ranked[0]
    frozen_kappa = chosen["kappa"]
    frozen_cap = chosen["max_isolated_fraction"]
    frozen_duration = _iso_duration()

    print("\n" + "=" * 78)
    print("FROZEN config (after calibration):")
    print("=" * 78)
    print(f"  kappa: {frozen_kappa}")
    print(f"  max_isolated_fraction: {frozen_cap}")
    print(f"  isolation_duration: {frozen_duration}")
    print(f"  energy_saving (isolate): 0.0")
    print(f"  migrate_service_loss_fraction: 0.2")
    print(f"  scale_down_capacity_reduction: 0.1")
    print(f"  propagation: rate=0.10, impact=0.05, neighbours=2")
    print("\nCalibration ranking (kappa, cap) -> OK/FAIL:")
    for r in ranked:
        print(f"  kappa={r['kappa']}, cap={r['max_isolated_fraction']}: "
              f"{'OK' if r['ok'] else 'FAIL ' + r['reason']}")

    print("\n" + "=" * 78)
    print(f"TUNING-SEED RESULTS TABLE (seeds 1000-1009, kappa={frozen_kappa}, "
          f"cap={frozen_cap}, duration={frozen_duration})")
    print("=" * 78)
    # Frozen-config table: a subset of the calibration grid (Gradual attack,
    # seeds 1000-1009), so it is read straight from the cache - no new runs.
    for strategy in STRATEGIES_V3:
        per_seed_outcomes = [
            _calib_cached(frozen_kappa, frozen_cap, "Gradual attack", seed,
                          strategy, twin_cfg)
            for seed in _TUNING_SEEDS
        ]
        mean_outcome = {k: float(np.nanmean([o[k] for o in per_seed_outcomes]))
                        for k in [
            "time_to_containment", "area_under_TSS",
            "cumulative_compromised_device_ticks", "isolate_share",
            "pre_onset_isolate_share", "service_availability_mean",
            "service_availability_min", "isolated_device_ticks_total",
            "false_isolation_ticks", "max_isolated_devices",
            "net_disruption_cost", "FairScore_mean", "peak_TSS",
            "mean_U_S_attack", "us_identical_fraction",
        ]}
        print(f"\n  [{strategy}] (mean over seeds 1000-1009)")
        for k in ["time_to_containment", "area_under_TSS",
                  "cumulative_compromised_device_ticks", "isolate_share",
                  "pre_onset_isolate_share", "service_availability_mean",
                  "service_availability_min", "isolated_device_ticks_total",
                  "false_isolation_ticks", "max_isolated_devices",
                  "net_disruption_cost", "FairScore_mean", "peak_TSS",
                  "mean_U_S_attack", "us_identical_fraction"]:
            print(f"    {k}: {mean_outcome[k]:.3f}")

    print("\n" + "=" * 78)
    print("Small sensitivity check (kappa x 3, max_isolated_fraction x 3, "
          "5 tuning seeds, Gradual attack)")
    print("=" * 78)
    # Sensitivity check: also a subset of the calibration grid (Gradual
    # attack, seeds 1000-1004, every kappa x cap cell) -> cache only.
    sens_records = []
    for kappa in kappa_candidates:
        for cap in cap_candidates:
            for seed in _TUNING_SEEDS_SMALL:
                for strategy in STRATEGIES_V3:
                    outcome = dict(_calib_cached(kappa, cap, "Gradual attack",
                                                 seed, strategy, twin_cfg))
                    outcome["kappa"] = kappa
                    outcome["max_isolated_fraction"] = cap
                    sens_records.append(outcome)
    sens_df = pd.DataFrame(sens_records)

    ranking_metrics = {
        "FairScore": ("mean_U_S_attack", True),
        "AUC_TSS": ("area_under_TSS", False),
        "time_to_containment": ("time_to_containment", False),
    }
    print("\nKendall tau-b between strategy rankings across kappa values "
          "(Gradual attack, 5 tuning seeds):")
    for metric, (col, higher) in ranking_metrics.items():
        rankings_by_kappa = {}
        for k in kappa_candidates:
            sub = sens_df[sens_df["kappa"] == k]
            rankings_by_kappa[k] = _rank_strategies(sub, col, higher)
        taus = []
        for i in range(len(kappa_candidates)):
            for j in range(i + 1, len(kappa_candidates)):
                taus.append(kendall_tau(rankings_by_kappa[kappa_candidates[i]],
                                        rankings_by_kappa[kappa_candidates[j]]))
        print(f"  {metric}:")
        for k, r in rankings_by_kappa.items():
            print(f"    kappa={k}: {' > '.join(r)}")
        print(f"    pairwise tau-b: {[round(t, 3) for t in taus]}")

    print("\nKendall tau-b between strategy rankings across max_isolated_fraction "
          "values (Gradual attack, 5 tuning seeds):")
    for metric, (col, higher) in ranking_metrics.items():
        rankings_by_cap = {}
        for c in cap_candidates:
            sub = sens_df[sens_df["max_isolated_fraction"] == c]
            rankings_by_cap[c] = _rank_strategies(sub, col, higher)
        taus = []
        for i in range(len(cap_candidates)):
            for j in range(i + 1, len(cap_candidates)):
                taus.append(kendall_tau(rankings_by_cap[cap_candidates[i]],
                                        rankings_by_cap[cap_candidates[j]]))
        print(f"  {metric}:")
        for c, r in rankings_by_cap.items():
            print(f"    cap={c}: {' > '.join(r)}")
        print(f"    pairwise tau-b: {[round(t, 3) for t in taus]}")

    return chosen, sens_df


OUTCOME_METRIC_COLUMNS_V3 = [
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
    "isolate_share",
    "pre_onset_isolate_share",
    # Phase 2b cost metrics (2b-3)
    "service_availability_mean",
    "service_availability_min",
    "isolated_device_ticks_total",
    "false_isolation_ticks",
    "max_isolated_devices",
    "net_disruption_cost",
    # per-phase (pre-onset vs attack)
    "normal_service_availability_mean",
    "attack_service_availability_mean",
    "normal_net_disruption_cost",
    "attack_net_disruption_cost",
    "normal_isolated_device_ticks",
    "attack_isolated_device_ticks",
    "normal_false_isolation_ticks",
    "attack_false_isolation_ticks",
    "attack_isolate_share",
    "attack_migrate_share",
    "attack_scale_down_share",
    "attack_no_op_share",
    "normal_isolate_share",
    "normal_migrate_share",
    "normal_scale_down_share",
    "normal_no_op_share",
    "us_identical_fraction",
]


def _official_v3_config():
    """Frozen v3 config used by the official run (kappa/cap from calibration)."""
    base_v3 = _get_v3_config()
    return deep_update(base_v3, {
        "actions": {
            "isolate": {
                "energy_saving": 0.0,
                "isolation_duration": _iso_duration(),
            },
        },
        "isolation_cap": {
            "max_isolated_fraction": _frozen_cap(),
        },
    })


def _v3_closed_loop_task(task):
    """One (scenario, seed) unit of work: all 8 strategies.

    Module-level so ProcessPoolExecutor (spawn) can pickle it on Windows.
    Each run seeds random/np at entry, so the task is a pure function of its
    arguments and worker scheduling cannot change any result.
    """
    scenario, seed, strategies, num_ticks, kappa, us_mode, ur_mode, twin_config, v3_config = task
    rows, outcomes = [], []
    for strategy in strategies:
        run_rows, outcome = run_v3_closed_loop(
            strategy, scenario, seed, kappa=kappa,
            twin_config=twin_config, v3_config=v3_config,
            num_ticks=num_ticks, us_mode=us_mode, ur_mode=ur_mode,
            tau=_tau(),
        )
        rows.extend(run_rows)
        outcomes.append(outcome)
    return rows, outcomes


def _load_v3_progress(raw_path, outcome_path, strategies, num_ticks,
                      us_mode, ur_mode, kappa):
    """Resume state from the CSVs of an interrupted v3 grid run."""
    if not (raw_path.exists() and outcome_path.exists()):
        return [], [], set()
    try:
        prev_out = pd.read_csv(outcome_path)
        prev_raw = pd.read_csv(raw_path)
    except Exception:
        return [], [], set()
    if not len(prev_out) or not len(prev_raw):
        return [], [], set()
    match = (
        (prev_out["us_mode"] == us_mode)
        & (prev_out["ur_mode"] == ur_mode)
        & (prev_out["num_ticks"].astype(int) == int(num_ticks))
    )
    if "kappa" in prev_out.columns:
        match = match & np.isclose(prev_out["kappa"].astype(float), float(kappa))
    prev_out = prev_out[match]
    wanted = set(strategies)
    done = {
        (str(scenario), int(seed))
        for (scenario, seed), grp in prev_out.groupby(["scenario", "seed"])
        if set(grp["strategy"]) == wanted
    }
    if not done:
        return [], [], set()
    raw_keep = np.array([(str(s), int(d)) in done
                         for s, d in zip(prev_raw["scenario"], prev_raw["seed"])])
    out_keep = np.array([(str(s), int(d)) in done
                         for s, d in zip(prev_out["scenario"], prev_out["seed"])])
    return (prev_raw[raw_keep].to_dict("records"),
            prev_out[out_keep].to_dict("records"), done)


def run_official(output_dir=None, workers=1, resume=True):
    """Official 4 x 8 x 20 x 100 closed-loop grid -> data/experiments_v3/closed_loop/.

    Keeps the parallel/resume runner and per-seed checkpoints of the v2 grid:
    tasks are (scenario, seed) pairs executed in canonical order, ``workers>1``
    uses a process pool and collects IN ORDER (CSVs identical to a serial run),
    and after every completed task the three CSVs are written directly as
    checkpoints, so an interrupted run resumes instead of restarting
    (``resume=True``).
    """
    from sim.attack_patterns_v2 import build_outcome_summary as _bos
    output_dir = output_dir or Path("data/experiments_v3/closed_loop")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "experiment_results_v3.csv"
    outcome_path = output_dir / "outcome_metrics_v3.csv"
    summary_path = output_dir / "outcome_summary_v3.csv"

    twin_cfg = get_twin_v2_config()
    v3_cfg = _official_v3_config()
    frozen_kappa = _frozen_kappa()
    frozen_cap = _frozen_cap()
    frozen_duration = _iso_duration()
    seeds = _official_seeds()

    print("=" * 78)
    print("Phase 2b official run (4 scenarios x 8 strategies x 20 seeds x 100 ticks)")
    print("=" * 78)
    print(f"Official seeds ({len(seeds)}): {seeds}")
    print("  (seed 1001 replaced by 12345 - it is inside the tuning range)")
    print("isolation_cost=v3, us_mode=v2, ur_mode=v3")
    print(f"kappa={frozen_kappa}, max_isolated_fraction={frozen_cap}, "
          f"isolation_duration={frozen_duration}")
    print()
    _print_official_header(twin_cfg, _get_v3_config(), frozen_kappa,
                           frozen_cap, frozen_duration)

    rows, outcomes, done = [], [], set()
    if resume:
        rows, outcomes, done = _load_v3_progress(
            raw_path, outcome_path, STRATEGIES_V3, _NUM_TICKS,
            "v2", "v3", frozen_kappa)
        if done:
            print(f"[grid] resume: {len(done)} (scenario, seed) pairs already done")

    tasks = [
        (scenario, seed, STRATEGIES_V3, _NUM_TICKS, frozen_kappa,
         "v2", "v3", twin_cfg, v3_cfg)
        for scenario in SCENARIOS_V3
        for seed in seeds
        if (scenario, seed) not in done
    ]
    total = len(tasks)
    started = time.perf_counter()

    def _checkpoint():
        results = pd.DataFrame(rows)
        outcome_df = pd.DataFrame(outcomes)
        summary_df = _bos(outcome_df, metric_columns=OUTCOME_METRIC_COLUMNS_V3)
        for frame, path in ((results, raw_path), (outcome_df, outcome_path),
                            (summary_df, summary_path)):
            frame.to_csv(path, index=False)
        return results, outcome_df, summary_df

    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            mapped = pool.map(_v3_closed_loop_task, tasks)
            for index, (task, (task_rows, task_outcomes)) in enumerate(
                    zip(tasks, mapped), start=1):
                rows.extend(task_rows)
                outcomes.extend(task_outcomes)
                print(f"  [{task[0]}] seed {task[1]} done "
                      f"({index}/{total}, {time.perf_counter()-started:.1f}s)",
                      flush=True)
                _checkpoint()
    else:
        for index, task in enumerate(tasks, start=1):
            task_rows, task_outcomes = _v3_closed_loop_task(task)
            rows.extend(task_rows)
            outcomes.extend(task_outcomes)
            print(f"  [{task[0]}] seed {task[1]} done "
                  f"({index}/{total}, {time.perf_counter()-started:.1f}s)",
                  flush=True)
            _checkpoint()

    results, outcome_df, summary_df = _checkpoint()
    print(f"\nSaved raw results to {raw_path} ({len(results)} rows)")
    print(f"Saved outcome metrics to {outcome_path} ({len(outcome_df)} rows)")
    print(f"Saved outcome summary to {summary_path} ({len(summary_df)} rows)")
    return outcome_df, summary_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--calibration", action="store_true",
                       help="Run calibration sandbox (tuning seeds 1000-1009).")
    group.add_argument("--official", action="store_true",
                       help="Run official 4x8x20x100 closed-loop grid.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Process-pool size for --calibration/--official "
                             "(default 1 = serial).")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing CSVs and re-run from scratch.")
    args = parser.parse_args()
    if args.calibration:
        if args.no_resume and CALIB_CSV.exists():
            CALIB_CSV.unlink()
        frozen, sens = calibration(workers=args.workers)
        print("\nCalibration complete. Frozen config printed above.")
        return
    if args.official:
        _, _ = run_official(workers=args.workers, resume=not args.no_resume)
        return


if __name__ == "__main__":
    main()
