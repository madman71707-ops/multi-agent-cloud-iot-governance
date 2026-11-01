"""Phase 0 read-only diagnostics (audit questions a, b, c, e, f, g).

Single seed (42) plus two extra seeds for the std-semantics check. Each
environment trace is simulated exactly once and reused. No CSV output, no
repo changes. Run: python -u diag_phase0.py > diag_out.txt 2>&1
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from negotiation.tapna import compute_dynamic_weight, pareto_filter
from sim.attack_patterns import (
    ATTACK_SCENARIOS,
    _iterate_scenario_environment,
    _replay_strategy_on_trace,
    build_attack_response_summary,
    gradual_attack_schedule,
    short_spike_schedule,
    repeated_attack_schedule,
    sustained_attack_schedule,
)

SEED = 42
LINE = "=" * 78


def section(title):
    print(f"\n{LINE}\n{title}\n{LINE}", flush=True)


def build_trace(schedule, seed, num_ticks):
    return list(_iterate_scenario_environment(schedule, seed, num_ticks))


def trace_fingerprint(trace):
    return round(sum(td["tss"] for td in trace), 12)


# ---------------------------------------------------------------------------
# (b) TAPNA on Gradual attack, seed 42: per-tick TSS / wS / chosen type,
#     plus a full candidate dump for one representative tick.
# ---------------------------------------------------------------------------
def question_b(trace):
    section("(b) TAPNA, Gradual attack (100-tick schedule), seed 42: per-tick view")
    prev_wS = 1 / 3
    rows = []
    for tick, td in enumerate(trace):
        actions = td["actions"]
        chosen, weights, prev_wS = _run_strategy_tick_tapna(
            actions, td["U_R_list"], td["U_S_list"], td["U_C_list"],
            td["tss"], prev_wS,
        )
        idx = actions.index(chosen)
        best_us_idx = int(np.argmax(td["U_S_list"]))
        rows.append({
            "tick": tick,
            "TSS": td["tss"],
            "wS": weights["wS"],
            "branch": "ramp" if td["tss"] > 0.15 else "decay",
            "n_cand": len(actions),
            "chosen_type": chosen["type"],
            "chosen_idx": idx,
            "isolate_is_US_argmax": actions[best_us_idx]["type"] == "isolate",
        })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    n_iso = int((df["chosen_type"] == "isolate").sum())
    print(f"\nTAPNA chose isolate in {n_iso}/{len(df)} ticks "
          f"(defensive-response rate {n_iso/len(df):.3f} over these attack ticks).")
    print(f"Ticks where the max-U_S candidate IS an isolate: "
          f"{int(df['isolate_is_US_argmax'].sum())}/{len(df)}.")

    # Representative tick: first tick with TSS >= 0.5 where TAPNA did NOT
    # choose isolate (the interesting regime); fall back to tick 50.
    interesting = df[(df["TSS"] >= 0.5) & (df["chosen_type"] != "isolate")]
    tick = int(interesting["tick"].iloc[0]) if len(interesting) else 50
    td = trace[tick]
    rec = df.iloc[tick]
    w = {"wR": (1 - rec["wS"]) / 2, "wS": rec["wS"], "wC": (1 - rec["wS"]) / 2}
    front = pareto_filter(
        td["actions"], {"U_R": td["U_R_list"], "U_S": td["U_S_list"],
                        "U_C": td["U_C_list"]})
    front_ids = {a["id"] for a in front}
    print(f"\nFull candidate dump, tick {tick} "
          f"(TSS={td['tss']:.4f}, severity={td['severity']:.4f}, "
          f"wS={w['wS']:.4f}, wR=wC={w['wR']:.4f}, "
          f"pareto front size={len(front_ids)}/{len(td['actions'])}):")
    print(f"{'idx':>3} {'type':<11} {'U_R':>7} {'U_S':>7} {'U_C':>7} "
          f"{'wS*dU_S':>8} {'(wR+wC)*d(R+C)':>15} {'score':>7} {'pareto':>6}")
    no_op_idx = next(i for i, a in enumerate(td["actions"])
                     if a["type"] == "no_op")
    for i, a in enumerate(td["actions"]):
        u_r, u_s, u_c = td["U_R_list"][i], td["U_S_list"][i], td["U_C_list"][i]
        score = w["wR"] * u_r + w["wS"] * u_s + w["wC"] * u_c
        d_us = u_s - td["U_S_list"][no_op_idx]
        d_rc = (u_r + u_c) - (td["U_R_list"][no_op_idx]
                              + td["U_C_list"][no_op_idx])
        print(f"{i:>3} {a['type']:<11} {u_r:>7.4f} {u_s:>7.4f} {u_c:>7.4f} "
              f"{w['wS'] * d_us:>8.4f} {w['wR'] * d_rc:>15.4f} {score:>7.4f} "
              f"{str(a['id'] in front_ids):>6}")
    chosen_idx = int(rec["chosen_idx"])
    print(f"chosen idx = {chosen_idx} ({td['actions'][chosen_idx]['type']}); "
          f"no_op idx = {no_op_idx}")
    print("score = wR*U_R + wS*U_S + wC*U_C; dU_S/d(R+C) measured vs no_op.")
    print("The isolate candidate wins only if wS*d(U_S) beats the residual")
    print("weight advantage (wR+wC)*d(U_R+U_C) of resource-friendlier picks.")
    return df


def _run_strategy_tick_tapna(actions, U_R_list, U_S_list, U_C_list, tss,
                             prev_wS):
    """Canonical TAPNA step (tau=0.15, lambda=0.3), same as _run_strategy_tick."""
    from negotiation.tapna import run_tapna
    chosen_action, weights = run_tapna(
        actions, U_R_list, U_S_list, U_C_list, tss, prev_wS,
    )
    return chosen_action, weights, weights["wS"]


# ---------------------------------------------------------------------------
# (c) Is gradual actually gradual? Per-tick TSS ramp, seed 42, 100 ticks.
# ---------------------------------------------------------------------------
def question_c(gradual, sustained):
    section("(c) TSS ramp: Gradual vs Sustained, seed 42, 100 ticks")
    for name, trace in (("Gradual", gradual), ("Sustained", sustained)):
        tss = np.array([td["tss"] for td in trace])
        sev = np.array([td["severity"] for td in trace])
        print(f"\n{name}: severity per tick {sev[0]:.3f} -> {sev[-1]:.3f}")
        print("tick: " + " ".join(f"{t:>6}" for t in range(0, 100, 5)))
        print("TSS : " + " ".join(f"{tss[t]:>6.3f}" for t in range(0, 100, 5)))
        print(f"mean TSS = {tss.mean():.4f}  min = {tss.min():.4f}  "
              f"max = {tss.max():.4f}")
        target = 0.95 * tss.max()
        reach = next((t for t in range(100) if tss[t] >= target), None)
        print(f"TSS first reaches 95% of max at tick {reach}; "
              f"first-20 mean = {tss[:20].mean():.4f}, "
              f"last-20 mean = {tss[-20:].mean():.4f}")


# ---------------------------------------------------------------------------
# (e) Candidates per tick: sizes, composition, per-device vs global.
# ---------------------------------------------------------------------------
def question_e(gradual):
    section("(e) Candidate composition: Gradual attack, seed 42, 100 ticks")
    sizes = [len(td["actions"]) for td in gradual]
    n_noop_ok = sum(1 for td in gradual
                    if sum(a["type"] == "no_op" for a in td["actions"]) == 1)
    iso_counts = [sum(a["type"] == "isolate" for a in td["actions"])
                  for td in gradual]
    types = {}
    for td in gradual:
        for a in td["actions"]:
            types[a["type"]] = types.get(a["type"], 0) + 1
    print(f"candidates per tick: min={min(sizes)}, max={max(sizes)}, "
          f"mean={np.mean(sizes):.2f} (generator: random.randint(5, 10))")
    print(f"ticks with exactly one no_op candidate: {n_noop_ok}/100")
    print(f"isolate candidates per tick: min={min(iso_counts)}, "
          f"max={max(iso_counts)}")
    print(f"action-type counts over 100 ticks "
          f"(total candidates = {sum(types.values())}): {types}")
    print("candidate list is GLOBAL per tick (one list, targets chosen at")
    print("random among the 100 devices); the first isolate always targets")
    print("the lowest-trust device (action_generator.py).")


# ---------------------------------------------------------------------------
# (f) tau=0.40: which branch triggers where? All 4 scenarios + canonical.
# ---------------------------------------------------------------------------
def question_f(traces):
    section("(f) Branch analysis: tau=0.40 vs tau=0.15, seed 42, 100 ticks")
    for name, trace in traces.items():
        tss = np.array([td["tss"] for td in trace])
        for tau in (0.40, 0.15):
            prev = 1 / 3
            ws = []
            for t in tss:
                prev = compute_dynamic_weight(float(t), tau=tau, prev_wS=prev)
                ws.append(prev)
            ws = np.array(ws)
            ramp = int((tss > tau).sum())
            wramp = ws[tss > tau].mean() if ramp else float("nan")
            wdec = ws[tss <= tau].mean() if (100 - ramp) else float("nan")
            print(f"{name:<16} tau={tau:.2f}: ramp ticks {ramp:>3}/100, "
                  f"mean wS = {ws.mean():.4f}, wS|ramp = {wramp:.4f}, "
                  f"wS|decay = {wdec:.4f}")
        print(f"{name:<16} TSS: max = {tss.max():.4f}, mean = {tss.mean():.4f}")


# ---------------------------------------------------------------------------
# (g) How is the reported std computed? Pooled vs per-seed, on a mini run.
# ---------------------------------------------------------------------------
def question_g():
    section("(g) std semantics: pooled tick x seed rows vs per-seed means")
    seeds = [42, 123, 456]
    rows = []
    for seed in seeds:
        trace = build_trace(gradual_attack_schedule(20), seed, 20)
        rows.extend(_replay_strategy_on_trace(trace, "TAPNA",
                                              "Gradual attack", seed))
    results = pd.DataFrame(rows)
    summary = build_attack_response_summary(results)
    pooled = float(summary.loc[0, "TSS_std"])
    per_seed = results.groupby("seed")["TSS"].mean()
    print("mini run: 3 seeds x 20 ticks, TAPNA, Gradual attack")
    print("per-seed mean TSS: " + ", ".join(f"{v:.4f}" for v in per_seed))
    print(f"pooled std over all 60 rows (what the summary reports): {pooled:.4f}")
    print(f"std across the 3 per-seed means: {per_seed.std(ddof=1):.4f}")
    print(f"mean of per-seed stds: "
          f"{results.groupby('seed')['TSS'].std(ddof=1).mean():.4f}")
    print("=> build_attack_response_summary pools ALL tick x seed rows "
          "(ddof=1); NOT the std across per-seed means.")


# ---------------------------------------------------------------------------
# (a) strategy-level defensive rate + action mix on a mini run.
# ---------------------------------------------------------------------------
def question_a_mix():
    section("(a) mini run: defensive rate + action mix by strategy/phase")
    seeds = [42, 123]
    rows = []
    for name, schedule, seed in [
        ("Gradual attack", gradual_attack_schedule(20), 42),
        ("Gradual attack", gradual_attack_schedule(20), 123),
        ("Sustained attack", sustained_attack_schedule(20), 42),
        ("Sustained attack", sustained_attack_schedule(20), 123),
    ]:
        trace = build_trace(schedule, seed, 20)
        for strategy in ("TAPNA", "Greedy Latency", "Static Weighted"):
            rows.extend(_replay_strategy_on_trace(trace, strategy, name, seed))
    results = pd.DataFrame(rows)
    summary = build_attack_response_summary(results)
    cols = ["scenario", "strategy", "defensive_response_rate",
            "mean_security_response_gain", "mean_response_lag",
            "never_responded_runs"]
    print(summary[cols].to_string(index=False))
    print("\naction mix (share of chosen types, attack ticks only):")
    attacked = results[results["under_attack"]]
    mix = (attacked.groupby(["strategy"])["chosen_action_type"]
           .value_counts(normalize=True).unstack(fill_value=0.0))
    print(mix.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    print("building traces ...", flush=True)
    gradual100 = build_trace(gradual_attack_schedule(100), SEED, 100)
    print("  gradual done", flush=True)
    sustained100 = build_trace(sustained_attack_schedule(100), SEED, 100)
    print("  sustained done", flush=True)
    spike100 = build_trace(short_spike_schedule(100), SEED, 100)
    print("  spike done", flush=True)
    repeated100 = build_trace(repeated_attack_schedule(100), SEED, 100)
    print("  repeated done", flush=True)
    canonical100 = build_trace({50: 0.5}, SEED, 100)
    print("  canonical done", flush=True)

    # (d) probe: same seed -> identical trace (env independent of any action).
    gradual_check = build_trace(gradual_attack_schedule(100), SEED, 100)
    assert trace_fingerprint(gradual_check) == trace_fingerprint(gradual100)
    print("\n(d) probe: re-simulating seed 42 reproduces the gradual trace "
          "exactly (TSS sum identical); no code path passes a chosen action "
          "back into the twin (see audit doc).", flush=True)

    question_b(gradual100)
    question_c(gradual100, sustained100)
    question_e(gradual100)
    question_f({
        "Gradual": gradual100,
        "Spike": spike100,
        "Repeated": repeated100,
        "Sustained": sustained100,
        "Canonical {50}": canonical100,
    })
    question_g()
    question_a_mix()
    print("\nALL DIAGNOSTICS DONE", flush=True)
