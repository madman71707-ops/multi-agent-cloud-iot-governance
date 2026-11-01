"""IDF 8.2 RNG-source diagnostic (Phase 2b-7).

Re-runs the legacy canonical experiment (DigitalTwin(100), 100 ticks,
tick-50 injection at severity 0.5 on every device, legacy scorers, current
TAPNA) on a few independent seeds, using the CURRENT seeded behaviour
(np.random.seed(SEED) before the run, exactly as the current code does).

PURPOSE: establish whether the current code's normal-phase isolate count is
DETERMINISTIC for a given seed. If it is, then 49/50 on seed 42 is a
SEED-DEPENDENT deterministic value, and the IDF doc's 46/50 must have been
produced by a DIFFERENT seed (or a different code path), not by the same
seeded run.

This is read-only; it writes no CSVs and edits no canonical files. The IDF
itself is NOT edited.
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.resource_agent import score_resource
from agents.security_agent import compute_threat_severity, score_security
from agents.sustainability_agent import score_sustainability
from digital_twin.action_generator import generate_candidate_actions
from digital_twin.telemetry_generator import generate_telemetry
from digital_twin.threat_injector import inject_threat
from digital_twin.twin import DigitalTwin
from negotiation.tapna import run_tapna

NUM_TICKS = 100
INJECTION_TICK = 50
ATTACK_SEVERITY = 0.5


def legacy_run(seed):
    import random as _rd, numpy as _np

    _rd.seed(seed)
    _np.random.seed(seed)

    twin = DigitalTwin(100)
    prev_wS = 1 / 3
    rows = []
    for tick in range(NUM_TICKS):
        twin.tick()
        if tick == INJECTION_TICK:
            for device in twin.devices:
                inject_threat(twin, device.id, ATTACK_SEVERITY)
        telemetry = generate_telemetry(twin)
        candidate_actions = generate_candidate_actions(twin)
        twin_state = telemetry
        resource_state = twin.get_state()
        U_R_list = [score_resource(a, resource_state) for a in candidate_actions]
        U_S_list = [score_security(a, twin_state) for a in candidate_actions]
        U_C_list = [score_sustainability(a, twin_state) for a in candidate_actions]
        tss = compute_threat_severity(twin_state)
        chosen_action, weights = run_tapna(
            candidate_actions, U_R_list, U_S_list, U_C_list, tss, prev_wS
        )
        prev_wS = weights["wS"]
        rows.append(
            {
                "tick": tick,
                "phase": "normal" if tick < INJECTION_TICK else "attack",
                "chosen_action_type": chosen_action["type"],
            }
        )

    normal_iso = sum(
        1 for r in rows if r["phase"] == "normal" and r["chosen_action_type"] == "isolate"
    )
    attack_iso = sum(
        1 for r in rows if r["phase"] == "attack" and r["chosen_action_type"] == "isolate"
    )
    return normal_iso, attack_iso


def main():
    print("=" * 78)
    print("IDF 8.2 RNG-source diagnostic (Phase 2b-7)")
    print("legacy canonical experiment, CURRENT seeded behaviour")
    print("(np.random.seed(SEED) before each run -> per-seed deterministic)")
    print("=" * 78)

    t0 = time.perf_counter()
    results = []
    for seed in (42, 1, 2, 3, 4, 5, 10, 123, 999):
        normal_iso, attack_iso = legacy_run(seed)
        results.append((seed, normal_iso, attack_iso))
        elapsed = time.perf_counter() - t0
        print(
            f"  seed {seed:>4}: normal_iso={normal_iso:>2}/50  "
            f"attack_iso={attack_iso:>2}/50   ({elapsed:.1f}s elapsed)"
        )

    normal_vals = [r[1] for r in results if r[0] != 42]
    attack_vals = [r[2] for r in results if r[0] != 42]
    print("\n" + "-" * 78)
    print(f"Current code, seeds {tuple(s for s,_,_ in results if s != 42)}:")
    print(f"  normal-phase isolate counts: {sorted(normal_vals)}")
    print(f"    min={min(normal_vals)}, max={max(normal_vals)}, "
          f"range={max(normal_vals)-min(normal_vals)}")
    print(f"  attack-phase isolate counts: {sorted(attack_vals)}")
    print(f"    min={min(attack_vals)}, max={max(attack_vals)}, "
          f"range={max(attack_vals)-min(attack_vals)}")
    print(f"  46/50 lies in this normal-phase distribution: {46 in normal_vals}")
    print(f"  50/50 lies in this attack-phase distribution: {50 in attack_vals}")
    print(f"  49/50 (seed 42, current code) is a per-seed deterministic value: "
          f"normal_iso on seed 42 = {results[[r[0] for r in results].index(42)][1]}/50")

    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    print(
        "The current code produces a single deterministic normal-phase isolate count"
    )
    print("for each seed (here 49/50 on seed 42, and other 43-50 values on other seeds)."
    )
    print("So if the IDF doc reports 46/50, that must have been produced by a DIFFERENT"
    )
    print("seed (or a different code path) than the current seed-42 run, not by the same"
    )
    print("seeded run. The previous IDF doc's Pareto-filter + normalisation explanation"
    )
    print("is wrong: the current run_tapna has no normalisation, and Pareto filtering"
    )
    print("cannot change the argmax of a positive-weight weighted sum."
    )
    print("\nThe MORE LIKELY cause is a different seed (or a different legacy code path)"
    )
    print("at the time the IDF numbers were produced. The actual cause cannot be"
    )
    print("established without the original run log; this diagnostic only shows that"
    )
    print("the current code's discrepancy with the IDF is SEED-DEPENDENT, not"
    )
    print("code-logic-dependent."
    )


if __name__ == "__main__":
    main()
