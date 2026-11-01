"""IDF section 8.2 reproduction (spec 2f) - read-only diagnostic.

Re-runs the canonical legacy experiment (sim/run_simulation.py protocol:
DigitalTwin(100), 100 ticks, deterministic threat injection at tick 50
pushing EVERY device by severity 0.5, TAPNA negotiation, legacy
scorers) with seed 42, and prints what the CURRENT code gives for:

  (i)  action counts per phase  (normal = ticks 0-49, attack = ticks 50-99)
  (ii) mean TSS and mean w_S per phase

The IDF (section 8.2) claims: after the tick-50 injection, TAPNA selected
`isolate` in 50 of 50 attack ticks, and in 46 of 50 normal ticks.

Writes no CSVs and touches no canonical files (same contract as
diag_phase0.py / diag_followup.py); output goes to stdout and is quoted in
docs/idf_reproduction.md. The IDF itself is NOT edited.
"""
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from digital_twin.action_generator import generate_candidate_actions
from digital_twin.telemetry_generator import generate_telemetry
from digital_twin.threat_injector import inject_threat
from digital_twin.twin import DigitalTwin
from agents.resource_agent import score_resource
from agents.security_agent import compute_threat_severity, score_security
from agents.sustainability_agent import score_sustainability
from negotiation.cigi import compute_cigi
from negotiation.tapna import run_tapna

SEED = 42
NUM_TICKS = 100
INJECTION_TICK = 50
ATTACK_SEVERITY = 0.5  # same constant as sim/run_simulation.py


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    twin = DigitalTwin(100)
    prev_wS = 1 / 3
    rows = []

    for tick in range(NUM_TICKS):
        twin.tick()

        # Controlled attack at tick 50: inject every device (canonical).
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
        cigi = compute_cigi(
            U_R_list[candidate_actions.index(chosen_action)],
            U_S_list[candidate_actions.index(chosen_action)],
            U_C_list[candidate_actions.index(chosen_action)],
            weights,
        )
        prev_wS = weights["wS"]

        rows.append(
            {
                "tick": tick,
                "phase": "normal" if tick < INJECTION_TICK else "attack",
                "chosen_action_type": chosen_action["type"],
                "TSS": float(tss),
                "wS": float(weights["wS"]),
                "cigi": float(cigi),
            }
        )

    # ---- report ----------------------------------------------------------
    print("=" * 78)
    print(f"IDF 8.2 reproduction - legacy canonical config, seed {SEED}")
    print("protocol: sim/run_simulation.py (injection at tick 50, severity 0.5)")
    print("=" * 78)
    for phase in ("normal", "attack"):
        subset = [r for r in rows if r["phase"] == phase]
        counts = Counter(r["chosen_action_type"] for r in subset)
        n = len(subset)
        print(f"\n--- {phase.upper()} phase (ticks "
              f"{subset[0]['tick']}-{subset[-1]['tick']}, n={n}) ---")
        print("action counts:", dict(counts.most_common()))
        for action_type, count in counts.most_common():
            print(f"  {action_type:<12} {count:>3}/{n}  ({count / n:.2f})")
        mean_tss = float(np.mean([r["TSS"] for r in subset]))
        mean_ws = float(np.mean([r["wS"] for r in subset]))
        print(f"  mean TSS = {mean_tss:.4f} | mean w_S = {mean_ws:.4f}")

    print("\n--- per-tick excerpt around the injection (ticks 46-56) ---")
    print(f"{'tick':>4} {'type':<12} {'TSS':>7} {'wS':>7} {'CIGI':>7}")
    for r in rows[46:57]:
        print(
            f"{r['tick']:>4} {r['chosen_action_type']:<12} "
            f"{r['TSS']:>7.4f} {r['wS']:>7.4f} {r['cigi']:>7.4f}"
        )

    attack_iso = sum(
        1 for r in rows if r["phase"] == "attack"
        and r["chosen_action_type"] == "isolate"
    )
    normal_iso = sum(
        1 for r in rows if r["phase"] == "normal"
        and r["chosen_action_type"] == "isolate"
    )
    print("\n--- IDF 8.2 claims vs current code ---")
    print(f"attack-phase isolate: claimed 50/50, current {attack_iso}/50")
    print(f"normal-phase isolate: claimed 46/50, current {normal_iso}/50")
    return rows


if __name__ == "__main__":
    main()
