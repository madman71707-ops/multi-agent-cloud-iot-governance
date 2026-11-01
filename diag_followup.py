"""Focused follow-up probes for audit (b) and (f).

(b) In high-TSS ticks, does U_S collapse to 0 for ALL candidates (flat
    security objective), making TAPNA's argmax identical to Static
    Weighted's? Count flat ticks, agreement rate, U_S spreads.
(f) After the tick-50 injection, does the twin state recover? Print avg
    trust / avg network per tick around the injection to show saturation
    and the absence of any recovery mechanism.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from negotiation.baselines import static_weighted_baseline
from sim.attack_patterns import (
    _iterate_scenario_environment,
    _run_strategy_tick,
    gradual_attack_schedule,
    sustained_attack_schedule,
)

SEED = 42


def probe_b(trace, name):
    prev = 1 / 3
    n_flat = 0
    n_agree = 0
    n_high = 0
    spreads = []
    for tick, td in enumerate(trace):
        actions = td["actions"]
        us = np.array(td["U_S_list"])
        chosen, weights, prev = _run_strategy_tick(
            "TAPNA", actions, td["U_R_list"], td["U_S_list"],
            td["U_C_list"], td["tss"], prev,
        )
        static = static_weighted_baseline(
            actions, td["U_R_list"], td["U_S_list"], td["U_C_list"],
        )
        flat = float(us.max()) == 0.0
        n_flat += flat
        if td["tss"] >= 0.5:
            n_high += 1
            n_agree += chosen["id"] == static["id"]
            spreads.append(float(us.max() - us.min()))
        if tick < 8 or flat != (td["tss"] >= 0.5):
            print(f"  {name} tick {tick:>3}: TSS={td['tss']:.4f} "
                  f"maxU_S={us.max():.6f} minU_S={us.min():.6f} "
                  f"flat={flat} TAPNA={chosen['type']} "
                  f"Static={static['type']} agree={chosen['id'] == static['id']}")
    print(f"{name}: flat-U_S ticks (max U_S == 0): {n_flat}/100; "
          f"TAPNA==StaticWeighted choice on TSS>=0.5 ticks: {n_agree}/{n_high}; "
          f"mean U_S spread on those ticks: {np.mean(spreads) if spreads else float('nan'):.6f}")


def probe_f():
    print("\n(f) twin recovery probe: canonical {50} schedule, seed 42")
    trace = list(_iterate_scenario_environment({50: 0.5}, SEED, 100))
    print("tick   avg_trust  avg_network   TSS")
    for tick, td in enumerate(trace):
        if tick in (45, 48, 49, 50, 51, 52, 55, 60, 70, 80, 90, 99):
            acts = td["actions"]
            _ = acts  # trace carries tss only; recompute telemetry cheaply
            print(f"{tick:>4}   (TSS={td['tss']:.4f})")
    # avg trust/network are not in the trace; recompute state separately.
    import random
    from digital_twin.twin import DigitalTwin
    from digital_twin.threat_injector import inject_threat
    from digital_twin.telemetry_generator import generate_telemetry
    random.seed(SEED)
    np.random.seed(SEED)
    twin = DigitalTwin(100)
    print("tick   avg_trust  avg_network   TSS-ish")
    for tick in range(100):
        twin.tick()
        if tick == 50:
            for device in twin.devices:
                inject_threat(twin, device.id, 0.5)
            inject_threat(twin, device.id, 0.5)
        state = generate_telemetry(twin)
        if tick in (45, 48, 49, 50, 51, 52, 55, 60, 70, 80, 90, 99):
            print(f"{tick:>4}   {state['avg_trust_score']:.4f}     "
                  f"{state['avg_network_load']:.4f}")


if __name__ == "__main__":
    print("building gradual trace ...", flush=True)
    gradual = list(_iterate_scenario_environment(
        gradual_attack_schedule(100), SEED, 100))
    print("building sustained trace ...", flush=True)
    sustained = list(_iterate_scenario_environment(
        sustained_attack_schedule(100), SEED, 100))
    print("\n=== (b) flat-U_S / agreement probe ===", flush=True)
    probe_b(gradual, "Gradual")
    print()
    probe_b(sustained, "Sustained")
    probe_f()
    print("\nFOLLOWUP DONE", flush=True)
