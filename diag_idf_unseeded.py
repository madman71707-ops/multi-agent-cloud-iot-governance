"""IDF 8.2 diagnostic, Phase 2b-7: the ORIGINAL unseeded telemetry behaviour.

Spec 2b-7: the previous doc attributed the 49/50 vs 46/50 normal-phase
difference to a "Pareto filter + normalised-WU change" in negotiation/tapna.py.
That explanation is wrong (the current file has no normalisation, and a Pareto
filter cannot change the argmax of a positive-weight weighted sum), so it was
deleted. The hypothesis to TEST instead is:

    the security-telemetry RNG was UNSEEDED when the IDF numbers were
    produced, and was later changed to the seeded global numpy stream.

Test (no assertion without evidence): restore the ORIGINAL telemetry noise
implementation - a fresh, unseeded ``np.random.default_rng()`` per telemetry
frame, exactly what agents/security_agent.py documents it replaced - and run
the canonical legacy experiment 30 independent times. Then report the
distribution of normal-phase and attack-phase isolate counts and say whether
46/50 and 50/50 lie inside them.

Canonical experiment (verbatim from sim/run_simulation.py): DigitalTwin(100),
100 ticks, deterministic injection at tick 50 of severity 0.5 on EVERY device,
legacy score_resource / score_security / score_sustainability, canonical TAPNA
(tau=0.15, lambda=0.30). Like run_simulation.main(), nothing is seeded: the
device universe and the candidate draws come from the ambient stream, and the
telemetry noise now comes from an unseeded default_rng() as it did originally.

Read-only: writes no CSVs, edits no canonical files, does NOT touch the IDF.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import agents.security_agent as sec
from agents.resource_agent import score_resource
from agents.security_agent import ThreatDetector, compute_threat_severity, score_security
from agents.sustainability_agent import score_sustainability
from digital_twin.action_generator import generate_candidate_actions
from digital_twin.telemetry_generator import generate_telemetry
from digital_twin.threat_injector import inject_threat
from digital_twin.twin import DigitalTwin
from negotiation.tapna import run_tapna

NUM_TICKS = 100
INJECTION_TICK = 50
ATTACK_SEVERITY = 0.5
N_RUNS = 30


def _original_telemetry_frame(twin_state):
    """The ORIGINAL _telemetry_frame: noise from a fresh UNSEEDED default_rng().

    Byte-for-byte the same feature construction as the current
    agents.security_agent._telemetry_frame, except for the RNG source, which
    is the one documented difference ("Draw from the global numpy stream
    instead of an unseeded np.random.default_rng()").
    """
    if isinstance(twin_state, pd.DataFrame):
        series = twin_state.mean() if len(twin_state) > 1 else twin_state.iloc[0]
        avg_trust = float(series.get("avg_trust_score", series.get("trust_score", 1.0)))
        avg_network = float(series.get("avg_network_load", series.get("network_load", 0.0)))
    else:
        avg_trust = float(twin_state.get("avg_trust_score", twin_state.get("trust_score", 1.0)))
        avg_network = float(twin_state.get("avg_network_load", twin_state.get("network_load", 0.0)))

    attack = avg_trust < 0.7 or avg_network > 0.7
    central = sec._ATTACK_CENTRAL if attack else sec._NORMAL_CENTRAL

    rng = np.random.default_rng()  # <-- the ORIGINAL, unseeded source
    features = {}
    for key in ThreatDetector.FEATURES:
        value = central[key] + rng.normal(0, abs(central[key]) * 0.05 + 0.01)
        features[key] = float(np.clip(value, sec._CLIP_MIN[key], sec._CLIP_MAX[key]))
    features["trust_score"] = avg_trust
    return pd.DataFrame([features])


def run_once():
    """One independent canonical run; returns (normal_iso, attack_iso)."""
    twin = DigitalTwin(100)
    prev_wS = 1 / 3
    normal_iso = attack_iso = 0
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
        if chosen_action["type"] == "isolate":
            if tick < INJECTION_TICK:
                normal_iso += 1
            else:
                attack_iso += 1
    return normal_iso, attack_iso


def _init_worker():
    # Every worker process patches the module once; the patch is global so
    # compute_threat_severity/score_security pick up the unseeded RNG.
    sec._telemetry_frame = _original_telemetry_frame


def main():
    print("=" * 78)
    print("IDF 8.2 diagnostic (Phase 2b-7): ORIGINAL UNSEEDED telemetry RNG")
    print("=" * 78)
    print(f"{N_RUNS} independent runs of the canonical legacy experiment")
    print(f"(injection at tick {INJECTION_TICK}, severity {ATTACK_SEVERITY}, all devices)")
    print("telemetry noise: np.random.default_rng() - fresh, unseeded per frame")
    print("nothing else is seeded (as in sim/run_simulation.py)")
    print()

    # Worker count is deliberately modest: OpenBLAS allocates a buffer per
    # process and this machine is often running the calibration grid at the
    # same time (too many processes -> "Memory allocation still failed").
    n_workers = 3
    if len(sys.argv) > 1:
        n_workers = max(1, int(sys.argv[1]))

    _init_worker()
    results = []
    started = time.perf_counter()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as pool:
        for i, (normal_iso, attack_iso) in enumerate(
                pool.map(_run_task, range(N_RUNS)), start=1):
            results.append((normal_iso, attack_iso))
            if i % 5 == 0 or i == N_RUNS:
                print(f"  {i}/{N_RUNS} runs done "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)

    normal = [r[0] for r in results]
    attack = [r[1] for r in results]
    normal_sorted, attack_sorted = sorted(normal), sorted(attack)
    n = len(normal)

    def _stats(values):
        v = sorted(values)
        median = v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2
        return min(v), median, max(v)

    n_min, n_med, n_max = _stats(normal)
    a_min, a_med, a_max = _stats(attack)

    print()
    print("=" * 78)
    print(f"DISTRIBUTION over {N_RUNS} independent runs (isolate counts /50)")
    print("=" * 78)
    print(f"  normal phase : min={n_min}  median={n_med}  max={n_max}")
    print(f"                 all values: {normal_sorted}")
    print(f"  attack phase : min={a_min}  median={a_med}  max={a_max}")
    print(f"                 all values: {attack_sorted}")
    print()
    print(f"  46/50 inside the NORMAL-phase distribution: "
          f"{n_min <= 46 <= n_max}")
    print(f"  50/50 inside the NORMAL-phase distribution: "
          f"{n_min <= 50 <= n_max}")
    print(f"  50/50 inside the ATTACK-phase distribution: "
          f"{a_min <= 50 <= a_max}")
    print(f"  46/50 inside the ATTACK-phase distribution: "
          f"{a_min <= 46 <= a_max}")
    print()
    print("INTERPRETATION")
    print("-" * 78)
    print("If 46/50 lies inside the normal-phase distribution produced by the")
    print("original unseeded telemetry RNG, the IDF's 46/50 is reproducible")
    print("under that RNG contract and the unseeded-telemetry hypothesis is")
    print("SUPPORTED (not proven - the original run log does not exist).")
    print("If it does not lie inside, the hypothesis is not supported and the")
    print("difference must be attributed to something else.")


def _run_task(index):
    return run_once()


if __name__ == "__main__":
    main()
