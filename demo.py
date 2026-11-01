"""
Multi-Agent Cloud-IoT Governance -- Interactive Terminal Demo
=============================================================
Demonstrates how the actual system (Digital Twin → Agents → TAPNA → CIGI)
produces different governance outputs for different operating conditions.

Run:  python demo.py
"""

import sys
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from faker import Faker

sys.path.insert(0, str(Path(__file__).resolve().parent))

from digital_twin.twin import DigitalTwin
from digital_twin.telemetry_generator import generate_telemetry
from digital_twin.action_generator import generate_candidate_actions
from digital_twin.threat_injector import inject_threat
from agents.resource_agent import score_resource
from agents.security_agent import compute_threat_severity, score_security
from agents.sustainability_agent import score_sustainability
from negotiation.tapna import run_tapna
from negotiation.cigi import compute_cigi
import agents.security_agent as security_agent_module

# ── Header ──────────────────────────────────────────────────────────────────

BANNER = r"""
====================================================================
   Multi-Agent Cloud-IoT Governance  --  Live Demo
   Digital Twin  |  AI Agents  |  TAPNA  |  CIGI
====================================================================
"""


def _divider():
    print("-" * 66)


# ── Scenario definitions ────────────────────────────────────────────────────

SCENARIOS = {
    1: "Normal Operation",
    2: "High Network Load",
    3: "Low Trust",
    4: "Controlled Cyber Threat",
    5: "High Resource Pressure",
    6: "Combined Threat + Resource Pressure",
}

DEMO_SEED = 20260918


@contextmanager
def _deterministic_security_features(seed: int):
    original_default_rng = security_agent_module.np.random.default_rng

    def seeded_default_rng(_seed=None):
        return np.random.Generator(np.random.PCG64(seed if _seed is None else _seed))

    security_agent_module.np.random.default_rng = seeded_default_rng
    try:
        yield
    finally:
        security_agent_module.np.random.default_rng = original_default_rng


def _seed_demo(choice: int) -> int:
    seed = DEMO_SEED + choice
    random.seed(seed)
    np.random.seed(seed)
    Faker.seed(seed)
    return seed


def _has_security_attack(telemetry: dict) -> bool:
    return (
        telemetry["avg_trust_score"] < 0.7
        or telemetry["avg_network_load"] > 0.7
    )


def _scenario_candidate_actions(twin: DigitalTwin, telemetry: dict) -> list[dict]:
    actions = generate_candidate_actions(twin)
    if _has_security_attack(telemetry):
        return actions

    # Healthy/resource-only demos should not offer isolation as a feasible response.
    non_isolation_actions = [a for a in actions if a["type"] != "isolate"]
    return non_isolation_actions or [a for a in actions if a["type"] == "no_op"]


def _apply_scenario(twin: DigitalTwin, choice: int) -> str:
    """Mutate the twin in-place so the chosen scenario is active.
    Returns a human-readable description of what was changed."""

    if choice == 1:
        # Normal demo state: healthy loads and high trust without attack triggers.
        for d in twin.devices:
            d.cpu_load = 0.35
            d.memory_load = 0.40
            d.network_load = 0.25
            d.trust_score = 0.95
        return "CPU 0.35, Memory 0.40, Network 0.25, Trust 0.95."

    if choice == 2:
        # Network-only pressure crosses the security attack trigger.
        for d in twin.devices:
            d.cpu_load = 0.35
            d.memory_load = 0.40
            d.network_load = 0.85
            d.trust_score = 0.92
        return "Every device network_load set to 0.85 (avg > 0.7)."

    if choice == 3:
        # Trust-only degradation crosses the security attack trigger.
        for d in twin.devices:
            d.cpu_load = 0.35
            d.memory_load = 0.40
            d.network_load = 0.30
            d.trust_score = 0.45
        return "Every device trust_score set to 0.45 (avg < 0.7)."

    if choice == 4:
        # Controlled cyber threat starts healthy, then uses the real injector.
        for d in twin.devices:
            d.cpu_load = 0.35
            d.memory_load = 0.40
            d.network_load = 0.30
            d.trust_score = 0.95
        for d in twin.devices:
            inject_threat(twin, d.id, severity=0.5)
        return "inject_threat(severity=0.5) applied to all 100 devices."

    if choice == 5:
        # High resource pressure (no security attack).
        for d in twin.devices:
            d.cpu_load = 0.90
            d.memory_load = 0.92
            d.network_load = 0.30
            d.trust_score = 0.90
        return "CPU 0.90, Memory 0.92 -- high resource, no security threat."

    if choice == 6:
        # Combined: resource pressure + cyber threat.
        for d in twin.devices:
            d.cpu_load = 0.88
            d.memory_load = 0.90
            d.network_load = 0.30
            d.trust_score = 0.90
        for d in twin.devices:
            inject_threat(twin, d.id, severity=0.5)
        return (
            "CPU 0.88, Memory 0.90 + inject_threat(severity=0.5). "
            "High resource AND attack conditions."
        )

    return ""


# ── Single scenario runner ──────────────────────────────────────────────────

def run_scenario(choice: int):
    security_seed = _seed_demo(choice) + 10_000
    twin = DigitalTwin(100)
    description = _apply_scenario(twin, choice)

    # --- pipeline (mirrors sim/run_simulation.py) ---
    telemetry = generate_telemetry(twin)
    twin_state = telemetry          # dict used by security & sustainability agents
    resource_state = twin.get_state()  # DataFrame used by resource agent

    candidate_actions = _scenario_candidate_actions(twin, telemetry)

    with _deterministic_security_features(security_seed):
        U_R_list = [score_resource(a, resource_state) for a in candidate_actions]
        U_S_list = [score_security(a, twin_state) for a in candidate_actions]
        U_C_list = [score_sustainability(a, twin_state) for a in candidate_actions]

        tss = compute_threat_severity(twin_state)

        chosen_action, weights = run_tapna(
            candidate_actions, U_R_list, U_S_list, U_C_list, tss, prev_wS=1 / 3,
        )
    idx = candidate_actions.index(chosen_action)
    cigi = compute_cigi(U_R_list[idx], U_S_list[idx], U_C_list[idx], weights)

    # --- display ---
    _divider()
    print(f"  Scenario : {SCENARIOS[choice]}")
    print(f"  Detail   : {description}")
    _divider()
    print(f"  Average Trust Score  : {telemetry['avg_trust_score']:.4f}")
    print(f"  Average Network Load : {telemetry['avg_network_load']:.4f}")
    print()
    print(f"  TSS                  : {tss:.4f}")
    print(f"  Resource Utility (U_R): {U_R_list[idx]:.4f}")
    print(f"  Security Utility (U_S): {U_S_list[idx]:.4f}")
    print(f"  Sustain. Utility (U_C): {U_C_list[idx]:.4f}")
    print()
    print(f"  wR                   : {weights['wR']:.4f}")
    print(f"  wS                   : {weights['wS']:.4f}")
    print(f"  wC                   : {weights['wC']:.4f}")
    print()
    print(f"  Selected Action      : {chosen_action['type']}")
    print(f"  CIGI                 : {cigi:.4f}")
    _divider()

    # --- adaptive explanation ---
    tau = 0.15
    if tss > tau:
        print(
            f"\n  [Adaptive Response]  TSS ({tss:.4f}) > threshold tau ({tau}). "
            "TAPNA increased the security weight wS."
        )
        print(
            f"                       wS = {weights['wS']:.4f} "
            f"(baseline 0.3333 + threat-driven increase)."
        )
    else:
        print(
            f"\n  [Adaptive Response]  TSS ({tss:.4f}) <= threshold tau ({tau}). "
            "Security weight remains near the balanced baseline."
        )
        print(
            f"                       wS = {weights['wS']:.4f} "
            f"(baseline 0.3333, decayed from any prior elevation)."
        )

    # --- resource-pressure note ---
    avg_cpu = resource_state["cpu_load"].mean()
    avg_mem = resource_state["memory_load"].mean()
    if avg_cpu > 0.75 or avg_mem > 0.75:
        print(
            f"\n  [Resource Note]  Avg CPU {avg_cpu:.4f}, "
            f"Memory {avg_mem:.4f} -- high resource pressure detected."
        )

    # --- security detail ---
    attack = _has_security_attack(telemetry)
    if attack:
        print(
            "\n  [Security Detail]  Attack condition triggered "
            "(trust < 0.7 OR network > 0.7)."
        )
        print(
            "                     Security telemetry shifted to "
            "attack-feature regime."
        )
    else:
        print(
            "\n  [Security Detail]  No attack condition. "
            "Security telemetry in normal-feature regime."
        )

    print()


# ── Main menu loop ──────────────────────────────────────────────────────────

def main():
    print(BANNER)
    print("  This demo runs the REAL governance pipeline.\n")
    print("  Each scenario applies a different Digital Twin state and")
    print("  feeds it through the actual Agents, TAPNA, and CIGI.\n")

    while True:
        _divider()
        print("  Select a scenario:\n")
        for k, v in SCENARIOS.items():
            print(f"    {k}. {v}")
        print(f"    0. Exit")
        _divider()

        try:
            raw = input("\n  Your choice: ").strip()
            choice = int(raw)
        except (ValueError, EOFError):
            print("\n  Please enter a number 0-6.\n")
            continue

        if choice == 0:
            print("\n  Goodbye.\n")
            break

        if choice not in SCENARIOS:
            print("\n  Unknown choice. Enter 0-6.\n")
            continue

        run_scenario(choice)


if __name__ == "__main__":
    main()
