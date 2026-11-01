import random
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import demo
from agents.resource_agent import score_resource
from agents.security_agent import compute_threat_severity, score_security
from agents.sustainability_agent import score_sustainability
from digital_twin.action_generator import generate_candidate_actions
from digital_twin.telemetry_generator import generate_telemetry
from digital_twin.threat_injector import inject_threat
from digital_twin.twin import DigitalTwin
from negotiation.cigi import compute_cigi
from negotiation.tapna import pareto_filter, run_tapna

TAU = 0.15
BASELINE_WS = 1 / 3
SCENARIO_OPTIONS = {key: name for key, name in demo.SCENARIOS.items()}

app = FastAPI(title="Multi-Agent Cloud-IoT Governance API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScenarioRequest(BaseModel):
    scenario: int


@contextmanager
def _deterministic_security_features(seed: int):
    import agents.security_agent as security_agent_module

    original_default_rng = security_agent_module.np.random.default_rng

    def seeded_default_rng(_seed=None):
        return security_agent_module.np.random.Generator(
            security_agent_module.np.random.PCG64(seed if _seed is None else _seed)
        )

    security_agent_module.np.random.default_rng = seeded_default_rng
    try:
        yield
    finally:
        security_agent_module.np.random.default_rng = original_default_rng


def _set_healthy_state(twin: DigitalTwin) -> None:
    for device in twin.devices:
        device.cpu_load = 0.35
        device.memory_load = 0.40
        device.network_load = 0.25
        device.trust_score = 0.95


def _apply_timeline_event(twin: DigitalTwin, choice: int) -> str:
    if choice == 2:
        for device in twin.devices:
            device.network_load = 0.85
            device.trust_score = 0.92
        return "High network load applied"
    if choice == 3:
        for device in twin.devices:
            device.network_load = 0.30
            device.trust_score = 0.45
        return "Trust degradation applied"
    if choice == 4:
        for device in twin.devices:
            inject_threat(twin, device.id, severity=0.5)
        return "Threat injected"
    if choice == 5:
        for device in twin.devices:
            device.cpu_load = 0.90
            device.memory_load = 0.92
            device.network_load = 0.30
            device.trust_score = 0.90
        return "Resource pressure applied"
    if choice == 6:
        for device in twin.devices:
            device.cpu_load = 0.88
            device.memory_load = 0.90
            device.network_load = 0.30
            device.trust_score = 0.90
        for device in twin.devices:
            inject_threat(twin, device.id, severity=0.5)
        return "Threat + resource pressure applied"
    return "Baseline operation"


def _has_security_attack(telemetry: dict) -> bool:
    return (
        telemetry["avg_trust_score"] < 0.7
        or telemetry["avg_network_load"] > 0.7
    )


def _scenario_candidate_actions(twin: DigitalTwin, telemetry: dict) -> list[dict]:
    actions = generate_candidate_actions(twin)
    if _has_security_attack(telemetry):
        return actions
    non_isolation_actions = [a for a in actions if a["type"] != "isolate"]
    return non_isolation_actions or [a for a in actions if a["type"] == "no_op"]


def _apply_scenario(twin: DigitalTwin, choice: int) -> str:
    if choice == 1:
        for d in twin.devices:
            d.cpu_load = 0.35
            d.memory_load = 0.40
            d.network_load = 0.25
            d.trust_score = 0.95
        return "CPU 0.35, Memory 0.40, Network 0.25, Trust 0.95."

    if choice == 2:
        for d in twin.devices:
            d.cpu_load = 0.35
            d.memory_load = 0.40
            d.network_load = 0.85
            d.trust_score = 0.92
        return "Every device network_load set to 0.85 (avg > 0.7)."

    if choice == 3:
        for d in twin.devices:
            d.cpu_load = 0.35
            d.memory_load = 0.40
            d.network_load = 0.30
            d.trust_score = 0.45
        return "Every device trust_score set to 0.45 (avg < 0.7)."

    if choice == 4:
        for d in twin.devices:
            d.cpu_load = 0.35
            d.memory_load = 0.40
            d.network_load = 0.30
            d.trust_score = 0.95
        for d in twin.devices:
            inject_threat(twin, d.id, severity=0.5)
        return "inject_threat(severity=0.5) applied to all 100 devices."

    if choice == 5:
        for d in twin.devices:
            d.cpu_load = 0.90
            d.memory_load = 0.92
            d.network_load = 0.30
            d.trust_score = 0.90
        return "CPU 0.90, Memory 0.92 -- high resource, no security threat."

    if choice == 6:
        for d in twin.devices:
            d.cpu_load = 0.88
            d.memory_load = 0.90
            d.network_load = 0.30
            d.trust_score = 0.90
        for d in twin.devices:
            inject_threat(twin, d.id, severity=0.5)
        return "CPU 0.88, Memory 0.90 + inject_threat(severity=0.5). High resource AND attack conditions."

    return ""


def _score_tick(twin: DigitalTwin, security_seed: int, prev_wS: float) -> dict:
    telemetry = generate_telemetry(twin)
    resource_state = twin.get_state()
    candidate_actions = _scenario_candidate_actions(twin, telemetry)

    with _deterministic_security_features(security_seed):
        u_r = [score_resource(action, resource_state) for action in candidate_actions]
        u_s = [score_security(action, telemetry) for action in candidate_actions]
        u_c = [score_sustainability(action, telemetry) for action in candidate_actions]
        tss = compute_threat_severity(telemetry)
        selected_action, weights = run_tapna(
            candidate_actions,
            u_r,
            u_s,
            u_c,
            tss,
            prev_wS=prev_wS,
        )

    selected_index = candidate_actions.index(selected_action)
    cigi = compute_cigi(
        u_r[selected_index],
        u_s[selected_index],
        u_c[selected_index],
        weights,
    )

    return {
        "telemetry": telemetry,
        "resource_state": resource_state,
        "candidate_actions": candidate_actions,
        "U_R_list": u_r,
        "U_S_list": u_s,
        "U_C_list": u_c,
        "TSS": tss,
        "weights": weights,
        "selected_action": selected_action,
        "selected_index": selected_index,
        "CIGI": cigi,
    }


def _build_insights(result: dict, avg_memory: float) -> list[str]:
    telemetry = result["telemetry"]
    weights = result["weights"]
    high_resource = telemetry["avg_cpu_load"] > 0.75 or avg_memory > 0.75
    high_network = telemetry["avg_network_load"] > 0.7
    low_trust = telemetry["avg_trust_score"] < 0.7
    dominant_weight = max(weights, key=weights.get)
    weakest_utility = min(
        [("resource", result["U_R"]), ("security", result["U_S"]), ("sustainability", result["U_C"])],
        key=lambda item: item[1],
    )

    if result["TSS"] > TAU:
        condition = (
            f"Infrastructure is in a threat-adaptive state: TSS {result['TSS']:.3f}, "
            f"trust {telemetry['avg_trust_score']:.3f}, network {telemetry['avg_network_load']:.3f}."
        )
    elif high_resource:
        condition = (
            f"Infrastructure is resource-constrained: CPU {telemetry['avg_cpu_load']:.3f}, "
            f"memory {avg_memory:.3f}, with TSS {result['TSS']:.3f}."
        )
    else:
        condition = (
            f"Infrastructure is nominal: trust {telemetry['avg_trust_score']:.3f}, "
            f"network {telemetry['avg_network_load']:.3f}, TSS {result['TSS']:.3f}."
        )

    if low_trust:
        concern = f"Dominant concern is reduced trust; security utility is {result['U_S']:.3f}."
    elif high_network:
        concern = (
            f"Dominant concern is network pressure at {telemetry['avg_network_load']:.3f}; "
            f"security utility is {result['U_S']:.3f}."
        )
    elif high_resource:
        concern = f"Dominant concern is resource pressure; resource utility is {result['U_R']:.3f}."
    else:
        concern = f"Dominant constraint is {weakest_utility[0]} utility at {weakest_utility[1]:.3f}."

    response = (
        f"TAPNA response: {dominant_weight} is highest at {weights[dominant_weight]:.3f}, "
        f"with wR {weights['wR']:.3f}, wS {weights['wS']:.3f}, wC {weights['wC']:.3f}."
    )
    action = (
        f"Governance action: {result['selected_action']['type']} with CIGI {result['CIGI']:.3f} "
        f"after combining U_R {result['U_R']:.3f}, U_S {result['U_S']:.3f}, and U_C {result['U_C']:.3f}."
    )

    return [condition, concern, response, action]


def _history(choice: int, tick_count: int = 12) -> list[dict]:
    seed = demo._seed_demo(choice)
    twin = DigitalTwin(100)
    _set_healthy_state(twin)
    prev_wS = BASELINE_WS
    rows = []
    event_label = None

    for tick in range(tick_count):
        if choice != 1 and tick == 4:
            event_label = _apply_timeline_event(twin, choice)
        elif tick:
            twin.tick()

        tick_result = _score_tick(twin, seed + 20_000 + tick, prev_wS)
        telemetry = tick_result["telemetry"]
        resource_state = tick_result["resource_state"]
        selected_action = tick_result["selected_action"]
        weights = tick_result["weights"]
        tss = tick_result["TSS"]
        cigi = tick_result["CIGI"]
        avg_memory = float(resource_state["memory_load"].mean())
        prev_wS = weights["wS"]

        rows.append(
            {
                "tick": tick,
                "TSS": float(tss),
                "wR": float(weights["wR"]),
                "wS": float(weights["wS"]),
                "wC": float(weights["wC"]),
                "cpu_load": float(telemetry["avg_cpu_load"]),
                "memory_load": float(avg_memory),
                "network_load": float(telemetry["avg_network_load"]),
                "trust_score": float(telemetry["avg_trust_score"]),
                "selected_action": selected_action["type"],
                "CIGI": float(cigi),
                "event": event_label if tick == 4 else "",
            }
        )

    return rows


def _simulate(choice: int) -> dict:
    seed = demo._seed_demo(choice)
    security_seed = seed + 10_000
    twin = DigitalTwin(100)
    description = _apply_scenario(twin, choice)
    telemetry = generate_telemetry(twin)
    resource_state = twin.get_state()
    candidate_actions = _scenario_candidate_actions(twin, telemetry)

    with _deterministic_security_features(security_seed):
        u_r = [score_resource(action, resource_state) for action in candidate_actions]
        u_s = [score_security(action, telemetry) for action in candidate_actions]
        u_c = [score_sustainability(action, telemetry) for action in candidate_actions]
        tss = compute_threat_severity(telemetry)
        selected_action, weights = run_tapna(
            candidate_actions,
            u_r,
            u_s,
            u_c,
            tss,
            prev_wS=BASELINE_WS,
        )

    selected_index = candidate_actions.index(selected_action)
    cigi = compute_cigi(u_r[selected_index], u_s[selected_index], u_c[selected_index], weights)
    avg_memory = float(resource_state["memory_load"].mean())
    result = {
        "scenario": choice,
        "scenario_name": demo.SCENARIOS[choice],
        "description": description,
        "device_count": int(len(twin.devices)),
        "telemetry": telemetry,
        "resource_state": resource_state,
        "U_R": float(u_r[selected_index]),
        "U_S": float(u_s[selected_index]),
        "U_C": float(u_c[selected_index]),
        "TSS": float(tss),
        "weights": {
            "wR": float(weights["wR"]),
            "wS": float(weights["wS"]),
            "wC": float(weights["wC"]),
        },
        "selected_action": {
            "type": selected_action["type"],
            "target_device_id": selected_action["target_device_id"],
            "id": selected_action["id"],
        },
        "CIGI": float(cigi),
        "candidate_actions": [
            {
                "id": action["id"],
                "type": action["type"],
                "target_device_id": action["target_device_id"],
                "U_R": float(u_r[i]),
                "U_S": float(u_s[i]),
                "U_C": float(u_c[i]),
            }
            for i, action in enumerate(candidate_actions)
        ],
        "insights": _build_insights(
            {
                "telemetry": telemetry,
                "U_R": u_r[selected_index],
                "U_S": u_s[selected_index],
                "U_C": u_c[selected_index],
                "TSS": tss,
                "weights": weights,
                "selected_action": selected_action,
                "CIGI": cigi,
            },
            avg_memory,
        ),
        "history": _history(choice),
        "status": "THREAT" if tss > TAU or telemetry["avg_trust_score"] < 0.7 or telemetry["avg_network_load"] > 0.7 else "NORMAL",
        "agent_status": {
            "resource": "High pressure" if telemetry["avg_cpu_load"] > 0.75 or avg_memory > 0.75 else "Within normal range",
            "security": "Threat condition active" if tss > TAU else "No elevated threat condition",
            "sustainability": "Cost and energy constrain this objective" if telemetry["total_cost_per_hour"] > 0 else "Awaiting cost and energy baseline",
        },
        "avg_memory": float(avg_memory),
    }

    return result


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/scenarios")
def scenarios():
    return {
        "scenarios": [
            {"id": key, "name": value}
            for key, value in sorted(SCENARIO_OPTIONS.items())
        ]
    }


@app.get("/api/scenario/{choice}")
def simulate_choice(choice: int):
    if choice not in SCENARIO_OPTIONS:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return _simulate(choice)


@app.post("/api/simulate")
def simulate(request: ScenarioRequest):
    if request.scenario not in SCENARIO_OPTIONS:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return _simulate(request.scenario)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=False)
