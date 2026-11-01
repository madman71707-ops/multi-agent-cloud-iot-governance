"""Standalone, in-memory cloud-IoT governance simulation for local demos.

This module does not connect to cloud resources. Resource, security, and
sustainability utilities are transparent demonstration formulas; adaptive
weights/action selection and CIGI reuse the project's TAPNA/CIGI functions.
Run locally with ``uvicorn backend.stateful_demo:app --reload``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from negotiation.cigi import compute_cigi
from negotiation.tapna import run_tapna

SCENARIOS: dict[str, dict[str, Any]] = {
    "normal_operation": {
        "name": "Normal Operation",
        "description": "Healthy, trusted infrastructure with low resource use.",
        "initial": {
            "cpu_load": 0.35, "memory_load": 0.40, "network_load": 0.25,
            "trust": 0.95, "attack_persistence": 0.0,
            "attack_active": False, "capacity": 0.82,
        },
    },
    "high_network_load": {
        "name": "High Network Load",
        "description": "A network bottleneck with otherwise healthy trust.",
        "initial": {
            "cpu_load": 0.42, "memory_load": 0.46, "network_load": 0.86,
            "trust": 0.92, "attack_persistence": 0.0,
            "attack_active": False, "capacity": 0.82,
        },
    },
    "low_trust": {
        "name": "Low Trust",
        "description": "Compromised trust without an active simulated attack.",
        "initial": {
            "cpu_load": 0.38, "memory_load": 0.43, "network_load": 0.32,
            "trust": 0.43, "attack_persistence": 0.0,
            "attack_active": False, "capacity": 0.82,
        },
    },
    "controlled_cyber_threat": {
        "name": "Controlled Cyber Threat",
        "description": "A persistent attack is active and requires containment.",
        "initial": {
            "cpu_load": 0.43, "memory_load": 0.47, "network_load": 0.38,
            "trust": 0.86, "attack_persistence": 0.62,
            "attack_active": True, "capacity": 0.82,
        },
    },
    "high_resource_pressure": {
        "name": "High Resource Pressure",
        "description": "CPU and memory are near capacity without an active attack.",
        "initial": {
            "cpu_load": 0.90, "memory_load": 0.92, "network_load": 0.34,
            "trust": 0.90, "attack_persistence": 0.0,
            "attack_active": False, "capacity": 0.82,
        },
    },
    "combined_threat_resource_pressure": {
        "name": "Combined Threat + Resource Pressure",
        "description": "An active attack coincides with high CPU and memory use.",
        "initial": {
            "cpu_load": 0.87, "memory_load": 0.89, "network_load": 0.42,
            "trust": 0.84, "attack_persistence": 0.68,
            "attack_active": True, "capacity": 0.82,
        },
    },
}

OPERATIONS = (
    "no_op",
    "isolate",
    "migrate",
    "scale_down",
    "scale_up",
    "restore_trust",
)

_RECOVERY_TARGETS = {
    "cpu_load": 0.35,
    "memory_load": 0.40,
    "network_load": 0.25,
}


def _clip(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _approach(value: float, target: float, fraction: float) -> float:
    return value + (target - value) * fraction


@dataclass
class TwinState:
    scenario: str
    cpu_load: float
    memory_load: float
    network_load: float
    trust: float
    attack_persistence: float
    attack_active: bool
    threat_contained: bool
    isolation_status: bool
    capacity: float
    migration_overhead: float = 0.0
    energy_estimate: float = 0.0
    cost_estimate: float = 0.0


class StatefulSimulation:
    """Owns one persistent Digital Twin state and its action history."""

    def __init__(self) -> None:
        self._previous_security_weight = 1 / 3
        self._history: list[dict[str, Any]] = []
        self._scenario = "normal_operation"
        self._tick = 0
        # False until the first Run tick initializes the selected scenario.
        self._initialized = False
        self._state = self._initial_state("normal_operation")
        self._governance = self._calculate_governance(
            self._state, self._previous_security_weight
        )
        self._previous_security_weight = self._governance["weights"]["wS"]

    @staticmethod
    def _initial_state(scenario: str) -> TwinState:
        initial = deepcopy(SCENARIOS[scenario]["initial"])
        state = TwinState(
            scenario=scenario,
            threat_contained=False,
            isolation_status=False,
            **initial,
        )
        StatefulSimulation._refresh_estimates(state)
        return state

    @staticmethod
    def _refresh_estimates(state: TwinState) -> None:
        state.energy_estimate = _clip(
            0.08
            + 0.28 * state.cpu_load
            + 0.22 * state.memory_load
            + 0.12 * state.network_load
            + 0.18 * state.capacity
            + 0.12 * state.migration_overhead
        )
        state.cost_estimate = _clip(
            0.05
            + 0.42 * state.capacity
            + 0.20 * state.cpu_load
            + 0.18 * state.memory_load
            + 0.10 * state.network_load
            + 0.05 * state.migration_overhead
        )

    @staticmethod
    def _threat_severity(state: TwinState) -> float:
        trust_deficit = max(0.0, (0.80 - state.trust) / 0.80)
        return _clip(
            0.70 * state.attack_persistence
            + 0.20 * trust_deficit
            + 0.10 * state.network_load
        )

    @classmethod
    def _utilities(cls, state: TwinState) -> dict[str, float]:
        tss = cls._threat_severity(state)
        return {
            "U_R": _clip(
                1.0
                - (
                    0.40 * state.cpu_load
                    + 0.30 * state.memory_load
                    + 0.16 * state.network_load
                    + 0.14 * (1.0 - state.capacity)
                )
            ),
            "U_S": _clip(0.60 * state.trust + 0.40 * (1.0 - tss)),
            "U_C": _clip(
                1.0 - 0.50 * state.energy_estimate - 0.50 * state.cost_estimate
            ),
        }

    @classmethod
    def _calculate_governance(
        cls, state: TwinState, previous_security_weight: float
    ) -> dict[str, Any]:
        utilities = cls._utilities(state)
        tss = cls._threat_severity(state)
        candidates = [{"type": action} for action in OPERATIONS]
        candidate_utilities = [[], [], []]

        for candidate in candidates:
            forecast, _ = cls._transition(state, candidate["type"])
            forecast_utilities = cls._utilities(forecast)
            for values, key in zip(candidate_utilities, ("U_R", "U_S", "U_C")):
                values.append(forecast_utilities[key])

        recommended, weights = run_tapna(
            candidates,
            *candidate_utilities,
            tss,
            previous_security_weight,
        )
        cigi = compute_cigi(
            utilities["U_R"], utilities["U_S"], utilities["U_C"], weights
        )

        if state.attack_active:
            status = "THREAT_ACTIVE"
        elif state.threat_contained and state.attack_persistence > 0.03:
            status = "RECOVERING"
        elif (
            utilities["U_R"] < 0.45
            or state.trust < 0.60
            or state.network_load > 0.75
        ):
            status = "DEGRADED"
        else:
            status = "HEALTHY"

        return {
            "utilities": utilities,
            "weights": weights,
            "TSS": tss,
            "CIGI": cigi,
            "status": status,
            "recommended_action": recommended["type"],
        }

    @classmethod
    def _transition(cls, current: TwinState, action: str) -> tuple[TwinState, str]:
        state = deepcopy(current)
        outcome: str

        if action == "no_op":
            outcome = "No control action was taken."
        elif action == "isolate":
            if state.attack_active:
                state.attack_active = False
                state.threat_contained = True
                state.attack_persistence *= 0.35
                state.network_load -= 0.10
                state.trust += 0.04
                outcome = "The active threat was contained; residual persistence is recovering."
            else:
                outcome = "The workload was precautionarily isolated; no active threat was present."
            state.isolation_status = True
            state.capacity -= 0.05
        elif action == "migrate":
            state.cpu_load *= 0.84
            state.memory_load *= 0.90
            state.network_load *= 0.94
            state.capacity += 0.04
            state.migration_overhead += 0.12
            outcome = "Workload was redistributed, reducing resource pressure; attacks remain independent."
        elif action == "scale_down":
            state.capacity -= 0.18
            state.cpu_load *= 0.91
            state.memory_load *= 0.93
            outcome = "Provisioned capacity and resource consumption were reduced."
        elif action == "scale_up":
            state.capacity += 0.20
            state.cpu_load *= 0.88
            state.memory_load *= 0.91
            outcome = "Additional capacity eased utilization and increased estimated energy/cost."
        elif action == "restore_trust":
            if state.attack_active:
                state.trust -= 0.02
                outcome = "Trust restoration was blocked by the active attack; no threat was removed."
            else:
                state.trust += 0.12
                outcome = "Trust improved because no active attack is present."
        else:
            raise ValueError(f"Unsupported operation: {action}")

        if action != "isolate" and state.attack_active:
            state.attack_persistence += 0.07
            state.trust -= 0.035
            state.network_load += 0.025
            state.cpu_load += 0.015
            state.memory_load += 0.010
            outcome += " The uncontained attack persisted and worsened system conditions."
        elif state.threat_contained:
            state.attack_persistence *= 0.78
            state.trust += 0.025
            for metric, target in _RECOVERY_TARGETS.items():
                setattr(state, metric, _approach(getattr(state, metric), target, 0.08))

        state.migration_overhead = max(0.0, state.migration_overhead - 0.01)
        state.cpu_load = _clip(state.cpu_load)
        state.memory_load = _clip(state.memory_load)
        state.network_load = _clip(state.network_load)
        state.trust = _clip(state.trust)
        state.attack_persistence = _clip(state.attack_persistence)
        state.capacity = min(1.0, max(0.45, state.capacity))
        state.migration_overhead = _clip(state.migration_overhead)
        cls._refresh_estimates(state)
        return state, outcome

    @staticmethod
    def _state_dict(state: TwinState) -> dict[str, Any]:
        return asdict(state)

    def _response(self) -> dict[str, Any]:
        return {
            "tick": self._tick,
            **self._state_dict(self._state),
            **deepcopy(self._governance["utilities"]),
            **deepcopy(self._governance),
        }

    def state(self) -> dict[str, Any]:
        return self._response()

    def scenarios(self) -> list[dict[str, Any]]:
        return [
            {
                "id": scenario_id,
                "name": details["name"],
                "description": details["description"],
                "initial_conditions": deepcopy(details["initial"]),
            }
            for scenario_id, details in SCENARIOS.items()
        ]

    def select_scenario(self, scenario_name: str) -> dict[str, Any]:
        scenario_id = scenario_name.strip().lower().replace("-", "_").replace(" ", "_")
        if scenario_id not in SCENARIOS:
            raise KeyError(scenario_name)
        self._scenario = scenario_id
        if not self._initialized:
            # The first Run still performs the full initialization for this
            # scenario; until then only the selection is recorded.
            self._state.scenario = scenario_id
        else:
            # Switching scenarios applies the new environmental conditions to
            # the EXISTING state: metrics are preserved (never reset to their
            # initial values) and only the environment changes. Loads, trust
            # and threat then evolve gradually on the following Run ticks.
            env = SCENARIOS[scenario_id]["initial"]
            state = self._state
            state.scenario = scenario_id
            state.attack_active = bool(env["attack_active"])
            if env["attack_active"]:
                state.attack_persistence = _clip(
                    max(state.attack_persistence, float(env["attack_persistence"]))
                )
                state.threat_contained = False
            elif state.attack_persistence > 0.02:
                state.threat_contained = True
        # Scenario switches keep the accumulated history; only /reset clears it.
        self._previous_security_weight = 1 / 3
        self._governance = self._calculate_governance(
            self._state, self._previous_security_weight
        )
        self._previous_security_weight = self._governance["weights"]["wS"]
        return self._response()

    def reset(self) -> dict[str, Any]:
        self._state = self._initial_state(self._scenario)
        self._tick = 0
        self._history.clear()
        self._previous_security_weight = 1 / 3
        self._initialized = True
        self._governance = self._calculate_governance(
            self._state, self._previous_security_weight
        )
        self._previous_security_weight = self._governance["weights"]["wS"]
        return self._response()

    def run(self) -> dict[str, Any]:
        """Advance the simulation by ONE tick under the selected scenario.

        The first run initializes the state from the selected scenario's
        initial conditions; every later run continues from the previous state
        and applies a bounded, fully deterministic transition (_advance).
        """
        before = self._response()
        if not self._initialized:
            self._state = self._initial_state(self._scenario)
            self._initialized = True
            outcome = (
                f"Initialized {SCENARIOS[self._scenario]['name']}: cpu "
                f"{self._state.cpu_load:.3f}, memory {self._state.memory_load:.3f}, "
                f"network {self._state.network_load:.3f}, trust {self._state.trust:.3f}."
            )
        else:
            self._state, outcome = self._advance(self._state)
        self._tick += 1
        self._governance = self._calculate_governance(
            self._state, self._previous_security_weight
        )
        self._previous_security_weight = self._governance["weights"]["wS"]
        after = self._response()
        self._history.append(
            {
                "step": self._tick,
                "action": self._scenario,
                "before": before,
                "after": after,
                "utilities": deepcopy(self._governance["utilities"]),
                "weights": deepcopy(self._governance["weights"]),
                "TSS": self._governance["TSS"],
                "CIGI": self._governance["CIGI"],
                "outcome": outcome,
            }
        )
        return after

    def _advance(self, current: TwinState) -> tuple[TwinState, str]:
        """One deterministic tick coupling trust, network/threat and resources.

        Three components evolve toward the selected scenario's environmental
        conditions at bounded rates (everything is clipped to [0, 1]):

        * Trust/security: decays toward an attack floor while an attack is
          active, and recovers gradually (never instantly) under safe
          conditions, slowed by any lingering threat or saturated network.
        * Network/threat: rises while an attack persists, declines during
          recovery, and converges on the scenario's network condition.
        * CPU/memory: accumulates under sustained resource pressure and
          relaxes gradually under normal conditions; attacks and network
          congestion add extra load (cross-component coupling).
        """
        env = SCENARIOS[self._scenario]["initial"]
        state = deepcopy(current)
        state.scenario = self._scenario

        attack_env = bool(env["attack_active"])
        target_persistence = float(env["attack_persistence"])
        net_env = float(env["network_load"])
        cpu_env = float(env["cpu_load"])
        mem_env = float(env["memory_load"])
        trust_env = float(env["trust"])

        # --- Trust / security component -------------------------------
        if attack_env:
            state.attack_active = True
            state.threat_contained = False
            state.attack_persistence = _clip(
                state.attack_persistence
                + (target_persistence - state.attack_persistence) * 0.35
                + 0.03
            )
            # An active attack pushes network load up ...
            state.network_load = _clip(
                state.network_load
                + (net_env - state.network_load) * 0.20
                + 0.04 * state.attack_persistence
            )
            # ... and erodes trust: a bounded decay toward an attack floor
            # (never an instant collapse to zero), worsened by congestion.
            state.trust = _clip(
                state.trust
                + (0.15 - state.trust) * (0.04 + 0.06 * state.attack_persistence)
                - 0.015 * max(0.0, state.network_load - 0.70)
            )
        else:
            if state.attack_active or state.attack_persistence > 0.02:
                state.threat_contained = True
            state.attack_active = False
            state.attack_persistence = _clip(state.attack_persistence * 0.85)
            # A lingering threat slows recovery instead of blocking it.
            residual = 1.0 - 0.6 * state.attack_persistence
            net_bonus = 0.006 if net_env >= 0.75 else 0.0
            state.network_load = _clip(
                state.network_load
                + (net_env - state.network_load) * 0.18 * residual
                + net_bonus
            )
            state.trust = _clip(
                state.trust
                + (trust_env - state.trust) * 0.12 * residual
                - 0.03 * state.attack_persistence
                - 0.02 * max(0.0, state.network_load - 0.80)
            )

        # --- CPU / memory resource component --------------------------
        # Pressure scenarios accumulate tick over tick (the environment's
        # heat keeps building even once the target level is reached); safe
        # conditions relax gradually instead.
        pressure_env = cpu_env >= 0.75 or mem_env >= 0.75
        rate = 0.30 if pressure_env else 0.15
        load = 0.02 * state.attack_persistence
        if state.network_load > 0.80:
            load += 0.01 * (state.network_load - 0.80)  # congestion feedback
        if pressure_env:
            load += 0.01  # sustained pressure keeps accumulating
        state.cpu_load = _clip(
            state.cpu_load + (cpu_env - state.cpu_load) * rate + load
        )
        state.memory_load = _clip(
            state.memory_load + (mem_env - state.memory_load) * rate + 0.7 * load
        )

        state.migration_overhead = max(0.0, state.migration_overhead - 0.01)
        StatefulSimulation._refresh_estimates(state)
        return state, self._tick_outcome(current, state)

    @staticmethod
    def _tick_outcome(before: TwinState, after: TwinState) -> str:
        name = SCENARIOS[after.scenario]["name"]
        if after.attack_active:
            threat = "attack active"
        elif after.threat_contained:
            threat = "threat contained"
        else:
            threat = "no active threat"
        trust_dir = "fell" if after.trust < before.trust else "rose"
        net_dir = "rose" if after.network_load > before.network_load else "eased"
        res_dir = (
            "built up"
            if after.cpu_load > before.cpu_load or after.memory_load > before.memory_load
            else "eased"
        )
        return (
            f"{name}: {threat}; trust {trust_dir} to {after.trust:.3f}, "
            f"network {net_dir} to {after.network_load:.3f}, CPU/memory {res_dir} "
            f"(cpu {after.cpu_load:.3f}, memory {after.memory_load:.3f})."
        )

    def operate(self, action_name: str) -> dict[str, Any]:
        action = action_name.strip().lower().replace("-", "_").replace(" ", "_")
        if action not in OPERATIONS:
            raise KeyError(action_name)

        before = self._response()
        self._state, outcome = self._transition(self._state, action)
        self._governance = self._calculate_governance(
            self._state, self._previous_security_weight
        )
        self._previous_security_weight = self._governance["weights"]["wS"]
        after = self._response()
        entry = {
            "step": len(self._history) + 1,
            "action": action,
            "before": before,
            "after": after,
            "utilities": deepcopy(self._governance["utilities"]),
            "weights": deepcopy(self._governance["weights"]),
            "TSS": self._governance["TSS"],
            "CIGI": self._governance["CIGI"],
            "outcome": outcome,
        }
        self._history.append(entry)
        return after

    def history(self) -> list[dict[str, Any]]:
        return deepcopy(self._history)


simulation = StatefulSimulation()
app = FastAPI(
    title="Standalone Stateful Cloud-IoT Governance Demo",
    description="Deterministic in-memory simulation; not connected to cloud infrastructure.",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/state")
def get_state() -> dict[str, Any]:
    return simulation.state()


@app.get("/scenarios")
def get_scenarios() -> list[dict[str, Any]]:
    return simulation.scenarios()


@app.post("/scenario/{scenario_name}")
def select_scenario(scenario_name: str) -> dict[str, Any]:
    try:
        return simulation.select_scenario(scenario_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown scenario") from exc


@app.post("/run")
def run_tick() -> dict[str, Any]:
    return simulation.run()


@app.post("/operate/{action_name}")
def operate(action_name: str) -> dict[str, Any]:
    try:
        return simulation.operate(action_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown operation") from exc


@app.post("/reset")
def reset() -> dict[str, Any]:
    return simulation.reset()


@app.get("/history")
def get_history() -> dict[str, Any]:
    return {"history": simulation.history()}
