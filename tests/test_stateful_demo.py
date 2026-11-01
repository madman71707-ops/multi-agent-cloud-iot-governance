"""Focused behavior tests for the standalone stateful simulation."""

from copy import deepcopy

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend import stateful_demo as demo


@pytest.fixture
def simulation():
    return demo.StatefulSimulation()


def test_scenarios_initialize_with_distinct_meaningful_conditions(simulation):
    snapshots = {}
    for scenario_id, description in demo.SCENARIOS.items():
        state = simulation.select_scenario(scenario_id)
        snapshots[scenario_id] = state
        assert state["scenario"] == scenario_id
        assert state["cpu_load"] == description["initial"]["cpu_load"]
        assert state["attack_active"] == description["initial"]["attack_active"]
        assert state["capacity"] == description["initial"]["capacity"]

    fingerprints = {
        tuple(
            state[key]
            for key in (
                "cpu_load", "memory_load", "network_load", "trust",
                "attack_persistence", "attack_active",
            )
        )
        for state in snapshots.values()
    }
    assert len(fingerprints) == len(demo.SCENARIOS)


def test_each_operation_has_a_distinct_state_transition(simulation):
    resulting_states = {}
    for action in demo.OPERATIONS:
        simulation.reset()
        if action == "no_op":
            simulation.select_scenario("controlled_cyber_threat")
        before = simulation.state()
        after = simulation.operate(action)
        resulting_states[action] = after
        assert after != before

    fingerprints = {
        tuple(
            state[key]
            for key in (
                "cpu_load", "memory_load", "network_load", "trust",
                "attack_persistence", "attack_active", "isolation_status",
                "capacity", "migration_overhead",
            )
        )
        for state in resulting_states.values()
    }
    assert len(fingerprints) == len(demo.OPERATIONS)


def test_repeated_uncontained_attacks_worsen_conditions(simulation):
    simulation.select_scenario("controlled_cyber_threat")
    initial = simulation.state()
    once = simulation.operate("no_op")
    twice = simulation.operate("no_op")
    assert twice["attack_persistence"] > once["attack_persistence"] > initial["attack_persistence"]
    assert twice["trust"] < once["trust"] < initial["trust"]
    assert twice["network_load"] > once["network_load"] > initial["network_load"]


def test_isolation_contains_attack_and_allows_gradual_recovery(simulation):
    simulation.select_scenario("controlled_cyber_threat")
    attacked = simulation.operate("no_op")
    isolated = simulation.operate("isolate")
    assert isolated["attack_active"] is False
    assert isolated["threat_contained"] is True
    assert isolated["isolation_status"] is True
    assert isolated["attack_persistence"] < attacked["attack_persistence"]

    trust_before_recovery = isolated["trust"]
    persistence_before_recovery = isolated["attack_persistence"]
    recovered = simulation.operate("no_op")
    assert recovered["trust"] > trust_before_recovery
    assert recovered["attack_persistence"] < persistence_before_recovery


def test_operations_apply_to_current_state_and_do_not_reset(simulation):
    simulation.select_scenario("high_resource_pressure")
    scaled = simulation.operate("scale_down")
    migrated = simulation.operate("migrate")
    assert migrated["scenario"] == "high_resource_pressure"
    assert migrated["capacity"] == scaled["capacity"] + 0.04
    assert migrated["cpu_load"] < scaled["cpu_load"]
    assert len(simulation.history()) == 2


def test_action_order_can_change_final_state(simulation):
    simulation.select_scenario("combined_threat_resource_pressure")
    for action in ("migrate", "isolate", "scale_down"):
        first_order = simulation.operate(action)

    simulation.select_scenario("combined_threat_resource_pressure")
    for action in ("scale_down", "isolate", "migrate"):
        second_order = simulation.operate(action)

    assert first_order != second_order
    assert first_order["migration_overhead"] != pytest.approx(
        second_order["migration_overhead"]
    )


def test_repeated_operations_preserve_state_and_append_history(simulation):
    simulation.select_scenario("normal_operation")
    first = simulation.operate("migrate")
    second = simulation.operate("migrate")
    assert second["migration_overhead"] > 0
    assert second["cpu_load"] < first["cpu_load"]
    assert len(simulation.history()) == 2


def test_metrics_weights_and_governance_scores_are_bounded(simulation):
    for scenario in demo.SCENARIOS:
        simulation.select_scenario(scenario)
        for action in ("no_op", "scale_up", "scale_down", "isolate", "restore_trust"):
            state = simulation.operate(action)
            for key in (
                "cpu_load", "memory_load", "network_load", "trust",
                "attack_persistence", "capacity", "migration_overhead",
                "energy_estimate", "cost_estimate", "U_R", "U_S", "U_C",
                "TSS", "CIGI",
            ):
                assert 0.0 <= state[key] <= 1.0, (scenario, action, key)
            assert sum(state["weights"].values()) == pytest.approx(1.0)
            assert all(0.0 <= weight <= 1.0 for weight in state["weights"].values())


def test_cigi_matches_weighted_utility_formula(simulation):
    state = simulation.operate("no_op")
    expected = sum(
        state["weights"][weight] * state["utilities"][utility]
        for weight, utility in (("wR", "U_R"), ("wS", "U_S"), ("wC", "U_C"))
    )
    assert state["CIGI"] == pytest.approx(expected)


def test_reset_restores_initial_normal_state_and_clears_history(simulation):
    simulation.select_scenario("combined_threat_resource_pressure")
    simulation.operate("no_op")
    reset_state = simulation.reset()
    assert reset_state["scenario"] == "normal_operation"
    assert reset_state["cpu_load"] == 0.35
    assert reset_state["memory_load"] == 0.40
    assert reset_state["network_load"] == 0.25
    assert reset_state["trust"] == 0.95
    assert reset_state["attack_persistence"] == 0
    assert reset_state["attack_active"] is False
    assert reset_state["capacity"] == 0.82
    assert simulation.history() == []


def test_invalid_scenario_and_operation_return_http_404():
    with pytest.raises(HTTPException) as scenario_error:
        demo.select_scenario("not-a-scenario")
    assert scenario_error.value.status_code == 404

    with pytest.raises(HTTPException) as action_error:
        demo.operate("not-an-operation")
    assert action_error.value.status_code == 404


def test_http_endpoints_persist_state_and_reject_unknown_names():
    client = TestClient(demo.app)
    client.post("/reset")

    state_response = client.get("/state")
    assert state_response.status_code == 200
    assert state_response.json()["scenario"] == "normal_operation"
    assert len(client.get("/scenarios").json()) == 6

    selected = client.post("/scenario/high_network_load")
    assert selected.status_code == 200
    changed = client.post("/operate/scale_up")
    assert changed.status_code == 200
    assert changed.json()["capacity"] > selected.json()["capacity"]
    assert len(client.get("/history").json()["history"]) == 1
    assert client.post("/scenario/not_a_scenario").status_code == 404
    assert client.post("/operate/not_an_operation").status_code == 404


def test_same_start_and_action_sequence_are_reproducible():
    actions = ("no_op", "migrate", "isolate", "scale_down", "restore_trust")
    first = demo.StatefulSimulation()
    second = demo.StatefulSimulation()
    first.select_scenario("combined_threat_resource_pressure")
    second.select_scenario("combined_threat_resource_pressure")
    for action in actions:
        assert first.operate(action) == second.operate(action)
    assert first.history() == second.history()


def test_history_records_before_after_scores_and_action(simulation):
    simulation.select_scenario("controlled_cyber_threat")
    before = simulation.state()
    after = simulation.operate("isolate")
    entry = simulation.history()[0]
    assert entry["action"] == "isolate"
    assert entry["before"] == before
    assert entry["after"] == after
    assert set(entry["utilities"]) == {"U_R", "U_S", "U_C"}
    assert set(entry["weights"]) == {"wR", "wS", "wC"}
    assert 0 <= entry["TSS"] <= 1
    assert 0 <= entry["CIGI"] <= 1
    assert entry["outcome"]


def test_test_setup_does_not_mutate_scenario_definitions():
    original = deepcopy(demo.SCENARIOS["normal_operation"]["initial"])
    demo.StatefulSimulation().operate("scale_down")
    assert demo.SCENARIOS["normal_operation"]["initial"] == original
