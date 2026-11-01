import math

from digital_twin.action_generator import generate_candidate_actions
from digital_twin.telemetry_generator import generate_telemetry
from digital_twin.twin import DigitalTwin
from agents.sustainability_agent import score_sustainability


def _make_state(cpu=0.35, memory=0.40, network=0.25, trust=0.95):
    twin = DigitalTwin(100)
    for device in twin.devices:
        device.cpu_load = cpu
        device.memory_load = memory
        device.network_load = network
        device.trust_score = trust
    return generate_telemetry(twin), twin


def test_sustainability_utility_is_finite_bounded_and_varies_by_action():
    telemetry, twin = _make_state()
    actions = generate_candidate_actions(twin)

    values = [score_sustainability(action, telemetry) for action in actions]

    assert values
    assert all(math.isfinite(v) for v in values)
    assert all(0.0 <= v <= 1.0 for v in values)
    assert len(set(round(v, 10) for v in values)) > 1
