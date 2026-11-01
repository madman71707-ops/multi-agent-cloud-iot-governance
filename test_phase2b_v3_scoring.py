"""Phase 2b v3 scoring unit tests.

Covers the spec's required additions for 2b-2:
  * service_loss(a) semantics: no_op=0, high-load isolate > low-load isolate,
    migrate and scale_down non-negative.
  * ur_mode="v3": U_R_v3(a) = clip01(U_R_legacy(a) - kappa * service_loss(a)),
    so no_op is unchanged and every non-no_op action has U_R_v3 <= U_R_legacy.
"""
import numpy as np
import pytest

from digital_twin.twin_v2 import deep_update, get_twin_v2_config, get_closed_loop_config
from agents.resource_agent import score_resource
from sim.scoring_v2 import _clip01
from sim.scoring_v3 import service_loss, ur_v3, ur_v3_list
from digital_twin.twin_v3 import DigitalTwinV3, get_closed_loop_v3_config


def _make_v3_twin():
    """A 3-device DigitalTwinV3 with known loads."""
    cfg = deep_update(get_twin_v2_config(), {"baseline_trust_range": [0.8, 0.9], "baseline_network_range": [0.1, 0.2]})
    v3cl = get_closed_loop_v3_config()
    twin = DigitalTwinV3(3, config=cfg, cl_config=v3cl)
    twin.devices[0].cpu_load = 0.5
    twin.devices[0].network_load = 0.5
    twin.devices[1].cpu_load = 0.1
    twin.devices[1].network_load = 0.1
    twin.devices[2].cpu_load = 0.1
    twin.devices[2].network_load = 0.1
    return twin, v3cl


def _v3_config():
    return deep_update(get_closed_loop_config(), {
        "actions": {
            "migrate": {"migrate_service_loss_fraction": 0.2},
            "scale_down": {"capacity_reduction": 0.1},
            "isolate": {"isolation_duration": 10, "energy_saving": 0.0},
            "no_op": {},
        },
        "isolation_cap": {"max_isolated_fraction": 0.15},
    })


def test_no_op_has_service_loss_zero():
    twin, _ = _make_v3_twin()
    v3 = _v3_config()
    loss = service_loss({"type": "no_op", "target_device_id": twin.devices[0].id}, twin, v3)
    assert loss == 0.0


def test_isolate_high_load_greater_than_low_load():
    twin, _ = _make_v3_twin()
    v3 = _v3_config()
    high = {"type": "isolate", "target_device_id": twin.devices[0].id}
    low = {"type": "isolate", "target_device_id": twin.devices[1].id}
    loss_high = service_loss(high, twin, v3)
    loss_low = service_loss(low, twin, v3)
    assert loss_high > loss_low
    assert loss_low >= 0.0 and loss_high >= 0.0


def test_isolate_world_total_check():
    twin, _ = _make_v3_twin()
    v3 = _v3_config()
    loss = service_loss({"type": "isolate", "target_device_id": twin.devices[0].id}, twin, v3)
    # device 0 load = 1.0; total = 1.0 + 0.2 + 0.2 = 1.4; loss = 1.0/1.4
    assert np.isclose(loss, 1.0 / 1.4)


def test_migrate_service_loss_fraction_applied():
    twin, _ = _make_v3_twin()
    v3 = _v3_config()
    mig = {"type": "migrate", "target_device_id": twin.devices[0].id}
    loss = service_loss(mig, twin, v3)
    assert np.isclose(loss, 0.2 * (1.0 / 1.4))


def test_scale_down_service_loss_applied():
    twin, _ = _make_v3_twin()
    v3 = _v3_config()
    sd = {"type": "scale_down", "target_device_id": twin.devices[0].id}
    loss = service_loss(sd, twin, v3)
    assert np.isclose(loss, 0.1 * (1.0 / 1.4))


def test_already_isolated_device_has_zero_additional_loss():
    twin, _ = _make_v3_twin()
    v3 = _v3_config()
    twin.devices[0].isolated = True
    loss = service_loss({"type": "isolate", "target_device_id": twin.devices[0].id}, twin, v3)
    assert loss == 0.0


def test_ur_v3_no_op_unchanged():
    twin, _ = _make_v3_twin()
    v3 = _v3_config()
    kappa = 2.0
    noop = {"type": "no_op", "target_device_id": twin.devices[0].id}
    ur_legacy = score_resource(noop, {
        "avg_cpu_load": float(np.mean([d.cpu_load for d in twin.devices])),
        "avg_memory_load": float(np.mean([d.memory_load for d in twin.devices])),
        "avg_network_load": float(np.mean([d.network_load for d in twin.devices])),
    })
    ur_v3_val = ur_v3(noop, twin, v3, kappa=kappa)
    assert ur_v3_val == ur_legacy
    assert service_loss(noop, twin, v3) == 0.0


def test_ur_v3_every_action_le_less_or_equal_legacy():
    twin, _ = _make_v3_twin()
    v3 = _v3_config()
    kappa = 2.0
    actions = [
        {"type": "isolate", "target_device_id": twin.devices[0].id},
        {"type": "migrate", "target_device_id": twin.devices[0].id},
        {"type": "scale_down", "target_device_id": twin.devices[0].id},
    ]
    fleet_state = {
        "avg_cpu_load": float(np.mean([d.cpu_load for d in twin.devices])),
        "avg_memory_load": float(np.mean([d.memory_load for d in twin.devices])),
        "avg_network_load": float(np.mean([d.network_load for d in twin.devices])),
    }
    for a in actions:
        legacy = score_resource(a, fleet_state)
        v3val = ur_v3(a, twin, v3, kappa=kappa)
        assert v3val <= legacy + 1e-9
        assert 0.0 <= v3val <= 1.0


def test_ur_v3_list():
    twin, _ = _make_v3_twin()
    v3 = _v3_config()
    kappa = 2.0
    actions = [
        {"type": "no_op", "target_device_id": twin.devices[0].id},
        {"type": "isolate", "target_device_id": twin.devices[0].id},
    ]
    fleet_state = {
        "avg_cpu_load": float(np.mean([d.cpu_load for d in twin.devices])),
        "avg_memory_load": float(np.mean([d.memory_load for d in twin.devices])),
        "avg_network_load": float(np.mean([d.network_load for d in twin.devices])),
    }
    legacy = [score_resource(a, fleet_state) for a in actions]
    v3vals = ur_v3_list(actions, twin, v3, kappa=kappa)
    assert len(v3vals) == len(actions)
    assert v3vals[0] == legacy[0]  # no_op unchanged
    assert v3vals[1] <= legacy[1]  # isolate reduced"
