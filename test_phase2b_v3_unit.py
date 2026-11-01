"""Phase 2b v3 environment tests (spec 2b-1..2b-3, 2b-8).

Part 1 - config contract (2b-1): the v3 config is a NEW file, keeps the
v2 default untouched, and carries the finite-cost isolation parameters.

Part 2 - twin behaviour (2b-1, 2b-3, 2b-8):
  * concurrency cap is enforced (isolate candidates dropped at the cap,
    apply_action refuses past the cap);
  * re-admission after isolation_duration ticks;
  * service-loss accounting (isolate/migrate/scale_down/no_op);
  * the new v3 metrics are non-negative and bounded;
  * NoOp never isolates (2b-4/2b-8).
"""
import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml
from digital_twin.twin_v2 import deep_update, get_closed_loop_config, get_twin_v2_config


def _v3_config():
    base = get_closed_loop_config()
    v3 = deep_update(base, yaml.safe_load(open("config/closed_loop_v3.yaml", encoding="utf-8")))
    return v3


def test_v3_config_isolated_from_v2():
    v2 = get_closed_loop_config()  # v2 is the default, reads closed_loop.yaml
    v3 = _v3_config()
    # v3 should have the new isolation_cap block
    assert "isolation_cap" in v3
    # v2 should NOT have it (legacy file untouched)
    assert "isolation_cap" not in v2


def test_v3_isolate_has_no_energy_saving():
    v3 = _v3_config()
    assert v3["actions"]["isolate"]["energy_saving"] == 0.0


def test_v3_isolate_has_isolation_duration():
    v3 = _v3_config()
    assert v3["actions"]["isolate"]["isolation_duration"] == 10


def test_v3_migrate_carries_service_loss_fraction():
    v3 = _v3_config()
    assert v3["actions"]["migrate"]["migrate_service_loss_fraction"] == 0.2


def test_v3_scale_down_has_capacity_reduction_service_loss_implication():
    v3 = _v3_config()
    # scale_down.capacity_reduction > 0 means the removed capacity is now lost
    # service (this is documented in the YAML). We check it is still present.
    assert v3["actions"]["scale_down"]["capacity_reduction"] == 0.1


def test_v3_concurrency_cap_exists():
    v3 = _v3_config()
    assert v3["isolation_cap"]["max_isolated_fraction"] == 0.15


def test_v3_no_op_is_still_no_op():
    v3 = _v3_config()
    assert v3["actions"]["no_op"] == {}


def test_v2_config_untouched():
    """closed_loop.yaml (v2 default) must not gain v3 keys."""
    v2 = get_closed_loop_config()
    assert "isolation_cap" not in v2
    assert "isolation_duration" not in v2["actions"]["isolate"]
    assert v2["actions"]["isolate"]["energy_saving"] == 0.10


# ---------------------------------------------------------------------------
# Twin behaviour (2b-1 / 2b-3 / 2b-8)
# ---------------------------------------------------------------------------
from digital_twin.twin_v3 import DigitalTwinV3, get_closed_loop_v3_config


def _twin(n=100, cap=0.15, duration=10):
    cfg = get_twin_v2_config()
    cl = deep_update(get_closed_loop_v3_config(), {
        "isolation_cap": {"max_isolated_fraction": cap},
        "actions": {"isolate": {"isolation_duration": duration}},
    })
    np.random.seed(1000)
    import random
    random.seed(1000)
    return DigitalTwinV3(n, config=cfg, cl_config=cl), cl


def _iso_action(twin, index=0):
    return {"id": "a0", "type": "isolate",
            "target_device_id": twin.devices[index].id}


def test_cap_enforced_in_candidate_generation():
    twin, cl = _twin(n=100, cap=0.15)
    assert twin.max_isolated_count == 15
    # Fill up to the cap.
    for i in range(15):
        twin.apply_action(_iso_action(twin, i), cl)
    assert twin.isolated_count == 15
    assert twin.isolation_cap_reached
    # No isolate candidate may be generated at the cap.
    actions = twin.generate_candidates(0)
    assert actions, "candidate set must never be empty"
    assert not any(a["type"] == "isolate" for a in actions)
    # And apply_action is a hard guard: isolating past the cap does nothing.
    twin.apply_action(_iso_action(twin, 50), cl)
    assert twin.isolated_count == 15


def test_cap_not_binding_below_cap():
    twin, cl = _twin(n=100, cap=0.15)
    for i in range(5):
        twin.apply_action(_iso_action(twin, i), cl)
    assert twin.isolated_count == 5
    assert not twin.isolation_cap_reached
    assert any(a["type"] == "isolate" for a in twin.generate_candidates(0))


def test_re_admission_after_isolation_duration():
    duration = 10
    twin, cl = _twin(n=100, cap=0.50, duration=duration)
    twin.advance(0.0, closed_loop_cfg=cl)   # tick 0
    tick_isolated = twin._v3_tick
    twin.apply_action(_iso_action(twin, 3), cl)
    assert twin.devices[3].isolated
    # Isolated for exactly `duration` telemetry ticks: t+1 .. t+duration.
    observed = []
    for _ in range(duration + 3):
        twin.advance(0.0, closed_loop_cfg=cl)
        observed.append(twin.devices[3].isolated)
    assert all(observed[:duration]), observed
    assert not any(observed[duration:]), observed
    assert not twin.devices[3].isolated
    # Service loss cleared on re-admission.
    assert twin.devices[3]._v3_isolation_loss == 0.0
    assert twin.offline_workload_share() < 0.02


def test_isolate_clears_compromise_and_readmission_keeps_it_cleared():
    twin, cl = _twin(n=100, cap=0.50, duration=3)
    twin.advance(0.0, closed_loop_cfg=cl)
    twin.devices[5].compromised = True
    twin.apply_action(_iso_action(twin, 5), cl)
    assert not twin.devices[5].compromised
    for _ in range(5):
        twin.advance(0.0, closed_loop_cfg=cl)
    # Re-admitted; the flag isolation cleared is NOT restored by re-admission.
    assert not twin.devices[5].isolated
    assert not twin.devices[5].compromised


def test_service_loss_accounting():
    twin, cl = _twin(n=100, cap=0.50, duration=10)
    # no_op: zero loss, nothing changes.
    before = twin.offline_workload_share()
    twin.apply_action({"id": "n", "type": "no_op",
                       "target_device_id": twin.devices[0].id}, cl)
    assert twin.offline_workload_share() == before == 0.0

    # isolate: the target's workload share goes offline.
    dev = twin.devices[0]
    total = sum(d.cpu_load + d.network_load for d in twin.devices)
    share = (dev.cpu_load + dev.network_load) / total
    twin.apply_action(_iso_action(twin, 0), cl)
    assert np.isclose(twin.devices[0]._v3_isolation_loss, share)
    assert np.isclose(twin.offline_workload_share(), share)
    assert np.isclose(twin.get_telemetry()["service_availability"], 1 - share)

    # scale_down on a non-isolated device: capacity_reduction * share.
    dev1 = twin.devices[1]
    total1 = sum(d.cpu_load + d.network_load for d in twin.devices)
    want = 0.1 * (dev1.cpu_load + dev1.network_load) / total1
    twin.apply_action({"id": "s", "type": "scale_down",
                       "target_device_id": dev1.id}, cl)
    assert np.isclose(dev1._v3_scale_loss, want, atol=1e-9)
    assert dev1._v3_scale_loss > 0

    # migrate: exactly one following tick of loss, then it expires.
    twin2, cl2 = _twin(n=100, cap=0.50)
    twin2.advance(0.0, closed_loop_cfg=cl2)      # tick 0
    d2 = twin2.devices[2]
    twin2.apply_action({"id": "m", "type": "migrate",
                        "target_device_id": d2.id}, cl2)
    assert d2._v3_migrate_loss > 0
    twin2.advance(0.0, closed_loop_cfg=cl2)      # tick 1: loss still counted
    assert twin2.offline_workload_share() > 0
    twin2.advance(0.0, closed_loop_cfg=cl2)      # tick 2: expired
    assert d2._v3_migrate_loss == 0.0


def test_new_metrics_non_negative_and_bounded():
    from sim.attack_patterns_v3 import run_v3_closed_loop
    rows, outcome = run_v3_closed_loop(
        "RandomAction", "Gradual attack", 1000, kappa=2.0, num_ticks=30)
    for key in ("service_availability_mean", "service_availability_min",
                "isolated_device_ticks_total", "false_isolation_ticks",
                "max_isolated_devices", "net_disruption_cost"):
        assert key in outcome, key
        assert np.isfinite(outcome[key]), (key, outcome[key])
        assert outcome[key] >= 0, (key, outcome[key])
    assert outcome["service_availability_mean"] <= 1.0
    assert outcome["service_availability_min"] <= 1.0
    assert outcome["service_availability_min"] <= outcome["service_availability_mean"]
    # The cap is a hard bound on max_isolated_devices (100 devices, cap 0.15).
    assert outcome["max_isolated_devices"] <= 15
    # Per-tick rows: availability is 1 - offline share, offline in [0, 1].
    for r in rows:
        assert 0.0 <= r["offline_workload_share"] <= 1.0
        assert np.isclose(r["service_availability"], 1 - r["offline_workload_share"])
        assert r["false_isolated_device_ticks"] >= 0


def test_noop_never_isolates():
    from sim.attack_patterns_v3 import run_v3_closed_loop
    for scenario in ("Gradual attack", "Sustained attack"):
        rows, outcome = run_v3_closed_loop(
            "NoOp", scenario, 1000, kappa=2.0, num_ticks=50)
        types = {r["chosen_action_type"] for r in rows}
        assert types == {"no_op"}, types
        assert outcome["isolate_share"] == 0.0
        assert outcome["max_isolated_devices"] == 0
        assert outcome["isolated_device_ticks_total"] == 0
        assert outcome["service_availability_mean"] == 1.0
        assert outcome["net_disruption_cost"] == 0.0
