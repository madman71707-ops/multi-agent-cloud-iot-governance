"""Phase 2b v3 resource utility (ur_mode="v3").

New ``ur_mode="v3"`` resource utility that sees the COST of isolation:

    service_loss(a) = fraction of fleet workload taken offline by action a

      - isolate: the target device's current workload share
        (cpu+network load over the fleet total) while it is quarantined;
        this is the device's REAL share at the moment of isolation (from the
        twin state), not a fixed constant.
      - migrate: a fraction (default 0.2) of the target's workload share,
        while the load is being moved.
      - scale_down: the removed capacity fraction (capacity_reduction) of the
        target's workload share while scaled down.
      - no_op: 0

    U_R_v3(a) = clip01( U_R_legacy(a) - kappa * service_loss(a) )

      - U_R_legacy(a) is the SAME per-action resource utility the legacy mode
        already computes (from the predicted post-action telemetry), so the
        legacy values are preserved and the v3 value is the legacy value minus
        a configurable penalty for the service lost by the action.
      - kappa is configurable (default 2.0), tuned in the calibration sandbox
        using TAPNA-agnostic criteria.

Properties (unit-tested below):

  * no_op has service_loss = 0  (so U_R_v3 = U_R_legacy, unchanged)
  * isolating a high-load device has a LARGER service_loss than isolating a
    low-load device (because the workload share is device-specific)
  * service_loss is in [0,1] and kappa>0 keeps U_R_v3 in [0,1] when U_R_legacy
    is in [0,1]
  * legacy ur_mode is unchanged (delegates to score_resource as before)

This module does NOT import or modify any v2 module; it reuses the same legacy
score_resource helper that the v2 scorer uses for U_R_legacy.
"""
import numpy as np

from agents.resource_agent import score_resource
from sim.scoring_v2 import _clip01


def service_loss(action, twin_state, v3_config):
    """Fraction of fleet workload taken offline by action a in the v3 model.

    Computed from the twin state (per-device loads) so the loss is the REAL
    device's share at the moment of the action, not a fixed constant.

    isolate: target device's workload share = (cpu + network load of target)
             over fleet total (cpu + network load summed across all devices),
             but ONLY if the target is NOT already isolated (an already-isolated
             device has no additional service loss).
    migrate: migrate_service_loss_fraction (default 0.2 in the YAML) of the
             target's workload share (the load being moved causes a temporary
             offline fraction while migrating).
    scale_down: capacity_reduction (default 0.1 in the YAML) of the target's
                workload share (the removed capacity is offline while scaled down).
    no_op: 0
    """
    from digital_twin.twin_v3 import DigitalTwinV3

    action_type = action["type"]
    target_id = action.get("target_device_id")
    fleet = twin_state

    if not isinstance(fleet, DigitalTwinV3):
        raise TypeError(
            "service_loss requires a DigitalTwinV3 twin_state, not a telemetry dict"
        )

    devices = fleet.devices
    if not devices:
        return 0.0

    total_load = sum(d.cpu_load + d.network_load for d in devices)
    if total_load <= 0:
        return 0.0

    target = fleet.get_device(target_id) if target_id else None

    if action_type == "no_op":
        return 0.0

    if action_type == "isolate":
        if target is None:
            return 0.0
        # already-isolated device: no additional loss (already offline)
        if getattr(target, "isolated", False):
            return 0.0
        target_load = target.cpu_load + target.network_load
        return target_load / total_load

    if action_type == "migrate":
        if target is None:
            return 0.0
        frac = v3_config["actions"]["migrate"].get("migrate_service_loss_fraction", 0.2)
        target_load = target.cpu_load + target.network_load
        return frac * (target_load / total_load)

    if action_type == "scale_down":
        if target is None:
            return 0.0
        cap = v3_config["actions"]["scale_down"].get("capacity_reduction", 0.1)
        target_load = target.cpu_load + target.network_load
        return cap * (target_load / total_load)

    return 0.0


def ur_v3(action, twin_state, v3_config, kappa=2.0):
    """U_R_v3(a) = clip01( U_R_legacy(a) - kappa * service_loss(a) )."""
    from digital_twin.twin_v3 import DigitalTwinV3
    legacy = score_resource(action, twin_state if not isinstance(twin_state, DigitalTwinV3) else {
        "avg_cpu_load": float(sum(d.cpu_load for d in twin_state.devices) / len(twin_state.devices)),
        "avg_memory_load": float(sum(d.memory_load for d in twin_state.devices) / len(twin_state.devices)),
        "avg_network_load": float(sum(d.network_load for d in twin_state.devices) / len(twin_state.devices)),
    })
    loss = service_loss(action, twin_state, v3_config)
    return _clip01(legacy - kappa * loss)


def ur_v3_list(actions, twin_state, v3_config, kappa=2.0, legacy=None):
    """U_R_v3 for every candidate.

    ``legacy`` optionally supplies the already-computed U_R_legacy values
    (e.g. the batched U_R from ``score_actions_v2``). They are numerically
    identical to ``score_resource`` on the same fleet state - see the
    module docstring of sim/scoring_v2.py - so passing them in just avoids
    re-running the resource predictor once per candidate (it dominated the
    per-tick cost of the v3 grid: ~70% of a run's wall time).
    """
    if legacy is None:
        legacy = [ur_v3(a, twin_state, v3_config, kappa=kappa) for a in actions]
        return legacy
    from digital_twin.twin_v3 import DigitalTwinV3
    if isinstance(twin_state, DigitalTwinV3):
        fleet = {
            "avg_cpu_load": float(sum(d.cpu_load for d in twin_state.devices) / len(twin_state.devices)),
            "avg_memory_load": float(sum(d.memory_load for d in twin_state.devices) / len(twin_state.devices)),
            "avg_network_load": float(sum(d.network_load for d in twin_state.devices) / len(twin_state.devices)),
        }
    else:
        fleet = twin_state
    return [
        _clip01(float(base) - kappa * service_loss(a, twin_state, v3_config))
        for a, base in zip(actions, legacy)
    ]


# ---------------------------------------------------------------------------
# Unit tests (run with: python -m pytest test_phase2b_v3_unit.py -q   or
#                     python -c "import test_phase2b_v3_unit; test_phase2b_v3_unit.test_...()"
# These are ALSO importable as module-level doctests-style checks below.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import test_phase2b_v3_unit as t3
    t3.test_v3_config_isolated_from_v2()
    t3.test_v3_isolate_has_no_energy_saving()
    t3.test_v3_isolate_has_isolation_duration()
    t3.test_v3_migrate_carries_service_loss_fraction()
    t3.test_v3_scale_down_has_capacity_reduction_service_loss_implication()
    t3.test_v3_concurrency_cap_exists()
    t3.test_v3_no_op_is_still_no_op()
    print("config smoke tests OK")
