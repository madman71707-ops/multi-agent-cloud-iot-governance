"""Digital Twin v3 - finite-cost isolation model (Phase 2b).

Built on top of the v2 twin (digital_twin/twin_v2.py). The v2 twin and its
config (config/twin_v2.yaml) are untouched. v3 REUSES the v2 config for the
per-tick dynamics (walks, recovery, attack application, propagation) and the
v2 candidate generator, but REPLACES the closed-loop action effects with the
finite-cost model defined in config/closed_loop_v3.yaml.

What v3 adds on top of v2 (DigitalTwinV3 extends DigitalTwinV2):

* No energy saving on isolation (a quarantined device still draws power).
* Temporary quarantine: an isolated device is RE-ADMITTED after
  ``isolation_duration`` ticks (config/closed_loop_v3.yaml, default 10). The
  twin keeps a tick counter (``_v3_tick``); ``apply_action`` stamps the
  isolation with ``reentry = current_tick + isolation_duration`` and
  ``advance`` re-admits the device (before that tick's dynamics) once
  ``current_tick > reentry``. A device isolated at the end of tick t is
  therefore reported isolated in the telemetry of ticks t+1 .. t+duration
  and free from tick t+duration+1 on - exactly ``isolation_duration``
  isolated ticks. The compromised flag cleared by isolation is NOT restored
  on re-admission (it can only become True again if the campaign still hits
  the device).
* Concurrency cap: at most ``max_isolated_fraction`` of the fleet may be
  isolated at once. It is enforced twice: ``generate_candidates`` drops ALL
  isolate candidates when the cap is reached (so no strategy can pick one),
  and ``apply_action`` refuses to isolate past the cap as a hard guard.
* Per-device service-loss accounting, exposed per tick by
  ``get_telemetry``:
      - isolate: the target's workload share (cpu+network over the fleet
        total) SNAPSHOT at isolation time, counted every tick while it is
        isolated, cleared on re-admission.
      - migrate: ``migrate_service_loss_fraction`` (default 0.2) of the
        target's workload share, counted for exactly one following tick
        (the move itself: the load is redistributed, not lost).
      - scale_down: ``capacity_reduction`` (default 0.1) of the target's
        workload share, counted from the action onwards (the device stays
        scaled down; nothing in the model un-scales it).
      - no_op: 0.
    ``offline_workload_share`` = sum of the above over the fleet, clipped to
    [0, 1]; ``service_availability = 1 - offline_workload_share``.
* ``false_isolated_device_ticks``: number of devices that are isolated THIS
  tick but were NOT compromised at the moment they were isolated (false
  positives of the isolation decision), i.e. device-ticks, summed over the
  run by the outcome metrics.

These hooks are consumed by sim/attack_patterns_v3.py (outcome metrics) and
tested in test_phase2b_v3_unit.py (2b-8).
"""
from pathlib import Path
import numpy as np

from digital_twin.twin_v2 import (
    DigitalTwinV2,
    generate_candidate_actions_v2,
    get_closed_loop_config,
    get_twin_v2_config,
    deep_update,
    build_v2_schedule,
)


def load_config(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_v3_config(path=None):
    p = path or Path(__file__).resolve().parents[1] / "config" / "closed_loop_v3.yaml"
    return load_config(p)


def get_closed_loop_v3_config(path=None):
    return load_config(path or Path(__file__).resolve().parents[1] / "config" / "closed_loop_v3.yaml")


class DeviceV3:
    """A v2 DeviceV2 with v3 service-loss / quarantine bookkeeping."""

    __slots__ = (
        "id", "cpu_load", "memory_load", "network_load", "trust_score",
        "cost_per_hour", "energy_kwh", "baseline_trust", "baseline_network",
        "susceptibility", "targeted", "compromised", "isolated", "leak_factor",
        "_v3_isolation_loss",
        "_v3_scale_loss",
        "_v3_migrate_loss",
        "_v3_migrate_set_tick",
        "_v3_isolation_reentry_tick",
        "_v3_was_compromised_at_isolation",
    )

    def __init__(self, from_v2_device):
        self.id = from_v2_device.id
        self.cpu_load = from_v2_device.cpu_load
        self.memory_load = from_v2_device.memory_load
        self.network_load = from_v2_device.network_load
        self.trust_score = from_v2_device.trust_score
        self.cost_per_hour = from_v2_device.cost_per_hour
        self.energy_kwh = from_v2_device.energy_kwh
        self.baseline_trust = from_v2_device.baseline_trust
        self.baseline_network = from_v2_device.baseline_network
        self.susceptibility = from_v2_device.susceptibility
        self.targeted = from_v2_device.targeted
        self.compromised = from_v2_device.compromised
        self.isolated = from_v2_device.isolated
        self.leak_factor = from_v2_device.leak_factor
        # v3 bookkeeping
        self._v3_isolation_loss = 0.0      # workload share offline while isolated
        self._v3_scale_loss = 0.0          # cumulative capacity removed (service loss)
        self._v3_migrate_loss = 0.0        # one-tick migrate service loss
        self._v3_migrate_set_tick = None   # tick at which the migrate loss was set
        self._v3_isolation_reentry_tick = None
        self._v3_was_compromised_at_isolation = None


class DigitalTwinV3(DigitalTwinV2):
    """Digital twin with the v3 finite-cost isolation model.

    Reuses v2 dynamics and candidate generation; overrides only:
      - __init__ (wraps v2 devices as DeviceV3, adds the tick counter)
      - advance (tick counter, re-admission, migrate-loss expiry, then v2)
      - apply_action (v3 finite-cost effects from closed_loop_v3.yaml)
      - generate_candidates (v2 candidates + isolation-cap gate)
      - get_telemetry (adds offline_workload_share, service availability
        inputs, false_isolated_device_ticks, isolated/compromised counts)
    """

    def __init__(self, num_devices=100, config=None, cl_config=None):
        cfg = config or get_twin_v2_config()
        cl = cl_config or get_closed_loop_v3_config()
        v2_self = DigitalTwinV2.__new__(DigitalTwinV2)
        DigitalTwinV2.__init__(v2_self, num_devices, config=cfg)
        # Copy EVERY v2 attribute (config, num_devices, targeted_fraction,
        # devices, _device_index, ...), then wrap the devices as DeviceV3.
        self.__dict__.update(v2_self.__dict__)
        self.cl_config = cl
        self.devices = [DeviceV3(d) for d in v2_self.devices]
        self._device_index = {d.id: d for d in self.devices}
        self._v3_tick = -1  # becomes 0 on the first advance()

    # ---- cap -------------------------------------------------------------
    @property
    def max_isolated_count(self) -> int:
        """Hard cap on simultaneously isolated devices (floor of fraction * n)."""
        cap = float(
            self.cl_config.get("isolation_cap", {}).get("max_isolated_fraction", 1.0)
        )
        return int(np.floor(cap * len(self.devices) + 1e-9))

    @property
    def isolation_cap_reached(self) -> bool:
        return self.isolated_count >= self.max_isolated_count

    # ---- counts ----------------------------------------------------------
    @property
    def isolated_count(self):
        return sum(1 for d in self.devices if d.isolated)

    @property
    def compromised_count(self):
        return sum(1 for d in self.devices if d.compromised and not d.isolated)

    # ---- per-tick service loss ------------------------------------------
    def offline_workload_share(self) -> float:
        """Fleet workload share currently offline (isolate + scale + migrate)."""
        total = sum(d.cpu_load + d.network_load for d in self.devices)
        if total <= 0:
            return 0.0
        offline = 0.0
        for d in self.devices:
            if d.isolated:
                offline += d._v3_isolation_loss
            else:
                offline += d._v3_scale_loss + d._v3_migrate_loss
        return float(np.clip(offline, 0.0, 1.0))

    def false_isolated_device_ticks(self) -> int:
        """Devices isolated this tick that were not compromised when isolated."""
        return sum(
            1
            for d in self.devices
            if d.isolated and d._v3_was_compromised_at_isolation is False
        )

    # ---- candidates ------------------------------------------------------
    def generate_candidates(self, tick=0) -> list:
        """v2 candidate set, with isolate candidates dropped at the cap.

        The cap is a capacity constraint: when it is reached NO isolate
        candidate is generated at all, so no strategy can circumvent it by
        picking a different isolate action.
        """
        actions = generate_candidate_actions_v2(self, tick)
        if self.isolation_cap_reached:
            actions = [a for a in actions if a["type"] != "isolate"]
        return actions

    # ---- overrides -------------------------------------------------------
    def advance(self, intensity, closed_loop_cfg=None):
        """One v3 tick: tick counter -> re-admission -> migrate expiry -> v2."""
        self._v3_tick += 1
        # Re-admission BEFORE this tick's dynamics, so a re-admitted device
        # takes part in the tick it returns (walk, recovery, attack exposure).
        isolation_duration = int(
            self.cl_config.get("actions", {}).get("isolate", {}).get(
                "isolation_duration", 10
            )
        )
        for d in self.devices:
            reentry = d._v3_isolation_reentry_tick
            if reentry is not None and self._v3_tick > reentry:
                d.isolated = False
                d._v3_isolation_reentry_tick = None
                d._v3_isolation_loss = 0.0
            elif reentry is not None:
                # Defensive: keep the isolation bounded even if the stamp was
                # written with a stale tick (duration measured from the stamp).
                if self._v3_tick - isolation_duration > reentry:
                    d.isolated = False
                    d._v3_isolation_reentry_tick = None
                    d._v3_isolation_loss = 0.0
        # Migrate service loss is visible for exactly ONE following tick:
        # set at tick t (after advance t), kept through advance t+1, dropped
        # at advance t+2.
        for d in self.devices:
            if (
                d._v3_migrate_set_tick is not None
                and d._v3_migrate_set_tick < self._v3_tick - 1
            ):
                d._v3_migrate_loss = 0.0
                d._v3_migrate_set_tick = None
        super().advance(intensity, closed_loop_cfg=closed_loop_cfg)

    def apply_action(self, action, cl_cfg):
        """Apply v3 closed-loop action effects.

        Effects land on the NEXT tick (called after negotiation of tick t).
        NO energy saving on isolate; service-loss fractions tracked;
        isolation re-admission stamped; concurrency cap enforced.
        """
        effects = cl_cfg["actions"]
        device = self._device_index.get(action.get("target_device_id"))
        if device is None:
            return
        action_type = action["type"]

        if action_type == "isolate":
            spec = effects["isolate"]
            if device.isolated:
                return  # documented: re-isolating has no further effect
            if self.isolation_cap_reached:
                return  # hard guard; candidates are already gated
            was_compromised = bool(device.compromised)
            # Workload share SNAPSHOT taken BEFORE the quarantine zeroes the
            # network load: this is the service that goes offline.
            total = sum(dd.cpu_load + dd.network_load for dd in self.devices)
            share = (
                (device.cpu_load + device.network_load) / total if total > 0 else 0.0
            )
            device.isolated = True
            device.compromised = False  # isolation clears compromise (v2 behaviour)
            device.network_load = 0.0
            device.cpu_load = float(np.clip(device.cpu_load + spec["capacity_penalty"], 0, 1))
            device.energy_kwh = max(0.0, device.energy_kwh * (1.0 - spec.get("energy_saving", 0.0)))
            device.cost_per_hour += spec["cost_penalty"]
            device._v3_was_compromised_at_isolation = was_compromised
            device._v3_isolation_loss = float(share)
            device._v3_isolation_reentry_tick = (
                self._v3_tick + int(spec.get("isolation_duration", 10))
            )

        elif action_type == "migrate":
            spec = effects["migrate"]
            # Service loss = migrate_service_loss_fraction of the target's
            # PRE-action workload share (the load being moved), counted for
            # exactly one following tick.
            frac = float(spec.get("migrate_service_loss_fraction", 0.2))
            total_pre = sum(dd.cpu_load + dd.network_load for dd in self.devices)
            pre_share = (
                (device.cpu_load + device.network_load) / total_pre
                if total_pre > 0 else 0.0
            )
            moved = device.network_load * spec["load_shift"]
            device.network_load = float(np.clip(device.network_load * (1.0 - spec["load_shift"]), 0, 1))
            others = [dd for dd in self.devices if dd is not device and not dd.isolated]
            if others and moved > 0:
                share = moved / len(others)
                for other in others:
                    other.network_load = float(np.clip(other.network_load + share, 0, 1))
            device.leak_factor = min(device.leak_factor, spec["threat_leak"])
            device.energy_kwh += spec["energy_cost"]
            device.cost_per_hour += spec["cost_penalty"]
            if not device.isolated:
                device._v3_migrate_loss = float(frac * pre_share)
                device._v3_migrate_set_tick = self._v3_tick

        elif action_type == "scale_down":
            spec = effects["scale_down"]
            # Service loss = the removed capacity fraction of the device's
            # workload share, measured on the PRE-action state (the share the
            # device was actually serving before it was shrunk).
            cap = float(spec.get("capacity_reduction", 0.1))
            total = sum(dd.cpu_load + dd.network_load for dd in self.devices)
            pre_share = (
                (device.cpu_load + device.network_load) / total if total > 0 else 0.0
            )
            device.energy_kwh = max(0.0, device.energy_kwh * (1.0 - spec["energy_reduction"]))
            device.cpu_load = float(np.clip(device.cpu_load * (1.0 - spec["capacity_reduction"]), 0, 1))
            if not device.isolated:
                device._v3_scale_loss = float(min(1.0, device._v3_scale_loss + cap * pre_share))

        elif action_type == "no_op":
            return
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    # ---- observations ----------------------------------------------------
    def get_telemetry(self):
        telemetry = super().get_telemetry()
        telemetry["offline_workload_share"] = self.offline_workload_share()
        telemetry["service_availability"] = 1.0 - telemetry["offline_workload_share"]
        telemetry["false_isolated_device_ticks"] = self.false_isolated_device_ticks()
        telemetry["isolated_devices"] = self.isolated_count
        telemetry["compromised_devices"] = self.compromised_count
        telemetry["isolation_cap_reached"] = self.isolation_cap_reached
        return telemetry
