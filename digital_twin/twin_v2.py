"""Digital Twin v2 dynamics, v2 attack schedules and closed-loop actions.

New "v2" mode introduced after the Phase 0 audit (docs/audit_phase0.md).
The legacy twin (digital_twin/twin.py) and every legacy experiment path are
untouched; nothing imports or modifies them.

What v2 changes and why:

* **Mean reversion** - trust and network load relax toward per-device
  baselines every tick (``trust_recovery_rate`` / ``load_recovery_rate``),
  so the twin recovers after an attack ends instead of staying saturated
  forever (audit finding: the legacy twin never recovers).
* **Per-device heterogeneity** - each device has its own baseline trust,
  baseline network load and attack susceptibility, and only a seed-drawn
  fraction of devices (default 30-60%) is targeted by the attack campaign.
  Default intensities therefore cannot drive all devices to trust 0 /
  network 1 (audit finding: legacy injections hit every device).
* **Seed-dependent schedules** - attack onset (base +/- jitter) and
  intensity (+/- jitter) are drawn from the experiment's seeded numpy
  stream, so seeds change the attack itself, not only the noise.
* **Closed-loop actions** (``closed_loop=True``) - the chosen action is
  applied after negotiation and changes the state the next tick sees
  (see config/closed_loop.yaml). Compromised non-isolated devices spread
  threat to neighbours each tick (``propagation``); isolating a device
  stops its propagation and exposure. With ``closed_loop=False`` the
  environment is action-independent (open-loop), like every legacy path.

RNG contract: the caller seeds ``random`` and ``np.random`` exactly once
per run before constructing the twin. Draw order is fixed and documented:
device attributes (vectorized, in the order given in ``__init__``), then
the targeted set, then the schedule's onset and intensity jitter. Given
the same seed the twin, schedule and every trajectory are reproducible.
"""
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
TWIN_V2_CONFIG_PATH = CONFIG_DIR / "twin_v2.yaml"
CLOSED_LOOP_CONFIG_PATH = CONFIG_DIR / "closed_loop.yaml"


def load_config(path):
    """Load a YAML config file into a fresh dict (deep copy per call)."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def deep_update(base, override):
    """Recursively merge ``override`` into ``base`` (returns a new dict)."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_twin_v2_config(path=None):
    return load_config(path or TWIN_V2_CONFIG_PATH)


def get_closed_loop_config(path=None):
    return load_config(path or CLOSED_LOOP_CONFIG_PATH)


@dataclass
class DeviceV2:
    """One simulated device with v2 state and parameters."""

    id: str
    cpu_load: float
    memory_load: float
    network_load: float
    trust_score: float
    cost_per_hour: float
    energy_kwh: float
    baseline_trust: float
    baseline_network: float
    susceptibility: float
    targeted: bool
    compromised: bool = False
    isolated: bool = False
    leak_factor: float = 1.0


SCENARIO_SCHEDULE_KEYS = {
    "Gradual attack": "gradual",
    "Short attack spike": "short_spike",
    "Repeated attacks": "repeated",
    "Sustained attack": "sustained",
}


class DigitalTwinV2:
    """Digital twin with v2 dynamics (mean reversion + heterogeneity)."""

    def __init__(self, num_devices=100, config=None):
        cfg = get_twin_v2_config() if config is None else dict(config)
        self.config = cfg
        n = num_devices
        self.num_devices = n

        # --- Fixed draw order (documented RNG contract) --------------------
        cpu = np.random.uniform(0.10, 0.80, n)
        memory = np.random.uniform(0.20, 0.90, n)
        baseline_network = np.random.uniform(*cfg["baseline_network_range"], n)
        baseline_trust = np.random.uniform(*cfg["baseline_trust_range"], n)
        susceptibility = np.random.uniform(*cfg["susceptibility_range"], n)
        trust_noise = np.random.normal(0.0, cfg["initial_noise_sigma"], n)
        network_noise = np.random.normal(0.0, cfg["initial_noise_sigma"], n)
        cost = np.random.uniform(0.05, 0.30, n)
        energy = np.random.uniform(0.10, 1.50, n)

        fraction = np.random.uniform(*cfg["attack"]["targeted_fraction_range"])
        self.targeted_fraction = float(fraction)
        k = int(round(fraction * n))
        targeted_idx = set(
            np.random.choice(n, size=k, replace=False).tolist()
        ) if k else set()

        self.devices = [
            DeviceV2(
                id=f"dev-{i:04d}",
                cpu_load=float(cpu[i]),
                memory_load=float(memory[i]),
                network_load=float(
                    min(max(baseline_network[i] + network_noise[i], 0.0), 1.0)
                ),
                trust_score=float(
                    min(max(baseline_trust[i] + trust_noise[i], 0.0), 1.0)
                ),
                cost_per_hour=float(cost[i]),
                energy_kwh=float(energy[i]),
                baseline_trust=float(baseline_trust[i]),
                baseline_network=float(baseline_network[i]),
                susceptibility=float(susceptibility[i]),
                targeted=(i in targeted_idx),
            )
            for i in range(n)
        ]
        self._device_index = {d.id: d for d in self.devices}

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def advance(self, intensity, closed_loop_cfg=None):
        """Advance one tick under the given attack intensity.

        Order: random walk -> attack pressure -> propagation (closed loop
        only) -> mean reversion -> clipping. ``closed_loop_cfg=None``
        means open-loop: no propagation and (by the caller's contract) no
        action effects, so the environment is strategy-independent.
        """
        cfg = self.config
        sigma = cfg["walk_sigma"]

        for device in self.devices:
            if device.isolated:
                device.network_load = 0.0  # off the network
                continue
            device.cpu_load = float(
                np.clip(device.cpu_load + np.random.normal(0, sigma), 0, 1)
            )
            device.memory_load = float(
                np.clip(device.memory_load + np.random.normal(0, sigma), 0, 1)
            )
            device.network_load = float(
                np.clip(device.network_load + np.random.normal(0, sigma), 0, 1)
            )
            device.trust_score = float(
                np.clip(device.trust_score + np.random.normal(0, sigma), 0, 1)
            )

        if intensity > 0:
            attack = cfg["attack"]
            for device in self.devices:
                if device.targeted and not device.isolated:
                    device.trust_score = float(np.clip(
                        device.trust_score
                        - intensity * device.susceptibility * attack["trust_rate"],
                        0, 1,
                    ))
                    device.network_load = float(np.clip(
                        device.network_load
                        + intensity * device.susceptibility * attack["load_rate"],
                        0, 1,
                    ))
                    # Compromise is sticky: it persists until the device is
                    # isolated (closed loop), so propagation continues after
                    # the attack itself ends.
                    device.compromised = True

        if closed_loop_cfg is not None:
            self._propagate(closed_loop_cfg)

        for device in self.devices:
            device.trust_score = float(np.clip(
                device.trust_score
                + cfg["trust_recovery_rate"]
                * (device.baseline_trust - device.trust_score),
                0, 1,
            ))
            if device.isolated:
                device.network_load = 0.0
            else:
                device.network_load = float(np.clip(
                    device.network_load
                    + cfg["load_recovery_rate"]
                    * (device.baseline_network - device.network_load),
                    0, 1,
                ))

    def _propagate(self, cl_cfg):
        """Compromised non-isolated devices damage their neighbours."""
        prop = cl_cfg["propagation"]
        sources = [d for d in self.devices if d.compromised and not d.isolated]
        candidates = [d for d in self.devices if not d.isolated]
        if not sources or len(candidates) < 2:
            return
        n_neighbours = min(int(prop["neighbours"]), len(candidates) - 1)
        for source in sources:
            impact = prop["rate"] * prop["impact"] * source.leak_factor
            pool = [d for d in candidates if d is not source]
            picks = np.random.choice(len(pool), size=n_neighbours, replace=False)
            for position in picks:
                neighbour = pool[int(position)]
                neighbour.trust_score = float(
                    np.clip(neighbour.trust_score - impact, 0, 1)
                )
                neighbour.network_load = float(
                    np.clip(neighbour.network_load + impact, 0, 1)
                )

    # ------------------------------------------------------------------
    # Closed-loop action application
    # ------------------------------------------------------------------
    def apply_action(self, action, cl_cfg):
        """Apply the chosen action; its effects land on the NEXT tick.

        Called after negotiation of tick t, before advance() of tick t+1.
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
            device.isolated = True
            device.compromised = False
            device.network_load = 0.0
            device.cpu_load = float(
                np.clip(device.cpu_load + spec["capacity_penalty"], 0, 1)
            )
            device.energy_kwh = max(
                0.0, device.energy_kwh * (1.0 - spec["energy_saving"])
            )
            device.cost_per_hour += spec["cost_penalty"]
        elif action_type == "migrate":
            spec = effects["migrate"]
            moved = device.network_load * spec["load_shift"]
            device.network_load = float(
                np.clip(device.network_load * (1.0 - spec["load_shift"]), 0, 1)
            )
            others = [
                d for d in self.devices
                if d is not device and not d.isolated
            ]
            if others and moved > 0:
                share = moved / len(others)
                for other in others:
                    other.network_load = float(
                        np.clip(other.network_load + share, 0, 1)
                    )
            device.leak_factor = min(device.leak_factor, spec["threat_leak"])
            device.energy_kwh += spec["energy_cost"]
            device.cost_per_hour += spec["cost_penalty"]
        elif action_type == "scale_down":
            spec = effects["scale_down"]
            device.energy_kwh = max(
                0.0, device.energy_kwh * (1.0 - spec["energy_reduction"])
            )
            device.cpu_load = float(
                np.clip(device.cpu_load * (1.0 - spec["capacity_reduction"]), 0, 1)
            )
        elif action_type == "no_op":
            return
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def get_telemetry(self):
        """Fleet-level telemetry, same keys as the legacy generator."""
        devices = self.devices
        return {
            "avg_cpu_load": float(
                sum(d.cpu_load for d in devices) / len(devices)
            ),
            "avg_memory_load": float(
                sum(d.memory_load for d in devices) / len(devices)
            ),
            "avg_network_load": float(
                sum(d.network_load for d in devices) / len(devices)
            ),
            "avg_trust_score": float(
                sum(d.trust_score for d in devices) / len(devices)
            ),
            "total_cost_per_hour": float(
                sum(d.cost_per_hour for d in devices)
            ),
            "total_energy_kwh": float(
                sum(d.energy_kwh for d in devices)
            ),
        }

    def get_device(self, device_id):
        """Return the DeviceV2 with the given id, or None if unknown."""
        return self._device_index.get(device_id)

    def get_state(self):
        """Per-device state frame (used for target-aware U_S and tests)."""
        return pd.DataFrame(
            [
                {
                    "id": d.id,
                    "cpu_load": d.cpu_load,
                    "memory_load": d.memory_load,
                    "network_load": d.network_load,
                    "trust_score": d.trust_score,
                    "cost_per_hour": d.cost_per_hour,
                    "energy_kwh": d.energy_kwh,
                    "baseline_trust": d.baseline_trust,
                    "baseline_network": d.baseline_network,
                    "susceptibility": d.susceptibility,
                    "targeted": d.targeted,
                    "compromised": d.compromised,
                    "isolated": d.isolated,
                    "leak_factor": d.leak_factor,
                }
                for d in self.devices
            ]
        )

    @property
    def compromised_count(self):
        return sum(1 for d in self.devices if d.compromised and not d.isolated)

    @property
    def isolated_count(self):
        return sum(1 for d in self.devices if d.isolated)


def build_v2_schedule(scenario, num_ticks, config=None, rng=None):
    """Build the v2 intensity schedule for one scenario run.

    Uses the already-seeded numpy stream (``rng`` defaults to ``np.random``)
    for the onset and intensity jitter; the targeted set was drawn earlier
    in ``DigitalTwinV2.__init__`` (fixed draw order, see module docstring).

    Returns ``(schedule, info)`` where ``schedule`` maps tick -> intensity
    (absent ticks mean intensity 0) and ``info`` records onset, peak
    intensity and the targeted fraction for reproducibility checks.
    """
    cfg = get_twin_v2_config() if config is None else config
    rng = np.random if rng is None else rng
    key = SCENARIO_SCHEDULE_KEYS.get(scenario)
    if key is None:
        raise ValueError(f"Unknown v2 scenario: {scenario}")

    attack = cfg["attack"]
    onset = int(
        round(attack["onset_base"] + rng.uniform(
            -attack["onset_jitter"], attack["onset_jitter"]
        ))
    )
    onset = max(0, min(onset, num_ticks - 1))
    jitter = 1.0 + rng.uniform(
        -attack["intensity_jitter"], attack["intensity_jitter"]
    )
    shape = cfg["schedules"][key]
    schedule = {}

    if key == "gradual":
        peak = shape["plateau_intensity"] * jitter
        ramp = max(1, int(shape["ramp_ticks"]))
        for tick in range(onset, num_ticks):
            steps_in = tick - onset + 1
            schedule[tick] = peak * min(1.0, steps_in / ramp)
    elif key == "short_spike":
        peak = shape["pulse_intensity"] * jitter
        for tick in range(onset, min(num_ticks, onset + int(shape["pulse_ticks"]))):
            schedule[tick] = peak
    elif key == "repeated":
        peak = shape["pulse_intensity"] * jitter
        pulse = int(shape["pulse_ticks"])
        gap = int(shape["gap_ticks"])
        for pulse_index in range(int(shape["pulses"])):
            start = onset + pulse_index * (pulse + gap)
            for tick in range(start, min(num_ticks, start + pulse)):
                schedule[tick] = peak
    elif key == "sustained":
        peak = shape["intensity"] * jitter
        for tick in range(onset, num_ticks):
            schedule[tick] = peak

    info = {
        "onset": onset,
        "peak_intensity": max(schedule.values()) if schedule else 0.0,
        "targeted_fraction": None,  # filled by the runner from the twin
    }
    return schedule, info


def generate_candidate_actions_v2(twin, tick=0) -> list:
    """V2 candidate generator - same distribution as the legacy one.

    Mirrors digital_twin.action_generator.generate_candidate_actions
    (5-10 global candidates; first isolate always targets the lowest-trust
    device; exactly one no_op; the rest random types on random targets with
    the legacy effect ranges) but reads the v2 twin's device list and uses
    deterministic candidate ids (t{tick}-c{i}), which keeps replay checks
    free of the legacy uuid identity columns.
    """
    devices = twin.devices
    total_actions = random.randint(5, 10)
    lowest_trust_device = min(devices, key=lambda d: d.trust_score)

    def make(action_id, action_type, target, cpu, network, trust, cost, energy):
        return {
            "id": action_id,
            "type": action_type,
            "target_device_id": target.id,
            "predicted_effect": {
                "cpu_load": cpu,
                "network_load": network,
                "trust_score": trust,
                "cost_per_hour": cost,
                "energy_kwh": energy,
            },
        }

    actions = [
        make(
            f"t{tick}-c0", "isolate", lowest_trust_device,
            random.uniform(-0.05, 0.02), random.uniform(-0.30, -0.10),
            random.uniform(0.02, 0.08), random.uniform(0.00, 0.03),
            random.uniform(-0.05, 0.02),
        ),
        make(
            f"t{tick}-c1", "no_op", random.choice(devices),
            0.0, 0.0, 0.0, 0.0, 0.0,
        ),
    ]

    action_types = ["migrate", "scale_down", "isolate"]
    effect_ranges = {
        "migrate": {
            "cpu_load": (-0.10, 0.04),
            "network_load": (-0.05, 0.10),
            "trust_score": (-0.02, 0.04),
            "cost_per_hour": (-0.02, 0.04),
            "energy_kwh": (-0.08, 0.05),
        },
        "scale_down": {
            "cpu_load": (-0.12, -0.02),
            "network_load": (-0.04, 0.02),
            "trust_score": (-0.03, 0.01),
            "cost_per_hour": (-0.05, -0.01),
            "energy_kwh": (-0.15, -0.03),
        },
        "isolate": {
            "cpu_load": (-0.05, 0.02),
            "network_load": (-0.30, -0.10),
            "trust_score": (0.02, 0.08),
            "cost_per_hour": (0.00, 0.03),
            "energy_kwh": (-0.05, 0.02),
        },
    }

    index = len(actions)
    while len(actions) < total_actions:
        action_type = random.choice(action_types)
        ranges = effect_ranges[action_type]
        actions.append(
            make(
                f"t{tick}-c{index}", action_type, random.choice(devices),
                *[random.uniform(*ranges[key]) for key in (
                    "cpu_load", "network_load", "trust_score",
                    "cost_per_hour", "energy_kwh",
                )],
            )
        )
        index += 1

    return actions
