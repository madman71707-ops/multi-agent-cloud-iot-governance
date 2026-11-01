"""Phase 2 tests for the redesigned evaluation environment.

Covers the spec's required additions after the Phase 0 audit
(docs/audit_phase0.md):

* 2a - v2 twin dynamics: recovery after an attack ends, the four schedule
  shapes, seed reproducibility, and partial seed-dependent targeting.
* 2b - us_mode="v2": no_op scores ~0.5, isolating a compromised device
  outscores isolating a healthy one, everything stays in [0, 1], the
  legacy mode still delegates to the original legacy scorers, and
  identical_us_fraction detects U_S collapse.
* 2c - closed-loop action effects (isolate / migrate / propagation).
* 2d - the sustained-run rule behind time_to_containment.

Tuning seeds (1000+) are used where a seed is needed; official evaluation
seeds are never touched by tests.
"""
import random

import numpy as np
import pytest

from digital_twin.twin_v2 import (
    DigitalTwinV2,
    build_v2_schedule,
    deep_update,
    generate_candidate_actions_v2,
    get_closed_loop_config,
    get_twin_v2_config,
)
from sim.scoring_v2 import identical_us_fraction, score_actions_v2, score_security_v2


def _seeded_twin(seed=1000, num_devices=100, override=None):
    random.seed(seed)
    np.random.seed(seed)
    cfg = get_twin_v2_config()
    if override:
        cfg = deep_update(cfg, override)
    return DigitalTwinV2(num_devices, config=cfg), cfg


def _mean(values):
    return float(np.mean(values))


# ---------------------------------------------------------------------------
# 2a - twin dynamics: recovery, schedule shapes, seed reproducibility
# ---------------------------------------------------------------------------
def test_trust_and_load_recover_after_the_attack_ends():
    twin, _ = _seeded_twin(seed=1000)
    baseline_trust = _mean([d.baseline_trust for d in twin.devices])
    baseline_load = _mean([d.baseline_network for d in twin.devices])

    for _ in range(30):
        twin.advance(1.0, closed_loop_cfg=None)
    degraded_trust = _mean([d.trust_score for d in twin.devices])
    raised_load = _mean([d.network_load for d in twin.devices])
    assert degraded_trust < baseline_trust - 0.05, "attack did not bite"
    assert raised_load > baseline_load + 0.02, "attack did not raise load"

    for _ in range(60):
        twin.advance(0.0, closed_loop_cfg=None)
    recovered_trust = _mean([d.trust_score for d in twin.devices])
    recovered_load = _mean([d.network_load for d in twin.devices])
    assert recovered_trust > degraded_trust + 0.1, "trust did not recover"
    assert recovered_trust > baseline_trust - 0.05, "trust did not reach baseline"
    assert recovered_load < baseline_load + 0.05, "load did not recover"


def _schedule(scenario, seed=1000, num_ticks=100):
    """Mirror the runner's draw order: twin first, then the schedule."""
    random.seed(seed)
    np.random.seed(seed)
    cfg = get_twin_v2_config()
    DigitalTwinV2(100, config=cfg)
    return build_v2_schedule(scenario, num_ticks, config=cfg)


def test_gradual_schedule_ramps_linearly_to_a_sub_saturation_plateau():
    schedule, info = _schedule("Gradual attack")
    onset = info["onset"]
    ramp = get_twin_v2_config()["schedules"]["gradual"]["ramp_ticks"]
    assert min(schedule) == onset
    steps = [schedule[onset + i] for i in range(min(ramp, 100 - onset))]
    assert all(b > a for a, b in zip(steps, steps[1:])), "ramp must increase"
    tail = [schedule[t] for t in range(onset + ramp, 100)]
    assert tail and all(v == pytest.approx(steps[-1]) for v in tail), "plateau"
    assert max(schedule.values()) < 1.0, "plateau must stay below saturation"


def test_short_spike_is_one_bounded_pulse_then_silence():
    schedule, info = _schedule("Short attack spike")
    shape = get_twin_v2_config()["schedules"]["short_spike"]
    onset = info["onset"]
    pulse = shape["pulse_ticks"]
    assert sorted(schedule) == list(range(onset, onset + pulse))
    peak = schedule[onset]
    assert all(v == pytest.approx(peak) for v in schedule.values())
    assert onset + pulse < 100, "test horizon must outlive the pulse"


def test_repeated_schedule_has_three_separated_pulses():
    schedule, info = _schedule("Repeated attacks")
    shape = get_twin_v2_config()["schedules"]["repeated"]
    onset = info["onset"]
    pulse, gap, n_pulses = shape["pulse_ticks"], shape["gap_ticks"], shape["pulses"]
    starts = [onset + i * (pulse + gap) for i in range(n_pulses)]
    expected = set()
    for start in starts:
        expected.update(range(start, start + pulse))
    assert set(schedule) == expected
    for i in range(n_pulses - 1):
        between = range(starts[i] + pulse, starts[i + 1])
        assert not (set(schedule) & set(between)), "pulses must be separated"


def test_sustained_schedule_is_a_constant_sub_saturation_plateau():
    schedule, info = _schedule("Sustained attack")
    onset = info["onset"]
    assert set(schedule) == set(range(onset, 100))
    values = list(schedule.values())
    assert max(values) == pytest.approx(min(values)), "must be constant"
    assert max(values) < 1.0


def test_the_four_schedules_are_all_different():
    schedules = {name: _schedule(name)[0] for name in (
        "Gradual attack", "Short attack spike", "Repeated attacks",
        "Sustained attack",
    )}
    serialized = [repr(sorted(s.items())) for s in schedules.values()]
    assert len(set(serialized)) == 4


def test_same_seed_reproduces_schedule_and_trajectory_and_new_seed_differs():
    def trace(seed):
        random.seed(seed)
        np.random.seed(seed)
        cfg = get_twin_v2_config()
        twin = DigitalTwinV2(100, config=cfg)
        schedule, info = build_v2_schedule("Gradual attack", 60, config=cfg)
        trust = []
        for tick in range(60):
            twin.advance(schedule.get(tick, 0.0), closed_loop_cfg=None)
            trust.append(twin.get_telemetry()["avg_trust_score"])
        return schedule, info, trust

    first, again = trace(1000), trace(1000)
    other = trace(1001)
    assert first[0] == again[0] and first[1] == again[1]
    assert first[2] == again[2], "same seed must reproduce the trajectory"
    assert other[2] != first[2], "different seed must change the trajectory"
    assert other[0] != first[0] or other[1]["onset"] != first[1]["onset"], (
        "different seed must change the attack itself"
    )


def test_targeting_is_partial_and_seed_dependent():
    fractions = []
    for seed in (1000, 1001, 1002):
        random.seed(seed)
        np.random.seed(seed)
        twin = DigitalTwinV2(100, config=get_twin_v2_config())
        targeted = sum(1 for d in twin.devices if d.targeted)
        assert 0.3 <= twin.targeted_fraction <= 0.6
        assert 0 < targeted < 100, "no seed may target the whole fleet"
        fractions.append(twin.targeted_fraction)
    assert len(set(fractions)) == 3, "targeting must depend on the seed"


# ---------------------------------------------------------------------------
# 2b - us_mode="v2": orderings and legacy preservation
# ---------------------------------------------------------------------------
def test_no_op_scores_half_and_attack_tick_utilities_discriminate():
    twin, cfg = _seeded_twin(seed=1011)
    twin_state = twin.get_telemetry()
    actions = generate_candidate_actions_v2(twin, tick=0)
    _, us_list, _ = score_actions_v2(
        actions, twin, twin_state, twin_state, 0.6, us_mode="v2", config=cfg
    )
    no_op_values = [
        u for a, u in zip(actions, us_list) if a["type"] == "no_op"
    ]
    assert no_op_values, "there is always a no_op candidate"
    assert all(u == pytest.approx(0.5, abs=1e-9) for u in no_op_values)
    assert min(us_list) >= 0.0 and max(us_list) <= 1.0
    assert identical_us_fraction([us_list]) == 0.0, "candidates must differ"


def test_isolate_compromised_outscores_isolate_healthy():
    twin, cfg = _seeded_twin(seed=1012)
    by_trust = sorted(twin.devices, key=lambda d: d.trust_score)
    worst, best = by_trust[0], by_trust[-1]
    worst.trust_score, worst.network_load, worst.compromised = 0.10, 0.90, True
    best.trust_score, best.network_load = 0.95, 0.10
    effect = {
        "cpu_load": -0.02, "network_load": -0.20, "trust_score": 0.05,
        "cost_per_hour": 0.01, "energy_kwh": -0.01,
    }

    def isolate(target, action_id):
        return {
            "id": action_id, "type": "isolate", "target_device_id": target.id,
            "predicted_effect": dict(effect),
        }

    u_compromised = score_security_v2(isolate(worst, "a"), twin, 0.6, config=cfg)
    u_healthy = score_security_v2(isolate(best, "b"), twin, 0.6, config=cfg)
    assert 0.0 <= u_healthy <= 1.0 and 0.0 <= u_compromised <= 1.0
    assert u_compromised > u_healthy, (
        "isolating the compromised device must score higher"
    )
    assert u_healthy >= 0.5, "any helping action beats no change (0.5)"
    assert u_compromised > u_healthy + 0.05, (
        "the ordering must have a real margin, not a rounding tie"
    )


def test_us_v2_is_stable_when_tss_changes_because_the_tss_term_cancels():
    twin, cfg = _seeded_twin(seed=1013)
    actions = generate_candidate_actions_v2(twin, tick=0)
    low = [score_security_v2(a, twin, 0.1, config=cfg) for a in actions]
    high = [score_security_v2(a, twin, 0.9, config=cfg) for a in actions]
    assert low == pytest.approx(high), (
        "the TSS term appears before and after the action and must cancel"
    )


def test_legacy_us_mode_still_delegates_to_the_legacy_scorers():
    from agents.security_agent import score_security
    from agents.resource_agent import score_resource
    from agents.sustainability_agent import score_sustainability

    twin, cfg = _seeded_twin(seed=1014)
    state = twin.get_telemetry()
    actions = generate_candidate_actions_v2(twin, tick=0)

    random.seed(42)
    np.random.seed(42)
    r_legacy, s_legacy, c_legacy = score_actions_v2(
        actions, twin, state, state, 0.5, us_mode="legacy", config=cfg
    )

    random.seed(42)
    np.random.seed(42)
    r_direct = [score_resource(a, state) for a in actions]
    s_direct = [score_security(a, state) for a in actions]
    c_direct = [score_sustainability(a, state) for a in actions]

    assert r_legacy == pytest.approx(r_direct)
    assert s_legacy == pytest.approx(s_direct)
    assert c_legacy == pytest.approx(c_direct)


def test_identical_us_fraction_flags_a_collapsed_and_a_discriminating_tick():
    assert identical_us_fraction([[0.0, 0.0], [0.0, 0.0]]) == 1.0
    assert identical_us_fraction([[0.0, 0.5], [0.5, 0.5]]) == 0.5
    assert identical_us_fraction([[0.0, 1.0]]) == 0.0
    assert np.isnan(identical_us_fraction([]))


# ---------------------------------------------------------------------------
# 2c - closed-loop action effects
# ---------------------------------------------------------------------------
def test_isolate_quarantines_the_target_and_keeps_it_off_the_network():
    twin, _ = _seeded_twin(seed=1015)
    cl_cfg = get_closed_loop_config()
    dev = twin.devices[0]
    dev.compromised, dev.network_load = True, 0.80
    twin.apply_action(
        {"type": "isolate", "target_device_id": dev.id, "predicted_effect": {}},
        cl_cfg,
    )
    assert dev.isolated and not dev.compromised
    assert dev.network_load == 0.0
    for _ in range(3):
        twin.advance(0.5, closed_loop_cfg=cl_cfg)
    assert dev.network_load == 0.0, "isolated devices stay off the network"
    assert not dev.compromised, "isolation clears the compromised flag"


def test_migrate_shifts_load_and_retains_only_the_threat_leak():
    twin, _ = _seeded_twin(seed=1016)
    cl_cfg = get_closed_loop_config()
    dev = twin.devices[0]
    dev.network_load, dev.leak_factor, dev.compromised = 0.80, 1.0, True
    energy_before = dev.energy_kwh
    twin.apply_action(
        {"type": "migrate", "target_device_id": dev.id, "predicted_effect": {}},
        cl_cfg,
    )
    spec = cl_cfg["actions"]["migrate"]
    assert dev.network_load == pytest.approx(0.80 * (1.0 - spec["load_shift"]))
    assert dev.leak_factor == pytest.approx(spec["threat_leak"])
    assert dev.energy_kwh == pytest.approx(energy_before + spec["energy_cost"])
    assert not dev.isolated


def test_scale_down_changes_energy_but_not_threat():
    twin, _ = _seeded_twin(seed=1017)
    cl_cfg = get_closed_loop_config()
    dev = twin.devices[0]
    dev.compromised = True
    energy_before, trust_before = dev.energy_kwh, dev.trust_score
    twin.apply_action(
        {"type": "scale_down", "target_device_id": dev.id, "predicted_effect": {}},
        cl_cfg,
    )
    spec = cl_cfg["actions"]["scale_down"]
    assert dev.energy_kwh == pytest.approx(
        energy_before * (1.0 - spec["energy_reduction"])
    )
    assert dev.trust_score == trust_before
    assert dev.compromised and not dev.isolated


def test_propagation_damages_neighbours_only_in_closed_loop():
    # Zero walk noise isolates the propagation effect exactly.
    random.seed(1018)
    np.random.seed(1018)
    cfg = deep_update(get_twin_v2_config(), {"walk_sigma": 0.0})
    twin = DigitalTwinV2(30, config=cfg)
    cl_cfg = get_closed_loop_config()

    def reset_to_baseline(compromised_source=True):
        for d in twin.devices:
            d.trust_score = d.baseline_trust
            d.network_load = d.baseline_network
            d.compromised = False
            d.isolated = False
            d.leak_factor = 1.0
        if compromised_source:
            twin.devices[0].compromised = True

    baselines = [d.baseline_trust for d in twin.devices]

    reset_to_baseline(compromised_source=False)
    twin.advance(0.0, closed_loop_cfg=None)
    assert all(
        d.trust_score == pytest.approx(b, abs=1e-12)
        for d, b in zip(twin.devices, baselines)
    ), "no attack + no propagation must leave trust at baseline"

    reset_to_baseline(compromised_source=True)
    twin.advance(0.0, closed_loop_cfg=cl_cfg)
    damaged = [
        d.trust_score < b - 1e-9
        for d, b in zip(twin.devices, baselines)
    ]
    assert any(damaged), "propagation must lower some neighbour's trust"
    assert not damaged[0], "the propagation source itself is not damaged"


# ---------------------------------------------------------------------------
# 2d - outcome metrics: the sustained-run containment rule
# ---------------------------------------------------------------------------
def _synthetic_rows(tss_by_tick):
    return [
        {
            "strategy": "TAPNA",
            "scenario": "Gradual attack",
            "seed": 1000,
            "closed_loop": True,
            "us_mode": "v2",
            "tick": tick,
            "TSS": float(value),
            "attack_intensity": 1.0 if tick >= 10 else 0.0,
            "avg_trust_score": 0.85,
            "total_energy_kwh": 1.0,
            "total_cost_per_hour": 1.0,
            "FairScore": 0.5,
            "U_S": 0.5,
            "chosen_action_type": "no_op",
            "compromised_devices": 0,
        }
        for tick, value in enumerate(tss_by_tick)
    ]


def _containment(tss_by_tick, tau=0.15):
    from sim.attack_patterns_v2 import compute_outcome_metrics

    info = {
        "onset": 10,
        "targeted_fraction": 0.5,
        "mean_baseline_trust": 0.85,
        "peak_intensity": 0.5,
    }
    outcome = compute_outcome_metrics(
        _synthetic_rows(tss_by_tick), info, num_ticks=len(tss_by_tick), tau=tau
    )
    return outcome["time_to_containment"], outcome["contained"]


def test_containment_needs_three_sustained_ticks_on_both_sides():
    # (a) attack bites at onset and stays above tau: never contained.
    ttc, contained = _containment([0.10] * 10 + [0.50] * 50)
    assert ttc == 50 and contained is False

    # (b) sustained containment at tick 30 -> 30 - onset ticks.
    ttc, contained = _containment([0.10] * 10 + [0.50] * 20 + [0.10] * 30)
    assert ttc == 20 and contained is True

    # (c) a single-tick dip inside an attack that runs to the horizon is
    # NOT containment (the attack resumes before any 3-tick run builds).
    ttc, contained = _containment(
        [0.10] * 10 + [0.50] * 10 + [0.10] + [0.50] * 39
    )
    assert ttc == 50 and contained is False

    # (d) a single-tick spike at onset is NOT an attack bite.
    ttc, contained = _containment(
        [0.10] * 10 + [0.50] + [0.10] * 49
    )
    assert ttc == 0 and contained is True
