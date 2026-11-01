"""V2 action scoring (``us_mode="v2"``) with an exactly-legacy fallback.

Phase 0 (docs/audit_phase0.md) showed the legacy security utility collapses
under attack: U_S = clip01((avg_trust - 0.3*avg_net) * (1 - TSS)) hits 0 for
EVERY candidate once the fleet-average trust is low enough, so all candidates
tie at U_S = 0 in 94-99% of attack ticks and TAPNA's security weight has
nothing to act on (TAPNA == Static Weighted there).

The v2 security utility scores the predicted *improvement in security
posture caused by the action*, evaluated on the action's TARGET device
(per-device state, not fleet averages):

    risk(trust, network, tss) = tw*(1-trust)^2 + nw*network^2 + tws*tss

    U_S(a) = clip01(0.5 + 0.5 * (risk_before - risk_after(a)) / risk_scale)

risk_before is evaluated on the target's current state; risk_after on the
predicted post-action state (trust/network shifted by the action's
predicted_effect and clipped to [0, 1]; TSS unchanged). Parameters live in
config/twin_v2.yaml (``us_v2``).

Properties (unit-tested in tests/test_phase2_v2.py):
  * bounded in [0, 1];
  * no_op has U_S = 0.5 exactly (zero predicted effect, so
    risk_after == risk_before);
  * the same isolate effect yields a LARGER U_S on a compromised target
    (low trust / high network) than on a healthy one, because the risk
    terms are convex in the deficits (1 - trust) and network;
  * candidates differ whenever their predicted effects differ, so U_S no
    longer collapses under severe threat (audit item 2b). The runner
    reports, per run, the fraction of ticks where U_S is identical across
    ALL candidates (``us_identical_fraction``; ~0 in v2 mode vs ~0.94-0.99
    in legacy mode).

Documented deviations from the form suggested in the Phase 2 brief:
  * a fixed ``risk_scale`` replaces division by max(risk_before, eps).
    Dividing by the target's own risk inflates tiny effects on healthy
    devices (whose risk_before is near zero) enough to BREAK the required
    ordering "isolate compromised > isolate healthy"; the fixed scale
    keeps every U_S in [0, 1] with one common denominator.
  * the TSS term enters risk identically before and after the action
    (candidate actions do not change the fleet TSS in this framework), so
    it cancels in risk_before - risk_after and does NOT move U_S. The
    threat LEVEL is deliberately kept out of U_S: it enters decisions
    through TAPNA's adaptive w_S and ThreatRuleOnly's tau rule. Separation
    of concerns: U_S ranks actions by predicted posture improvement, w_S
    scales how much security matters right now.

``us_mode="legacy"`` reproduces the legacy utilities exactly: the same
score_resource / score_security / score_sustainability calls, in the same
order, as sim/experiment_runner._score_actions (which also preserves the
legacy RNG draw order inside score_security). The legacy path is provided
for equivalence tests and legacy-mode experiments; every v2 grid uses
``us_mode="v2"``. Note that the v2 runner passes the fleet telemetry dict
as ``resource_state``: score_resource reads only the three ``avg_*`` keys,
so the values equal the legacy per-device-frame path.

PER-TICK PERFORMANCE (measured on the reference machine during Phase 2
calibration): the legacy per-action scoring path calls three ML predictors
per candidate, which costs ~2.6 s per tick at the default candidate count and
makes a 4x6x20x100 grid infeasible on this machine. ``us_mode="v2"``
batches the resource and sustainability predictors (which depend only on the
shared fleet state and the per-candidate predicted effect) so the whole tick
scores in well under one tick budget; the legacy path is kept byte-identical
and still used only where it is required (equivalence tests, legacy-mode
experiments). Documented in docs/phase2_summary.md.
"""
import numpy as np

from agents.resource_agent import (
    _get_resource_predictor,
    _metric,
    _predicted_telemetry,
    score_resource,
)
from agents.security_agent import score_security
from agents.sustainability_agent import (
    _get_sustainability_predictor,
    _telemetry_frame,
    _prediction_totals,
    _clip01 as _clip01_sus,
)
from agents.sustainability_agent import score_sustainability
from digital_twin.twin_v2 import get_twin_v2_config


# One-time sustainability predictor warmup. The predictor's internal
# StandardScaler pays a ~1.5 s one-time cost on its very first .predict call
# in a process (it computes feature-range statistics the first time they are
# needed). To avoid paying that cost inside the per-tick scoring loop, this
# module warms the predictor once at import time (best-effort; if it fails the
# first real tick just pays the cost via the lazy re-warm below).
_SUSTAINABILITY_WARMED = False

_FEATURES = [
    "rst_count",
    "ICMP",
    "Min",
    "AVG",
    "IAT",
    "Number",
    "Variance",
    "Weight",
]


def _warm_sustainability_predictor():
    global _SUSTAINABILITY_WARMED
    if _SUSTAINABILITY_WARMED:
        return
    try:
        _get_sustainability_predictor().predict(
            __import__("pandas").DataFrame(
                [{"cpu_load": 0.5, "memory_load": 0.5, "network_load": 0.5}]
            )
        )
    except Exception:
        pass
    _SUSTAINABILITY_WARMED = True


def _clip01(value: float) -> float:
    return float(min(max(float(value), 0.0), 1.0))


def device_risk(trust: float, network: float, tss: float, us_cfg: dict) -> float:
    """Risk of one device state under the config/twin_v2.yaml ``us_v2`` weights.

    Convex in both deficits: improving a worse device (low trust, high
    network) removes more risk than improving a healthy one, which is what
    makes the same predicted effect score higher on compromised targets.
    """
    return float(
        us_cfg["trust_weight"] * (1.0 - trust) ** 2
        + us_cfg["network_weight"] * network ** 2
        + us_cfg["tss_weight"] * tss
    )


def _tss_from_frame(feat_frame, det, trust):
    """Per-tick TSS computed from one pre-built security-feature frame.

    Uses the detector's own :meth:`compute_tss` so the feature clipping, anomaly
    normalization, and anomaly-vs-trust blend are shared with the rest of the
    security agent. ``trust`` is the fleet avg_trust_score for this tick (a
    scalar); it is repeated across device rows, which is correct for the current
    fleet-uniform TSS model.
    """
    return det.compute_tss(feat_frame.assign(trust_score=trust)).ravel()


def _tss_batch(states):
    """Batch per-tick TSS for one or more fleet states.

    Each element of ``states`` is a single fleet telemetry snapshot (the same
    shape :func:`agents.security_agent._telemetry_frame` accepts). Returns a
    ``(len(states), N)`` array, one TSS row per state, each over the N devices
    in that state. For the current scoring model TSS is fleet-uniform within a
    tick, so callers normally read row[0].

    *Important:* this function builds one security-feature frame per state with
    the shared detector RNG, so it advances global randomness. For replay-/
    equivalence-safe runs (open-loop trace scoring, tests), capture the frame
    *once* at the tick boundary and use :func:`_tss_from_frame` on that exact
    frame instead; see the runner's ``score_actions_v2`` legacy path and the
    batched trace scorer.
    """
    from agents.security_agent import _telemetry_frame

    _td = _get_threat_detector()
    rows = []
    for state in states:
        _ff = _telemetry_frame(state)
        trust = float(_ff["trust_score"].iloc[0])
        ff = _ff[_FEATURES]
        rows.append(_tss_from_frame(ff, _td, trust))
    return np.array(rows)


# ---------------------------------------------------------------------------
# Batched TSS computation.
#
# ``compute_threat_severity`` calls ``_telemetry_frame`` (which draws 8
# per-row normal samples from the global numpy rng) and then the sklearn
# ``IsolationForest.decision_function`` *for each call*. sklearn's Python
# dispatcher pays a measurable per-call overhead, so computing TSS for 100
# ticks sequentially costs ~6 ms/tick = ~0.6 s/run just for TSS.
#
# The correct batched approach: pre-build one ``_telemetry_frame`` per tick
# state (advancing the rng exactly once per tick, as ``compute_threat_severity``
# would), concat them, call ``decision_function`` ONCE on the combined numpy
# array, then normalize/clip/blend in numpy per row. This reproduces the SAME
# per-tick TSS values (each row's noise was drawn by its own
# ``_telemetry_frame`` call) at roughly the cost of a single
# ``decision_function`` call. Verified against ``compute_threat_severity`` below
# and in the smoke/verification step.
# ---------------------------------------------------------------------------
_SECURITY_FEATURES = [
    "rst_count",
    "ICMP",
    "Min",
    "AVG",
    "IAT",
    "Number",
    "Variance",
    "Weight",
]


def _tss_batch(states):
    """Return per-tick TSS for a list of telemetry dicts, batched.

    Each dict must carry ``avg_trust_score`` (and, for the anomaly signal,
    ``avg_network_load``) as the twin's ``get_telemetry()`` does. Builds one
    agents.security_agent._telemetry_frame per state (advancing the rng
    exactly as compute_threat_severity would), the 8 security features are
    stacked, decision_function is called once, and the result is normalized/
    clipped/blended per row in numpy.

    Returns a numpy array of shape (len(states),) with per-tick TSS values.
    """
    import pandas as pd

    from agents.security_agent import _get_threat_detector
    from agents.security_agent import _telemetry_frame as _sec_frame

    td = _get_threat_detector()
    if not states:
        return np.array([])
    full_frames = [_sec_frame(s) for s in states]
    feat_frames = [ff[_SECURITY_FEATURES] for ff in full_frames]
    X = pd.concat(feat_frames, ignore_index=True).values.astype(float)
    raw_scores = -td.model.decision_function(X)
    denom = td._train_score_max - td._train_score_min
    if denom == 0.0:
        anomaly_score = np.zeros_like(raw_scores, dtype=float)
    else:
        anomaly_score = np.clip(
            (raw_scores - td._train_score_min) / denom, 0.0, 1.0
        )
    trusts = np.concatenate(
        [ff["trust_score"].to_numpy() for ff in full_frames]
    ).astype(float)
    tss = 0.7 * anomaly_score.ravel() + 0.3 * (1.0 - trusts)
    return np.clip(tss, 0.0, 1.0)


def score_security_v2(action: dict, twin, tss: float, config: dict | None = None) -> float:
    """V2 security utility of one candidate action (see module docstring)."""
    cfg = config or get_twin_v2_config()
    us_cfg = cfg["us_v2"]
    device = twin.get_device(action["target_device_id"])
    if device is None:
        raise KeyError(f"Unknown target device: {action['target_device_id']}")

    effect = action.get("predicted_effect", {})
    trust_after = _clip01(device.trust_score + effect.get("trust_score", 0.0))
    network_after = _clip01(device.network_load + effect.get("network_load", 0.0))

    risk_before = device_risk(device.trust_score, device.network_load, tss, us_cfg)
    risk_after = device_risk(trust_after, network_after, tss, us_cfg)

    improvement = risk_before - risk_after
    return _clip01(0.5 + 0.5 * improvement / us_cfg["risk_scale"])


def _predicted_telemetry_batch(actions, resource_state):
    """Build one ``(n, 3)`` predicted-telemetry frame for all candidate actions.

    Equivalent to calling ``_predicted_telemetry(action, resource_state)`` for
each action and stacking: the fleet state is shared, and each row adds that
action's ``predicted_effect``. Needed by the batched resource scorer because
the per-action ``score_resource`` path calls the predictor once per candidate.
    """
    cpu = float(_metric(resource_state, "avg_cpu_load", "cpu_load"))
    mem = float(_metric(resource_state, "avg_memory_load", "memory_load"))
    net = float(_metric(resource_state, "avg_network_load", "network_load"))

    rows = []
    for action in actions:
        pe = action.get("predicted_effect", {})
        rows.append(
            {
                "cpu_load": min(max(cpu + pe.get("cpu_load", 0.0), 0.0), 1.0),
                "memory_load": min(
                    max(mem + pe.get("memory_load", 0.0), 0.0), 1.0
                ),
                "network_load": min(
                    max(net + pe.get("network_load", 0.0), 0.0), 1.0
                ),
            }
        )
    return rows


def score_actions_v2(
    actions,
    twin,
    twin_state,
    resource_state,
    tss,
    us_mode: str = "v2",
    config: dict | None = None,
):
    """Score all candidate actions; returns ``(U_R_list, U_S_list, U_C_list)``.

    ``us_mode="legacy"`` reproduces sim/experiment_runner._score_actions
    exactly (same calls, same order, same RNG consumption). ``us_mode="v2"``
    keeps the legacy resource and sustainability utilities and replaces only
    U_S with :func:`score_security_v2`; in v2 mode scoring consumes no RNG
    at all (TSS is supplied by the caller).

    The v2 path batches the resource and sustainability predictors (they
    depend only on the shared fleet state and the per-candidate predicted
    effect) so the whole tick costs well under one tick budget; the U_S
    loop remains per-action because it reads per-device state from the twin.
    """
    if us_mode == "legacy":
        U_R_list = [score_resource(action, resource_state) for action in actions]
        U_S_list = [score_security(action, twin_state) for action in actions]
        U_C_list = [score_sustainability(action, twin_state) for action in actions]
        return U_R_list, U_S_list, U_C_list
    if us_mode != "v2":
        raise ValueError(f"Unknown us_mode: {us_mode!r} (expected 'v2' or 'legacy')")

    if not _SUSTAINABILITY_WARMED:
        _warm_sustainability_predictor()  # lazy re-warm if module-level warm failed

    # --- U_R: batched resource prediction (identical values to per-action) ---
    rows = _predicted_telemetry_batch(actions, resource_state)
    resource_pressure = _get_resource_predictor().predict(
        __import__("pandas").DataFrame(rows)
    )
    U_R_list = [
        float(_clip01(1.0 - float(p))) for p in resource_pressure
    ]

    # --- U_S: per-action (uses the supplied TSS, no detector call) ---------
    # Note: ``tss`` is supplied by the caller (_environment_step). The caller
    # may compute it via the batched helper :func:`_tss_batch` to avoid calling
    # ``compute_threat_severity`` once per tick.
    U_S_list = [
        score_security_v2(action, twin, tss, config=config) for action in actions
    ]

    # --- U_C: batched sustainability prediction (identical values) ---------
    sus_cfg_required = (
        __import__("ai_models.sustainability.sustainability_predictor", fromlist=["SustainabilityPredictor"])
        .SustainabilityPredictor.REQUIRED_FEATURES
    )
    telemetry_current = _telemetry_frame(twin_state)
    baseline_energy, baseline_cost = _prediction_totals(telemetry_current)
    baseline_energy = max(float(baseline_energy), 1e-9)
    baseline_cost = max(float(baseline_cost), 1e-9)

    rows_sus = []
    for action in actions:
        pe = action.get("predicted_effect", {})
        row = {
            "cpu_load": float(_metric(twin_state, "cpu_load", "avg_cpu_load")),
            "memory_load": float(
                _metric(twin_state, "memory_load", "avg_memory_load")
            ),
            "network_load": float(
                _metric(twin_state, "network_load", "avg_network_load")
            ),
        }
        for feature in sus_cfg_required:
            if feature in pe:
                row[feature] = min(max(row[feature] + pe[feature], 0.0), 1.0)
        rows_sus.append(row)

    predictions = _get_sustainability_predictor().predict(
        __import__("pandas").DataFrame(rows_sus)[sus_cfg_required]
    )
    energy_after = predictions["energy_kwh"]
    cost_after = predictions["cost_per_hour"]

    U_C_list = []
    for i in range(len(actions)):
        ea = float(energy_after[i])
        ca = float(cost_after[i])
        energy_reduction = max(0.0, 1.0 - (ea / baseline_energy))
        cost_reduction = max(0.0, 1.0 - (ca / baseline_cost))
        raw_utility = 0.5 * (energy_reduction + cost_reduction)
        U_C_list.append(_clip01_sus(float(raw_utility)))

    return U_R_list, U_S_list, U_C_list


def identical_us_fraction(U_S_per_tick) -> float:
    """Fraction of ticks where U_S is identical across ALL candidates.

    Phase 0 metric for the U_S collapse: legacy mode saturates near 0.94-0.99
    during attacks; v2 mode should be ~0 (exact ties can still occur when two
    candidates have identical predicted effects on the same target).
    ``U_S_per_tick`` is a sequence of per-tick U_S lists.
    """
    ticks = list(U_S_per_tick)
    if not ticks:
        return float("nan")
    flat = sum(1 for values in ticks if len(values) and max(values) == min(values))
    return flat / len(ticks)


# ---------------------------------------------------------------------------
# Batched TSS computation.
#
# ``compute_threat_severity`` calls ``_telemetry_frame`` (which draws 8
# per-row normal samples from the global numpy rng) and then the sklearn
# ``IsolationForest.decision_function`` *for each call*. sklearn's Python
# dispatcher pays a measurable per-call overhead, so computing TSS for 100
# ticks sequentially costs ~6 ms/tick = ~0.6 s/run just for TSS.
#
# The correct batched approach: pre-build one ``_telemetry_frame`` per tick
# state (advancing the rng exactly once per tick, as ``compute_threat_severity``
# would), concat them, call ``decision_function`` ONCE on the combined numpy
# array, then normalize/clip/blend in numpy per row. This reproduces the SAME
# per-tick TSS values (each row's noise was drawn by its own
# ``_telemetry_frame`` call) at roughly the cost of a single
# ``decision_function`` call. Verified against ``compute_threat_severity`` below
# and in the smoke/verification step.
# ---------------------------------------------------------------------------
_SECURITY_FEATURES = [
    "rst_count",
    "ICMP",
    "Min",
    "AVG",
    "IAT",
    "Number",
    "Variance",
    "Weight",
]


def _tss_batch(states):
    """Return per-tick TSS for a list of telemetry dicts, batched.

    Each dict must carry ``avg_trust_score`` (and, for the anomaly signal,
    ``avg_network_load``) as the twin's ``get_telemetry()`` does. Builds one
    agents.security_agent._telemetry_frame per state (advancing the rng
    exactly as compute_threat_severity would), the 8 security features are
    stacked, decision_function is called once, and the result is normalized/
    clipped/blended per row in numpy.

    Returns a numpy array of shape (len(states),) with per-tick TSS values.
    """
    import pandas as pd

    from agents.security_agent import _get_threat_detector
    from agents.security_agent import _telemetry_frame as _sec_frame

    td = _get_threat_detector()
    if not states:
        return np.array([])
    full_frames = [_sec_frame(s) for s in states]
    feat_frames = [ff[_SECURITY_FEATURES] for ff in full_frames]
    X = pd.concat(feat_frames, ignore_index=True).values.astype(float)
    raw_scores = -td.model.decision_function(X)
    denom = td._train_score_max - td._train_score_min
    if denom == 0.0:
        anomaly_score = np.zeros_like(raw_scores, dtype=float)
    else:
        anomaly_score = np.clip(
            (raw_scores - td._train_score_min) / denom, 0.0, 1.0
        )
    trusts = np.concatenate(
        [ff["trust_score"].to_numpy() for ff in full_frames]
    ).astype(float)
    tss = 0.7 * anomaly_score.ravel() + 0.3 * (1.0 - trusts)
    return np.clip(tss, 0.0, 1.0)


def score_actions_v2_batch_eq(actions, twin, twin_state, resource_state, tss, config):
    """Return ``True`` if the batched v2 scorer and the per-action v2 scorer

    agree on all three utility lists for the given tick (within float tolerance).
    Used by smoke/verification only; not called during production runs.
    """
    import pandas as pd
    from agents.resource_agent import score_resource
    from agents.sustainability_agent import score_sustainability

    ur_b, us_b, uc_b = score_actions_v2(
        actions, twin, twin_state, resource_state, tss, us_mode="v2", config=config
    )
    ur_p = [score_resource(a, resource_state) for a in actions]
    us_p = [score_security_v2(a, twin, tss, config=config) for a in actions]
    uc_p = [score_sustainability(a, twin_state) for a in actions]

    def close(a, b):
        return all(
            abs(x - y) <= 1e-9 + 1e-12 * max(abs(x), abs(y))
            for x, y in zip(a, b)
        )

    return close(ur_b, ur_p) and close(us_b, us_p) and close(uc_b, uc_p)


# Warm the sustainability predictor once at import time so the first v2 tick
# in any process is fast. Best-effort; failures are silent and the lazy path
# in score_actions_v2 will pay the cost on first use.
try:
    _warm_sustainability_predictor()
except Exception:
    pass
