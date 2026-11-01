from pathlib import Path

import numpy as np
import pandas as pd

from ai_models.security.threat_detector import ThreatDetector


_MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "threat_detector.joblib"
_THREAT_DETECTOR = None

# ---------------------------------------------------------------------------
# CICIoT2023 central values derived from the training dataset statistics.
# Normal uses benign medians because the benign mean falls outside the
# IsolationForest's learned inlier region for this skewed feature set.
# Attack uses attack-traffic central values.
# ---------------------------------------------------------------------------
_NORMAL_CENTRAL: dict[str, float] = {
    "rst_count": 0.0,
    "ICMP": 0.0,
    "Min": 54.0,
    "AVG": 54.0,
    "IAT": 83_124_520.23094302,
    "Number": 9.5,
    "Variance": 0.0,
    "Weight": 141.55,
}

_ATTACK_CENTRAL: dict[str, float] = {
    "rst_count": 1070.0,
    "ICMP": 0.0,
    "Min": 180.0,
    "AVG": 618.0,
    "IAT": 83_320_000.0,
    "Number": 9.5,
    "Variance": 0.861,
    "Weight": 141.5,
}

# Realistic clip ranges derived from the CICIoT2023 dataset min/max.
_CLIP_MIN: dict[str, float] = {
    "rst_count": 0.0,
    "ICMP": 0.0,
    "Min": 42.0,
    "AVG": 42.0,
    "IAT": 0.0,
    "Number": 1.5,
    "Variance": 0.0,
    "Weight": 2.5,
}
_CLIP_MAX: dict[str, float] = {
    "rst_count": 9_332.0,
    "ICMP": 1.0,
    "Min": 3_252.0,
    "AVG": 4_060.0,
    "IAT": 167_640_000.0,
    "Number": 15.0,
    "Variance": 1.0,
    "Weight": 245.0,
}


def _get_threat_detector() -> ThreatDetector:
    global _THREAT_DETECTOR

    if _THREAT_DETECTOR is None:
        detector = ThreatDetector()
        detector.load(str(_MODEL_PATH))
        _THREAT_DETECTOR = detector

    return _THREAT_DETECTOR


def _clip01(value: float) -> float:
    return float(min(max(value, 0), 1))


def _telemetry_frame(twin_state) -> pd.DataFrame:
    """Generate the 8 CICIoT2023 security features from Digital Twin state.

    Uses an explicit attack condition derived from the Digital Twin:
        attack = avg_trust_score < 0.7  OR  avg_network_load > 0.7

    Normal telemetry uses the benign central values; attack telemetry
    uses the attack central values.  Small random noise is added for
    realism and clipped to the dataset's observed min/max ranges.

    ``trust_score`` is kept as an extra column so ``compute_tss()``
    can combine anomaly severity with trust degradation.
    """
    if isinstance(twin_state, pd.DataFrame):
        series = twin_state.mean() if len(twin_state) > 1 else twin_state.iloc[0]
        avg_trust = float(series.get("avg_trust_score", series.get("trust_score", 1.0)))
        avg_network = float(series.get("avg_network_load", series.get("network_load", 0.0)))
    else:
        avg_trust = float(twin_state.get("avg_trust_score", twin_state.get("trust_score", 1.0)))
        avg_network = float(twin_state.get("avg_network_load", twin_state.get("network_load", 0.0)))

    attack = avg_trust < 0.7 or avg_network > 0.7
    central = _ATTACK_CENTRAL if attack else _NORMAL_CENTRAL

    # Draw from the global numpy stream instead of an unseeded
    # np.random.default_rng(): the experiment runner calls
    # np.random.seed(seed) for every seed, so this makes TSS/U_S noise
    # reproducible per experiment seed. The distribution is unchanged -
    # the same normal noise per feature, clipped exactly as before.
    rng = np.random
    features: dict[str, float] = {}
    for key in ThreatDetector.FEATURES:
        value = central[key] + rng.normal(0, abs(central[key]) * 0.05 + 0.01)
        features[key] = float(np.clip(value, _CLIP_MIN[key], _CLIP_MAX[key]))

    # Keep trust_score so compute_tss() can blend anomaly + trust degradation.
    features["trust_score"] = avg_trust

    return pd.DataFrame([features])


def score_security(action: dict, twin_state: dict) -> float:
    predicted_trust_after = (
        twin_state["avg_trust_score"] +
        action["predicted_effect"]["trust_score"]
    )
    predicted_network_after = (
        twin_state["avg_network_load"] +
        action["predicted_effect"]["network_load"]
    )

    utility = predicted_trust_after - 0.3 * predicted_network_after
    threat_severity = compute_threat_severity(twin_state)

    return _clip01(utility * (1 - threat_severity))


def compute_threat_severity(twin_state, baseline_trust=0.8) -> float:
    telemetry = _telemetry_frame(twin_state)
    severity = _get_threat_detector().compute_tss(telemetry).mean()

    return _clip01(severity)
