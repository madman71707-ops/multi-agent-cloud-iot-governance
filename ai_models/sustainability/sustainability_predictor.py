from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SustainabilityPredictor:
    """Predict energy consumption and operational cost from normalized telemetry."""

    REQUIRED_FEATURES = ["cpu_load", "memory_load", "network_load"]

    def __init__(self, random_state: int = 42, electricity_price_per_kwh: float = 0.12) -> None:
        self.random_state = random_state
        self.electricity_price_per_kwh = electricity_price_per_kwh
        self.energy_model: RandomForestRegressor | None = None
        self.feature_scaler = StandardScaler()
        self.feature_ranges: dict[str, tuple[float, float]] = {}
        self.energy_validation_mae: float | None = None
        self.cost_validation_mae: float | None = None

    def _get_feature_ranges(self, data: pd.DataFrame | None = None) -> dict[str, tuple[float, float]]:
        if data is not None:
            ranges: dict[str, tuple[float, float]] = {}
            for feature in self.REQUIRED_FEATURES:
                values = pd.to_numeric(data[feature], errors="coerce").dropna()
                if values.empty:
                    ranges[feature] = (0.0, 1.0)
                    continue
                lo = float(values.min())
                hi = float(values.max())
                if hi <= lo:
                    hi = lo + 1.0
                ranges[feature] = (lo, hi)
            self.feature_ranges = ranges
            return ranges

        if self.feature_ranges:
            return self.feature_ranges

        project_root = Path(__file__).resolve().parents[2]
        source_path = project_root / "data" / "real_sustainability" / "job_table.parquet"
        if not source_path.exists():
            self.feature_ranges = {feature: (0.0, 1.0) for feature in self.REQUIRED_FEATURES}
            return self.feature_ranges

        try:
            raw_data = pd.read_parquet(source_path)
        except Exception:
            self.feature_ranges = {feature: (0.0, 1.0) for feature in self.REQUIRED_FEATURES}
            return self.feature_ranges

        column_map = {
            "cpu_load": "num_cores_alloc",
            "memory_load": "mem_alloc",
            "network_load": "run_time",
        }
        remapped = pd.DataFrame()
        for feature in self.REQUIRED_FEATURES:
            source_col = column_map.get(feature, feature)
            if source_col in raw_data.columns:
                remapped[feature] = pd.to_numeric(raw_data[source_col], errors="coerce")
            else:
                remapped[feature] = pd.to_numeric(raw_data[feature], errors="coerce")
        return self._get_feature_ranges(remapped)

    def _transform_inference_features(self, data: pd.DataFrame) -> pd.DataFrame:
        transformed = data[self.REQUIRED_FEATURES].copy()
        feature_ranges = self._get_feature_ranges()

        for feature in self.REQUIRED_FEATURES:
            values = pd.to_numeric(transformed[feature], errors="coerce")
            lo, hi = feature_ranges.get(feature, (0.0, 1.0))
            if hi <= lo:
                continue
            if values.notna().all() and values.min() >= 0.0 and values.max() <= 1.0:
                transformed[feature] = lo + values * (hi - lo)

        return transformed

    def train(self, data: pd.DataFrame) -> None:
        self._validate_features(data)

        features = data[self.REQUIRED_FEATURES]
        if "power_consumption" not in data.columns:
            raise ValueError("Missing required target: power_consumption")

        energy_kwh = data["power_consumption"]

        x_train, x_val, y_energy_train, y_energy_val = train_test_split(
            features,
            energy_kwh,
            test_size=0.2,
            random_state=self.random_state,
        )

        self.energy_model = RandomForestRegressor(
            n_estimators=100,
            random_state=self.random_state,
        )
        self._get_feature_ranges(features)
        x_train_scaled = self.feature_scaler.fit_transform(x_train)
        x_val_scaled = self.feature_scaler.transform(x_val)

        self.energy_model.fit(x_train_scaled, y_energy_train)

        energy_predictions = self.energy_model.predict(x_val_scaled)

        self.energy_validation_mae = float(mean_absolute_error(y_energy_val, energy_predictions))
        self.cost_validation_mae = None

    def predict(self, data: pd.DataFrame) -> dict:
        self._validate_features(data)
        if self.energy_model is None:
            raise ValueError("Models are not trained or loaded.")

        features = self._transform_inference_features(data)
        features_scaled = self.feature_scaler.transform(features)
        energy_predictions = np.asarray(self.energy_model.predict(features_scaled))
        return {
            "energy_kwh": energy_predictions,
            "cost_per_hour": energy_predictions * self.electricity_price_per_kwh,
        }

    def save(self, path: str) -> None:
        if self.energy_model is None:
            raise ValueError("Models are not trained.")

        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "energy_model": self.energy_model,
                "feature_scaler": self.feature_scaler,
                "feature_ranges": self.feature_ranges,
                "electricity_price_per_kwh": self.electricity_price_per_kwh,
                "energy_validation_mae": self.energy_validation_mae,
                "cost_validation_mae": self.cost_validation_mae,
            },
            model_path,
        )

    def load(self, path: str) -> None:
        payload = joblib.load(path)
        self.energy_model = payload["energy_model"]
        self.feature_scaler = payload.get("feature_scaler", StandardScaler())
        self.feature_ranges = payload.get("feature_ranges", {})
        self.electricity_price_per_kwh = payload.get(
            "electricity_price_per_kwh", self.electricity_price_per_kwh
        )
        self.energy_validation_mae = payload.get("energy_validation_mae")
        self.cost_validation_mae = payload.get("cost_validation_mae")
        if not self.feature_ranges:
            self._get_feature_ranges()

    def _validate_features(self, data: pd.DataFrame) -> None:
        missing_features = [
            feature for feature in self.REQUIRED_FEATURES if feature not in data.columns
        ]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")


