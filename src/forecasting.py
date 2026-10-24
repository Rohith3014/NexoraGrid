"""
EcoGrid AI - Forecasting Module
Time-series forecasting for electricity, water, and solar generation.
"""
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
try:
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    HAS_SKLEARN_ENSEMBLE = True
except Exception:
    HAS_SKLEARN_ENSEMBLE = False
    HistGradientBoostingRegressor, RandomForestRegressor = None, None

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
from .config import PROJECT_ROOT, get_config
from .utils import get_logger, timer, safe_divide

logger = get_logger("forecasting")

RESOURCE_COLS = {
    "electricity": "electricity_kwh",
    "water": "water_litres",
    "solar": "solar_generation_kwh"
}


def create_lag_features(df: pd.DataFrame, target_col: str, lags: list = None,
                        rolling_windows: list = None) -> pd.DataFrame:
    """Create lagged and rolling features for time-series forecasting."""
    config = get_config()
    if lags is None:
        lags = config["forecasting"]["lag_features"]
    if rolling_windows is None:
        rolling_windows = config["forecasting"]["rolling_windows"]
    
    result = df.copy()
    
    for lag in lags:
        result[f"{target_col}_lag_{lag}"] = result[target_col].shift(lag)
    
    for window in rolling_windows:
        result[f"{target_col}_rolling_mean_{window}"] = (
            result[target_col].shift(1).rolling(window=window, min_periods=1).mean()
        )
        result[f"{target_col}_rolling_std_{window}"] = (
            result[target_col].shift(1).rolling(window=window, min_periods=1).std()
        )
    
    return result


def prepare_forecast_features(hourly_df: pd.DataFrame, building_id: str,
                              resource: str) -> pd.DataFrame:
    """Prepare feature matrix for a single building and resource."""
    target_col = RESOURCE_COLS.get(resource)
    if target_col is None:
        raise ValueError(f"Unknown resource: {resource}")
    
    df = hourly_df[hourly_df["building_id"] == building_id].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    if len(df) < 48:  # Need at least 2 days of data
        logger.warning(f"Insufficient data for {building_id}/{resource}: {len(df)} records")
        return None
    
    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
    df["is_peak"] = ((df["hour"] >= 9) & (df["hour"] <= 17)).astype(int)
    df["month"] = df["timestamp"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    # Lag features
    df = create_lag_features(df, target_col)
    
    # Calendar features if available
    calendar_cols = ["avg_temp_c", "rainfall_mm", "humidity_pct", "is_holiday", "is_exam", "has_event"]
    for col in calendar_cols:
        if col not in df.columns:
            df[col] = 0
    
    return df


def get_feature_columns(target_col: str) -> list:
    """Get feature columns for forecasting (excludes target and identifiers)."""
    base_features = [
        "hour", "day_of_week", "is_weekend", "is_night", "is_peak", "month",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "avg_temp_c", "rainfall_mm", "humidity_pct",
        "is_holiday", "is_exam", "has_event",
    ]
    
    lag_features = [f"{target_col}_lag_{l}" for l in [1, 2, 3, 6, 12, 24]]
    rolling_features = []
    for w in [6, 12, 24]:
        rolling_features.extend([
            f"{target_col}_rolling_mean_{w}",
            f"{target_col}_rolling_std_{w}"
        ])
    
    return base_features + lag_features + rolling_features


class ResourceForecaster:
    """Forecasting model for campus resources."""
    
    def __init__(self):
        self.models = {}  # {(building_id, resource): model}
        self.evaluation_results = {}
    
    @timer
    def train_and_evaluate(self, hourly_df: pd.DataFrame, building_id: str,
                           resource: str) -> dict:
        """Train forecasting model for a specific building and resource."""
        target_col = RESOURCE_COLS[resource]
        
        df = prepare_forecast_features(hourly_df, building_id, resource)
        if df is None:
            return {"error": "Insufficient data"}
        
        feature_cols = get_feature_columns(target_col)
        available_features = [c for c in feature_cols if c in df.columns]
        
        # Drop rows with NaN in target or all features NaN
        df = df.dropna(subset=[target_col])
        df = df.dropna(subset=available_features, how="all")
        
        if len(df) < 100:
            return {"error": f"Too few valid records: {len(df)}"}
        
        # Time-based split (80/20)
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        X_train = train_df[available_features]
        y_train = train_df[target_col]
        X_test = test_df[available_features]
        y_test = test_df[target_col]
        
        # Naive baseline: last known value
        y_baseline = test_df[f"{target_col}_lag_1"].fillna(y_train.mean())
        baseline_mae = mean_absolute_error(y_test, y_baseline)
        baseline_rmse = np.sqrt(mean_squared_error(y_test, y_baseline))
        
        # Configure best available regressor (XGBoost prioritized for speed & edge efficiency)
        if HAS_XGBOOST:
            regressor = XGBRegressor(
                n_estimators=150, max_depth=6, learning_rate=0.08,
                random_state=42, n_jobs=-1
            )
        elif HAS_SKLEARN_ENSEMBLE and HistGradientBoostingRegressor is not None:
            regressor = HistGradientBoostingRegressor(
                max_iter=200, max_depth=8, learning_rate=0.1,
                random_state=42, early_stopping=True, validation_fraction=0.1
            )
        else:
            raise RuntimeError("No supported regressor found (XGBoost or scikit-learn).")

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", regressor)
        ])
        
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start_time
        
        y_pred = model.predict(X_test)
        model_mae = mean_absolute_error(y_test, y_pred)
        model_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # WAPE (Weighted Absolute Percentage Error)
        total_actual = y_test.sum()
        wape = safe_divide(np.abs(y_test.values - y_pred).sum(), total_actual) * 100
        
        # Store model
        key = (building_id, resource)
        self.models[key] = {
            "model": model,
            "features": available_features,
            "target_col": target_col,
        }
        
        result = {
            "building_id": building_id,
            "resource": resource,
            "target_column": target_col,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "training_period": {
                "start": str(train_df["timestamp"].min()),
                "end": str(train_df["timestamp"].max()),
            },
            "test_period": {
                "start": str(test_df["timestamp"].min()),
                "end": str(test_df["timestamp"].max()),
            },
            "baseline": {
                "method": "naive_lag_1",
                "mae": round(baseline_mae, 4),
                "rmse": round(baseline_rmse, 4),
            },
            "model": {
                "type": "HistGradientBoostingRegressor",
                "mae": round(model_mae, 4),
                "rmse": round(model_rmse, 4),
                "wape_pct": round(wape, 2),
                "improvement_over_baseline_pct": round(
                    safe_divide(baseline_mae - model_mae, baseline_mae) * 100, 2
                ),
            },
            "training_time_s": round(train_time, 3),
            "trained_at": pd.Timestamp.now().isoformat(),
        }
        
        self.evaluation_results[key] = result
        logger.info(f"Trained {building_id}/{resource}: MAE={model_mae:.2f} (baseline: {baseline_mae:.2f})")
        
        return result
    
    def forecast(self, hourly_df: pd.DataFrame, building_id: str,
                 resource: str, horizon_hours: int = 6) -> pd.DataFrame:
        """Generate forecast for the next N hours."""
        key = (building_id, resource)
        if key not in self.models:
            logger.warning(f"No trained model for {building_id}/{resource}")
            return None
        
        model_info = self.models[key]
        target_col = model_info["target_col"]
        features = model_info["features"]
        model = model_info["model"]
        
        # Get last known data
        df = prepare_forecast_features(hourly_df, building_id, resource)
        if df is None:
            return None
        
        last_ts = df["timestamp"].max()
        
        # Generate future timestamps
        future_ts = pd.date_range(
            start=last_ts + pd.Timedelta(hours=1),
            periods=horizon_hours,
            freq="h"
        )
        
        forecasts = []
        current_df = df.copy()
        
        for ts in future_ts:
            # Create feature row
            row = pd.DataFrame([{
                "timestamp": ts,
                "building_id": building_id,
                target_col: np.nan,
                "hour": ts.hour,
                "day_of_week": ts.dayofweek,
                "is_weekend": int(ts.dayofweek >= 5),
                "is_night": int(ts.hour < 6 or ts.hour >= 22),
                "is_peak": int(9 <= ts.hour <= 17),
                "month": ts.month,
                "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
                "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
                "dow_sin": np.sin(2 * np.pi * ts.dayofweek / 7),
                "dow_cos": np.cos(2 * np.pi * ts.dayofweek / 7),
                "avg_temp_c": df["avg_temp_c"].iloc[-1] if "avg_temp_c" in df.columns else 0,
                "rainfall_mm": 0,
                "humidity_pct": df["humidity_pct"].iloc[-1] if "humidity_pct" in df.columns else 0,
                "is_holiday": 0,
                "is_exam": 0,
                "has_event": 0,
            }])
            
            # Add lag features from current_df
            for lag in [1, 2, 3, 6, 12, 24]:
                col_name = f"{target_col}_lag_{lag}"
                if len(current_df) >= lag:
                    row[col_name] = current_df[target_col].iloc[-lag]
                else:
                    row[col_name] = np.nan
            
            for w in [6, 12, 24]:
                recent = current_df[target_col].tail(w)
                row[f"{target_col}_rolling_mean_{w}"] = recent.mean()
                row[f"{target_col}_rolling_std_{w}"] = recent.std()
            
            # Fill missing feature columns
            for f in features:
                if f not in row.columns:
                    row[f] = 0
            
            X = row[features]
            pred = model.predict(X)[0]
            pred = max(0, pred)  # Non-negative predictions
            
            forecasts.append({
                "timestamp": ts,
                "building_id": building_id,
                "resource": resource,
                "predicted_value": round(pred, 2),
                "model_type": "HistGradientBoostingRegressor",
                "horizon_hours": horizon_hours,
                "is_forecast": True,
            })
            
            # Update current_df with prediction for recursive forecasting
            new_row = row.copy()
            new_row[target_col] = pred
            current_df = pd.concat([current_df, new_row], ignore_index=True)
        
        return pd.DataFrame(forecasts)
    
    def train_all(self, hourly_df: pd.DataFrame) -> dict:
        """Train models for all buildings and resources."""
        results = {}
        buildings = hourly_df["building_id"].unique()
        
        for bid in buildings:
            for resource in RESOURCE_COLS:
                try:
                    result = self.train_and_evaluate(hourly_df, bid, resource)
                    results[f"{bid}_{resource}"] = result
                except Exception as e:
                    logger.error(f"Failed to train {bid}/{resource}: {e}")
                    results[f"{bid}_{resource}"] = {"error": str(e)}
        
        # Save evaluation report
        reports_dir = PROJECT_ROOT / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        with open(reports_dir / "forecast_evaluation.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def save(self, path: Path = None):
        """Save all trained models."""
        if path is None:
            path = PROJECT_ROOT / "models" / "forecasters.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "models": self.models,
            "evaluation_results": self.evaluation_results,
        }, path)
        logger.info(f"Forecasters saved to {path}")
    
    def load(self, path: Path = None) -> bool:
        """Load trained models from disk."""
        if path is None:
            path = PROJECT_ROOT / "models" / "forecasters.joblib"
        if not path.exists():
            return False
        saved = joblib.load(path)
        self.models = saved["models"]
        self.evaluation_results = saved.get("evaluation_results", {})
        logger.info(f"Loaded {len(self.models)} forecasting models")
        return True
