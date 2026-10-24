"""
EcoGrid AI - Anomaly Detection Module
Detects resource anomalies using statistical and ML methods.
"""
import numpy as np
import pandas as pd
try:
    from sklearn.ensemble import IsolationForest
    HAS_ISOLATION_FOREST = True
except Exception:
    HAS_ISOLATION_FOREST = False
    IsolationForest = None

from .config import get_config
from .utils import get_logger, generate_stable_id
import uuid

logger = get_logger("anomaly_detection")


class AnomalyDetector:
    """Multi-method anomaly detection for campus resources."""
    
    def __init__(self):
        config = get_config()
        self.zscore_threshold = config["anomaly_detection"]["zscore_threshold"]
        self.mad_threshold = config["anomaly_detection"]["mad_threshold"]
        self.contamination = config["anomaly_detection"]["isolation_forest_contamination"]
        self.alerts = []
    
    def detect_statistical_anomalies(self, df: pd.DataFrame, building_id: str,
                                      resource_col: str, resource_name: str,
                                      window: int = 24) -> list:
        """Detect anomalies using rolling stats and MAD."""
        alerts = []
        bdf = df[df["building_id"] == building_id].sort_values("timestamp").copy()
        
        if len(bdf) < window * 2:
            return alerts
        
        values = bdf[resource_col].dropna()
        if len(values) < window:
            return alerts
        
        # Rolling median and MAD
        rolling_median = values.rolling(window=window, min_periods=max(1, window // 2)).median()
        rolling_mad = values.rolling(window=window, min_periods=max(1, window // 2)).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True
        )
        
        # Prevent division by zero
        rolling_mad = rolling_mad.replace(0, values.std() * 0.1 if values.std() > 0 else 1)
        
        modified_zscore = 0.6745 * (values - rolling_median) / rolling_mad
        
        for idx in values.index:
            if idx not in modified_zscore.index:
                continue
            score = abs(modified_zscore.loc[idx])
            if score > self.mad_threshold:
                row = bdf.loc[idx] if idx in bdf.index else None
                if row is None:
                    continue
                
                observed = values.loc[idx]
                expected = rolling_median.loc[idx] if idx in rolling_median.index else values.mean()
                deviation_pct = ((observed - expected) / expected * 100) if expected != 0 else 0
                
                severity = "low"
                if score > self.mad_threshold * 2:
                    severity = "high"
                elif score > self.mad_threshold * 1.5:
                    severity = "medium"
                
                ts = row["timestamp"] if "timestamp" in bdf.columns else pd.Timestamp.now()
                
                alert = {
                    "alert_id": f"ANM-{generate_stable_id(building_id, resource_name, str(ts))}",
                    "building_id": building_id,
                    "resource": resource_name,
                    "detection_timestamp": str(ts),
                    "observed_value": round(float(observed), 2),
                    "expected_value": round(float(expected), 2),
                    "deviation_pct": round(float(deviation_pct), 1),
                    "zscore": round(float(score), 2),
                    "severity": severity,
                    "method": "rolling_mad",
                    "evidence": f"{resource_name} reading of {observed:.1f} is {abs(deviation_pct):.1f}% {'above' if deviation_pct > 0 else 'below'} the rolling median of {expected:.1f}",
                    "recommended_investigation": self._get_investigation_suggestion(resource_name, deviation_pct),
                    "status": "active",
                }
                alerts.append(alert)
        
        return alerts
    
    def detect_overnight_anomalies(self, hourly_df: pd.DataFrame, building_id: str) -> list:
        """Detect unusually high overnight consumption."""
        alerts = []
        bdf = hourly_df[hourly_df["building_id"] == building_id].copy()
        bdf["hour"] = bdf["timestamp"].dt.hour
        
        night_hours = bdf[bdf["hour"].between(0, 5)]
        day_hours = bdf[bdf["hour"].between(8, 18)]
        
        for resource_col, resource_name in [("electricity_kwh", "electricity"), ("water_litres", "water")]:
            if resource_col not in bdf.columns:
                continue
            
            night_mean = night_hours[resource_col].mean()
            day_mean = day_hours[resource_col].mean()
            
            if pd.isna(night_mean) or pd.isna(day_mean) or day_mean == 0:
                continue
            
            night_ratio = night_mean / day_mean
            
            if night_ratio > 0.5:  # Night > 50% of day consumption
                alerts.append({
                    "alert_id": f"ANM-NIGHT-{generate_stable_id(building_id, resource_name)}",
                    "building_id": building_id,
                    "resource": resource_name,
                    "detection_timestamp": str(pd.Timestamp.now()),
                    "observed_value": round(float(night_mean), 2),
                    "expected_value": round(float(day_mean * 0.3), 2),
                    "deviation_pct": round(float((night_ratio - 0.3) / 0.3 * 100), 1),
                    "severity": "medium" if night_ratio < 0.7 else "high",
                    "method": "overnight_ratio",
                    "evidence": f"Average overnight {resource_name} ({night_mean:.1f}) is {night_ratio*100:.0f}% of daytime average ({day_mean:.1f}). This may indicate possible equipment left running or a potential leak.",
                    "recommended_investigation": f"Inspect {resource_name} meters and equipment during overnight hours (00:00-06:00). Check for automated systems, leaks, or equipment that should be on scheduled shutdown.",
                    "status": "active",
                })
        
        return alerts
    
    def detect_isolation_forest(self, df: pd.DataFrame, building_id: str,
                                 features: list = None) -> list:
        """Use Isolation Forest for multivariate anomaly detection."""
        alerts = []
        bdf = df[df["building_id"] == building_id].copy()
        
        if features is None:
            features = [c for c in ["electricity_kwh", "water_litres", "solar_generation_kwh", "footfall"]
                       if c in bdf.columns]
        
        valid_features = [c for c in features if c in bdf.columns]
        if len(valid_features) < 2:
            return alerts
        
        X = bdf[valid_features].dropna()
        if len(X) < 50:
            return alerts
        
        if HAS_ISOLATION_FOREST and IsolationForest is not None:
            iso = IsolationForest(contamination=self.contamination, random_state=42)
            predictions = iso.fit_predict(X)
            anomaly_mask = predictions == -1
            anomaly_indices = X[anomaly_mask].index
            method_used = "isolation_forest"
        else:
            # Multivariate robust Mahalanobis/Z-score combination
            norm_X = (X - X.median()) / (X.std().replace(0, 1))
            multivar_score = np.sqrt((norm_X ** 2).sum(axis=1))
            thresh = np.percentile(multivar_score, 100 * (1 - self.contamination))
            anomaly_mask = multivar_score >= thresh
            anomaly_indices = X[anomaly_mask].index
            method_used = "robust_multivariate"
        
        for idx in anomaly_indices[:20]:  # Limit to top 20
            row = bdf.loc[idx]
            ts = row.get("timestamp", pd.Timestamp.now())
            
            alerts.append({
                "alert_id": f"ANM-IF-{generate_stable_id(building_id, str(ts))}",
                "building_id": building_id,
                "resource": "multivariate",
                "detection_timestamp": str(ts),
                "observed_value": {c: round(float(row[c]), 2) for c in valid_features if pd.notna(row.get(c))},
                "expected_value": {c: round(float(X[c].median()), 2) for c in valid_features},
                "deviation_pct": 0,
                "severity": "medium",
                "method": method_used,
                "evidence": f"Multivariate anomaly detected. Readings deviate from normal patterns across {', '.join(valid_features)}.",
                "recommended_investigation": "Review all resource readings for this time period. Check for concurrent equipment issues or unusual building activity.",
                "status": "active",
            })
        
        return alerts
    
    def detect_all(self, hourly_df: pd.DataFrame) -> list:
        """Run all anomaly detection methods across all buildings."""
        all_alerts = []
        buildings = hourly_df["building_id"].unique()
        
        for bid in buildings:
            # Statistical anomalies for each resource
            for col, name in [("electricity_kwh", "electricity"),
                             ("water_litres", "water"),
                             ("solar_generation_kwh", "solar")]:
                if col in hourly_df.columns:
                    try:
                        alerts = self.detect_statistical_anomalies(
                            hourly_df, bid, col, name, window=24
                        )
                        # Keep only top anomalies per building/resource
                        alerts.sort(key=lambda x: abs(x.get("deviation_pct", 0)), reverse=True)
                        all_alerts.extend(alerts[:5])
                    except Exception as e:
                        logger.error(f"Statistical anomaly detection failed for {bid}/{name}: {e}")
            
            # Overnight anomalies
            try:
                overnight = self.detect_overnight_anomalies(hourly_df, bid)
                all_alerts.extend(overnight)
            except Exception as e:
                logger.error(f"Overnight anomaly detection failed for {bid}: {e}")
            
            # Isolation Forest
            try:
                iso_alerts = self.detect_isolation_forest(hourly_df, bid)
                all_alerts.extend(iso_alerts[:3])
            except Exception as e:
                logger.error(f"Isolation Forest failed for {bid}: {e}")
        
        self.alerts = all_alerts
        logger.info(f"Detected {len(all_alerts)} anomalies across {len(buildings)} buildings")
        return all_alerts
    
    def _get_investigation_suggestion(self, resource: str, deviation_pct: float) -> str:
        """Generate investigation suggestion based on resource and deviation."""
        direction = "high" if deviation_pct > 0 else "low"
        
        suggestions = {
            "electricity": {
                "high": "Inspect electrical panels, HVAC systems, and lighting schedules. Check for equipment malfunction or unusual usage patterns.",
                "low": "Verify meter readings. Check for power outages or load shedding. Confirm occupancy data.",
            },
            "water": {
                "high": "Inspect water meters, check for possible leaks in plumbing. Review maintenance logs for ongoing work. Check storage tank levels.",
                "low": "Verify meter readings. Check for supply interruptions or pump failures.",
            },
            "solar": {
                "high": "Verify inverter readings. Check for unusual grid conditions.",
                "low": "Check for panel soiling, shading, or inverter issues. Review weather conditions for cloud cover.",
            },
        }
        
        return suggestions.get(resource, {}).get(direction,
            f"Investigate unusual {resource} reading. Cross-reference with maintenance logs and occupancy data.")
