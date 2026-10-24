"""
EcoGrid AI - Root Cause Analysis Module
Evidence-based explanation for detected anomalies.
"""
import pandas as pd
import numpy as np
from .utils import get_logger

logger = get_logger("root_cause")


class RootCauseAnalyzer:
    """Generates evidence-based explanations for anomalies."""
    
    def analyze(self, alert: dict, data: dict) -> dict:
        """Analyze an anomaly and produce ranked explanations."""
        building_id = alert.get("building_id", "")
        resource = alert.get("resource", "")
        ts_str = alert.get("detection_timestamp", "")
        
        try:
            ts = pd.to_datetime(ts_str)
        except Exception:
            ts = pd.Timestamp.now()
        
        explanations = []
        evidence_items = []
        
        # 1. Check occupancy patterns
        occupancy_check = self._check_occupancy(data, building_id, ts)
        if occupancy_check:
            explanations.append(occupancy_check)
            evidence_items.append(occupancy_check["evidence"])
        
        # 2. Check weather
        weather_check = self._check_weather(data, building_id, ts)
        if weather_check:
            explanations.append(weather_check)
            evidence_items.append(weather_check["evidence"])
        
        # 3. Check maintenance logs
        maintenance_check = self._check_maintenance(data, building_id, ts)
        if maintenance_check:
            explanations.append(maintenance_check)
            evidence_items.append(maintenance_check["evidence"])
        
        # 4. Check calendar events
        event_check = self._check_events(data, ts)
        if event_check:
            explanations.append(event_check)
            evidence_items.append(event_check["evidence"])
        
        # 5. Check sensor health
        sensor_check = self._check_sensor_health(data, building_id, ts, resource)
        if sensor_check:
            explanations.append(sensor_check)
            evidence_items.append(sensor_check["evidence"])
        
        # 6. Check cross-resource correlations
        cross_check = self._check_cross_resource(data, building_id, ts, resource)
        if cross_check:
            explanations.append(cross_check)
            evidence_items.append(cross_check["evidence"])
        
        # Rank explanations by plausibility score
        explanations.sort(key=lambda x: x.get("plausibility", 0), reverse=True)
        
        # Generate recommendation
        if explanations:
            top = explanations[0]
            recommendation = top.get("suggested_action", "Investigate further with on-site inspection.")
        else:
            recommendation = "Cause undetermined. Recommend on-site inspection and sensor verification. Additional data from maintenance team and occupancy records would help narrow the cause."
        
        return {
            "alert_id": alert.get("alert_id", ""),
            "building_id": building_id,
            "resource": resource,
            "ranked_explanations": explanations,
            "evidence_summary": evidence_items,
            "recommended_action": recommendation,
            "cause_determined": len(explanations) > 0,
            "confidence": explanations[0]["plausibility"] if explanations else 0,
        }
    
    def _check_occupancy(self, data: dict, building_id: str, ts: pd.Timestamp) -> dict:
        """Check if occupancy explains the anomaly."""
        daily = data.get("daily_summary")
        if daily is None:
            return None
        
        date = ts.date()
        bday = daily[(daily["building_id"] == building_id) &
                     (daily["date"].dt.date == date)]
        
        if bday.empty or "occupancy_avg" not in bday.columns:
            return None
        
        occ = bday["occupancy_avg"].iloc[0]
        avg_occ = daily[daily["building_id"] == building_id]["occupancy_avg"].mean()
        
        if pd.isna(occ) or pd.isna(avg_occ) or avg_occ == 0:
            return None
        
        deviation = (occ - avg_occ) / avg_occ * 100
        
        if abs(deviation) > 20:
            return {
                "factor": "occupancy",
                "evidence": f"Occupancy on {date} was {occ:.0f} (avg: {avg_occ:.0f}, deviation: {deviation:+.1f}%)",
                "plausibility": min(0.8, abs(deviation) / 100),
                "suggested_action": "Verify if the anomaly correlates with the unusual occupancy level.",
            }
        return None
    
    def _check_weather(self, data: dict, building_id: str, ts: pd.Timestamp) -> dict:
        """Check weather conditions."""
        calendar = data.get("calendar")
        if calendar is None:
            return None
        
        date = ts.date()
        day = calendar[calendar["date"].dt.date == date]
        
        if day.empty:
            return None
        
        row = day.iloc[0]
        issues = []
        plausibility = 0
        
        if row.get("avg_temp_c", 0) > 35:
            issues.append(f"High temperature ({row['avg_temp_c']:.1f}°C) may increase HVAC load")
            plausibility = max(plausibility, 0.6)
        elif row.get("avg_temp_c", 0) < 15:
            issues.append(f"Low temperature ({row['avg_temp_c']:.1f}°C) may increase heating")
            plausibility = max(plausibility, 0.5)
        
        if row.get("rainfall_mm", 0) > 20:
            issues.append(f"Heavy rainfall ({row['rainfall_mm']:.1f}mm) may affect solar and water systems")
            plausibility = max(plausibility, 0.7)
        
        if issues:
            return {
                "factor": "weather",
                "evidence": "; ".join(issues),
                "plausibility": plausibility,
                "suggested_action": "Cross-reference weather impact on resource consumption patterns.",
            }
        return None
    
    def _check_maintenance(self, data: dict, building_id: str, ts: pd.Timestamp) -> dict:
        """Check maintenance logs for related events."""
        logs = data.get("manual_logs")
        if logs is None:
            return None
        
        # Look for logs within ±2 days
        date = ts.date()
        nearby = logs[
            (logs["building_id"] == building_id) &
            (abs((logs["logged_at"].dt.date - date).apply(lambda x: x.days)) <= 2)
        ]
        
        if nearby.empty:
            return None
        
        notes = nearby[nearby["stream"] == "note"]
        if not notes.empty:
            note_texts = notes["note"].dropna().tolist()[:3]
            return {
                "factor": "maintenance",
                "evidence": f"Maintenance log entries near this time: {'; '.join(str(n) for n in note_texts)}",
                "plausibility": 0.7,
                "suggested_action": "Review maintenance activities for correlation with the anomaly.",
            }
        return None
    
    def _check_events(self, data: dict, ts: pd.Timestamp) -> dict:
        """Check calendar for campus events."""
        calendar = data.get("calendar")
        if calendar is None:
            return None
        
        date = ts.date()
        day = calendar[calendar["date"].dt.date == date]
        
        if day.empty:
            return None
        
        row = day.iloc[0]
        if pd.notna(row.get("event")):
            return {
                "factor": "campus_event",
                "evidence": f"Campus event on {date}: {row['event']}",
                "plausibility": 0.65,
                "suggested_action": "Events can cause atypical resource consumption. Verify if consumption returns to normal after the event.",
            }
        
        if row.get("day_type") in ["exam", "holiday", "semester_break"]:
            return {
                "factor": "schedule",
                "evidence": f"Day type: {row['day_type']} — expected consumption patterns differ from normal working days",
                "plausibility": 0.5,
                "suggested_action": "Compare with other days of the same type to verify if the reading is abnormal for this context.",
            }
        return None
    
    def _check_sensor_health(self, data: dict, building_id: str,
                              ts: pd.Timestamp, resource: str) -> dict:
        """Check for sensor data quality issues."""
        hourly = data.get("hourly_streams")
        if hourly is None:
            return None
        
        bdf = hourly[hourly["building_id"] == building_id].copy()
        
        # Check for gaps
        ts_sorted = bdf["timestamp"].sort_values()
        if len(ts_sorted) < 2:
            return None
        
        gaps = ts_sorted.diff().dt.total_seconds() / 3600  # Hours
        large_gaps = gaps[gaps > 2]
        
        if len(large_gaps) > len(gaps) * 0.05:  # >5% gaps
            return {
                "factor": "sensor_health",
                "evidence": f"Detected {len(large_gaps)} data gaps (>2 hours) for {building_id}. Sensor or connectivity issues may affect readings.",
                "plausibility": 0.4,
                "suggested_action": "Check sensor connectivity and gateway health. Missing data periods may lead to inaccurate aggregations.",
            }
        return None
    
    def _check_cross_resource(self, data: dict, building_id: str,
                               ts: pd.Timestamp, resource: str) -> dict:
        """Check if other resources also show anomalies."""
        daily = data.get("daily_summary")
        if daily is None:
            return None
        
        date = ts.date()
        bday = daily[(daily["building_id"] == building_id) &
                     (daily["date"].dt.date == date)]
        
        if bday.empty:
            return None
        
        row = bday.iloc[0]
        bavg = daily[daily["building_id"] == building_id].mean(numeric_only=True)
        
        anomalous_resources = []
        for col, name in [("electricity_kwh", "electricity"), ("water_litres", "water"),
                         ("solar_generation_kwh", "solar")]:
            if name == resource or col not in row.index:
                continue
            val = row.get(col, 0)
            avg = bavg.get(col, 0)
            if avg > 0 and abs(val - avg) / avg > 0.3:
                anomalous_resources.append(f"{name} ({val:.1f} vs avg {avg:.1f})")
        
        if anomalous_resources:
            return {
                "factor": "cross_resource",
                "evidence": f"Other resources also showing unusual patterns: {', '.join(anomalous_resources)}",
                "plausibility": 0.55,
                "suggested_action": "Multiple resource anomalies may indicate a building-wide issue (power outage, special activity, or meter problem).",
            }
        return None
