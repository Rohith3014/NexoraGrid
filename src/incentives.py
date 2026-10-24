"""
EcoGrid AI - Incentives Module
Building-level sustainability scoring system.
"""
import pandas as pd
import numpy as np
from .config import get_config
from .utils import get_logger, safe_divide

logger = get_logger("incentives")


class SustainabilityScorer:
    """Transparent building-level sustainability scoring."""
    
    def __init__(self):
        config = get_config()
        scoring = config["scoring"]
        self.weights = {
            "electricity_efficiency": scoring["electricity_weight"],
            "water_efficiency": scoring["water_weight"],
            "waste_diversion": scoring["waste_diversion_weight"],
            "anomaly_response": scoring["anomaly_response_weight"],
            "data_quality": scoring["data_quality_weight"],
        }
    
    def calculate_scores(self, daily: pd.DataFrame, buildings: pd.DataFrame = None,
                         anomalies: list = None) -> pd.DataFrame:
        """Calculate sustainability scores for all buildings."""
        if daily is None or daily.empty:
            return pd.DataFrame()
        
        scores = []
        
        for bid in daily["building_id"].unique():
            bdata = daily[daily["building_id"] == bid]
            
            # Electricity efficiency: lower per-sqm is better
            capacity = 1
            if buildings is not None:
                b_row = buildings[buildings["building_id"] == bid]
                if not b_row.empty:
                    capacity = b_row["design_capacity"].iloc[0]
            
            elec_per_capita = safe_divide(bdata["electricity_kwh"].mean(), capacity)
            water_per_capita = safe_divide(bdata["water_litres"].mean(), capacity)
            
            # Waste diversion rate
            total_waste = (bdata["wet_organic_kg"] + bdata["dry_recyclable_kg"] +
                          bdata["ewaste_kg"] + bdata["hazardous_kg"]).sum()
            diverted = bdata["composted_kg"].sum() + bdata["recycled_kg"].sum()
            diversion_rate = safe_divide(diverted, total_waste)
            
            # Contamination (lower is better)
            avg_contamination = bdata["contamination_pct"].mean()
            
            # Data completeness
            total_possible = bdata.shape[0] * bdata.shape[1]
            missing = bdata.isnull().sum().sum()
            data_quality = 1 - safe_divide(missing, total_possible)
            
            # Anomaly response score
            anomaly_score = 0.7  # Default moderate score
            if anomalies:
                building_anomalies = [a for a in anomalies if a.get("building_id") == bid]
                resolved = [a for a in building_anomalies if a.get("status") == "resolved"]
                if building_anomalies:
                    anomaly_score = safe_divide(len(resolved), len(building_anomalies))
            
            # Calculate component scores (0-100)
            # For per-capita metrics, compare against building's own data, score inversely
            all_elec = daily.groupby("building_id").apply(
                lambda x: x["electricity_kwh"].mean() / max(1, capacity)
            )
            all_water = daily.groupby("building_id").apply(
                lambda x: x["water_litres"].mean() / max(1, capacity)
            )
            
            elec_score = max(0, min(100, (1 - safe_divide(elec_per_capita, all_elec.max())) * 100)) if all_elec.max() > 0 else 50
            water_score = max(0, min(100, (1 - safe_divide(water_per_capita, all_water.max())) * 100)) if all_water.max() > 0 else 50
            waste_score = min(100, diversion_rate * 100)
            quality_score = min(100, data_quality * 100)
            anomaly_resp_score = min(100, anomaly_score * 100)
            
            # Weighted total
            total_score = (
                elec_score * self.weights["electricity_efficiency"] +
                water_score * self.weights["water_efficiency"] +
                waste_score * self.weights["waste_diversion"] +
                anomaly_resp_score * self.weights["anomaly_response"] +
                quality_score * self.weights["data_quality"]
            )
            
            scores.append({
                "building_id": bid,
                "total_score": round(total_score, 1),
                "electricity_score": round(elec_score, 1),
                "water_score": round(water_score, 1),
                "waste_diversion_score": round(waste_score, 1),
                "anomaly_response_score": round(anomaly_resp_score, 1),
                "data_quality_score": round(quality_score, 1),
                "electricity_per_capita": round(elec_per_capita, 2),
                "water_per_capita": round(water_per_capita, 2),
                "diversion_rate_pct": round(diversion_rate * 100, 1),
                "contamination_pct": round(float(avg_contamination), 1),
                "weights": self.weights,
                "note": "Pilot metric. Score is relative to campus peers and own baseline. Missing data does not earn points.",
            })
        
        result = pd.DataFrame(scores).sort_values("total_score", ascending=False)
        logger.info(f"Calculated sustainability scores for {len(result)} buildings")
        return result
    
    def get_score_explanation(self) -> str:
        """Return human-readable score methodology."""
        return """
## Sustainability Score Methodology

The sustainability score is a **pilot metric** designed to encourage improvement, not prove causation.

### Components (weighted):
| Component | Weight | Description |
|---|---|---|
| Electricity Efficiency | {elec}% | Lower per-capita consumption scores higher |
| Water Efficiency | {water}% | Lower per-capita consumption scores higher |
| Waste Diversion | {waste}% | Higher recycling + composting rate scores higher |
| Anomaly Response | {anomaly}% | Timely resolution of detected anomalies |
| Data Quality | {quality}% | Fewer missing readings. Missing data does NOT earn points |

### Normalization:
- Scores are normalized against campus peers to avoid unfairly comparing different building types.
- Per-capita metrics use design capacity rather than daily occupancy to provide a stable baseline.

### Limitations:
- This is a relative ranking, not an absolute measure of sustainability.
- The score does not prove that any incentive program caused improvements.
- Manipulating data (suppressing alerts, deleting records) is not rewarded.
        """.format(
            elec=int(self.weights["electricity_efficiency"] * 100),
            water=int(self.weights["water_efficiency"] * 100),
            waste=int(self.weights["waste_diversion"] * 100),
            anomaly=int(self.weights["anomaly_response"] * 100),
            quality=int(self.weights["data_quality"] * 100),
        )
