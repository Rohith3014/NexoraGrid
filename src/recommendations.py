"""
EcoGrid AI - Recommendations Module
Evidence-based resource-saving recommendations.
"""
import pandas as pd
import numpy as np
from .config import get_campus_constants
from .utils import get_logger, safe_divide, generate_stable_id

logger = get_logger("recommendations")


class RecommendationEngine:
    """Generates actionable resource-saving recommendations."""
    
    def __init__(self):
        self.recommendations = []
        self.constants = get_campus_constants()
    
    def generate_all(self, data: dict, anomalies: list = None) -> list:
        """Generate all recommendations from data analysis."""
        self.recommendations = []
        
        daily = data.get("daily_summary")
        buildings = data.get("buildings")
        hourly = data.get("hourly_streams")
        
        if daily is not None:
            self._check_electricity_efficiency(daily, buildings)
            self._check_water_efficiency(daily, buildings)
            self._check_waste_segregation(daily)
            self._check_solar_performance(daily, buildings)
            self._check_recycling_opportunities(daily)
        
        if hourly is not None:
            self._check_off_hours_consumption(hourly)
        
        if anomalies:
            self._recommendations_from_anomalies(anomalies)
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        self.recommendations.sort(key=lambda r: priority_order.get(r.get("priority", "low"), 3))
        
        logger.info(f"Generated {len(self.recommendations)} recommendations")
        return self.recommendations
    
    def _add_recommendation(self, **kwargs):
        """Add a recommendation with default fields."""
        rec = {
            "rec_id": f"REC-{generate_stable_id(kwargs.get('title', ''), kwargs.get('building', ''))}",
            "title": "",
            "building": "",
            "evidence": "",
            "suggested_action": "",
            "expected_impact": "",
            "confidence": "medium",
            "priority": "medium",
            "calculation_method": "",
            "verification_status": "pending",
            "user_feedback": None,
        }
        rec.update(kwargs)
        self.recommendations.append(rec)
    
    def _check_electricity_efficiency(self, daily: pd.DataFrame, buildings: pd.DataFrame):
        """Check electricity consumption per building for efficiency opportunities."""
        if buildings is None:
            return
        merged = daily.merge(buildings[["building_id", "area_sqm", "design_capacity"]], on="building_id")
        
        for bid in merged["building_id"].unique():
            bdata = merged[merged["building_id"] == bid]
            avg_elec = bdata["electricity_kwh"].mean()
            area = bdata["area_sqm"].iloc[0]
            
            intensity = safe_divide(avg_elec, area) * 1000  # Wh/sqm/day
            
            if intensity > 100:  # High intensity threshold
                tariff = self.constants["electricity_tariff_rs_per_kwh"]
                saving_potential = (avg_elec * 0.1)  # 10% reduction target
                cost_saving = saving_potential * tariff
                
                self._add_recommendation(
                    title="High electricity intensity detected",
                    building=bid,
                    evidence=f"Average daily consumption: {avg_elec:.0f} kWh ({intensity:.1f} Wh/sqm/day). This is above the efficient benchmark of 100 Wh/sqm/day.",
                    suggested_action="Audit HVAC schedules, lighting, and equipment. Consider installing occupancy-based controls and LED upgrades.",
                    expected_impact=f"Estimated 10% reduction could save ~{saving_potential:.0f} kWh/day (₹{cost_saving:.0f}/day). This is an estimate based on typical efficiency gains.",
                    confidence="medium",
                    priority="high" if intensity > 150 else "medium",
                    calculation_method=f"Intensity = avg_daily_kwh / area_sqm * 1000. Saving = 10% of avg consumption × ₹{tariff}/kWh.",
                )
    
    def _check_water_efficiency(self, daily: pd.DataFrame, buildings: pd.DataFrame):
        """Check water consumption efficiency."""
        if buildings is None:
            return
        merged = daily.merge(buildings[["building_id", "design_capacity"]], on="building_id")
        
        for bid in merged["building_id"].unique():
            bdata = merged[merged["building_id"] == bid]
            avg_water = bdata["water_litres"].mean()
            capacity = bdata["design_capacity"].iloc[0]
            
            per_capita = safe_divide(avg_water, capacity)
            
            if per_capita > 100:  # > 100 L/person/day
                self._add_recommendation(
                    title="High per-capita water consumption",
                    building=bid,
                    evidence=f"Average daily water: {avg_water:.0f} L ({per_capita:.0f} L/person at design capacity {capacity}). Above typical benchmark of 100 L/person/day.",
                    suggested_action="Inspect for leaks, install flow restrictors, and review irrigation schedules. Check for overnight consumption anomalies.",
                    expected_impact=f"Reducing to 100 L/person/day could save ~{(avg_water - capacity * 100):.0f} L/day.",
                    confidence="medium",
                    priority="medium",
                    calculation_method="Per-capita = avg_daily_litres / design_capacity.",
                )
    
    def _check_waste_segregation(self, daily: pd.DataFrame):
        """Check waste segregation quality."""
        for bid in daily["building_id"].unique():
            bdata = daily[daily["building_id"] == bid]
            avg_contamination = bdata["contamination_pct"].mean()
            
            if avg_contamination > 10:
                self._add_recommendation(
                    title="High waste contamination rate",
                    building=bid,
                    evidence=f"Average contamination in recycling stream: {avg_contamination:.1f}%. Target: <10%.",
                    suggested_action="Improve signage at waste bins. Conduct waste segregation awareness campaign. Consider dedicated sorting staff during peak hours.",
                    expected_impact=f"Reducing contamination to 10% could increase effective recycling by {(avg_contamination - 10):.1f}% of current recyclable volume.",
                    confidence="medium",
                    priority="medium",
                    calculation_method="Average contamination_pct from daily_summary.",
                )
    
    def _check_solar_performance(self, daily: pd.DataFrame, buildings: pd.DataFrame):
        """Check solar generation against capacity."""
        if buildings is None:
            return
        solar_buildings = buildings[buildings["has_solar"] == 1]
        
        for _, brow in solar_buildings.iterrows():
            bid = brow["building_id"]
            capacity_kw = brow["solar_capacity_kw"]
            
            bdata = daily[daily["building_id"] == bid]
            avg_solar = bdata["solar_generation_kwh"].mean()
            
            # Expected ~4-5 peak hours for India
            expected_daily = capacity_kw * 4.5
            ratio = safe_divide(avg_solar, expected_daily)
            
            if ratio < 0.7 and expected_daily > 0:
                self._add_recommendation(
                    title="Below-expected solar generation",
                    building=bid,
                    evidence=f"Average solar generation: {avg_solar:.0f} kWh/day. Expected: ~{expected_daily:.0f} kWh/day ({ratio*100:.0f}% of potential).",
                    suggested_action="Inspect panels for soiling or damage. Check inverter status. Review shading from new construction or vegetation.",
                    expected_impact=f"Restoring to expected output could increase generation by ~{(expected_daily - avg_solar):.0f} kWh/day.",
                    confidence="low",
                    priority="medium",
                    calculation_method=f"Expected = capacity_kw ({capacity_kw}) × 4.5 peak hours. Actual/Expected ratio.",
                )
    
    def _check_recycling_opportunities(self, daily: pd.DataFrame):
        """Identify buildings with low recycling rates."""
        for bid in daily["building_id"].unique():
            bdata = daily[daily["building_id"] == bid]
            
            total_waste = (bdata["wet_organic_kg"] + bdata["dry_recyclable_kg"] +
                          bdata["ewaste_kg"] + bdata["hazardous_kg"]).mean()
            recycled = bdata["recycled_kg"].mean()
            composted = bdata["composted_kg"].mean()
            
            diverted = recycled + composted
            diversion_rate = safe_divide(diverted, total_waste) * 100
            
            if diversion_rate < 40 and total_waste > 0:
                self._add_recommendation(
                    title="Low waste diversion rate",
                    building=bid,
                    evidence=f"Waste diversion rate: {diversion_rate:.1f}% (recycled: {recycled:.1f} kg, composted: {composted:.1f} kg, total: {total_waste:.1f} kg/day avg). Target: >50%.",
                    suggested_action="Increase organic waste collection for composting. Improve dry recyclable separation. Engage building occupants in waste reduction.",
                    expected_impact=f"Increasing diversion to 50% could divert additional ~{(total_waste * 0.5 - diverted):.1f} kg/day from landfill.",
                    confidence="medium",
                    priority="medium",
                    calculation_method="Diversion rate = (recycled_kg + composted_kg) / total_waste × 100.",
                )
    
    def _check_off_hours_consumption(self, hourly: pd.DataFrame):
        """Check for consumption during off-hours."""
        hourly = hourly.copy()
        hourly["hour"] = hourly["timestamp"].dt.hour
        
        for bid in hourly["building_id"].unique():
            bdata = hourly[hourly["building_id"] == bid]
            
            off_hours = bdata[bdata["hour"].between(22, 23) | bdata["hour"].between(0, 5)]
            peak_hours = bdata[bdata["hour"].between(9, 17)]
            
            for col, name in [("electricity_kwh", "electricity")]:
                if col not in bdata.columns:
                    continue
                
                off_mean = off_hours[col].mean()
                peak_mean = peak_hours[col].mean()
                
                if pd.isna(off_mean) or pd.isna(peak_mean) or peak_mean == 0:
                    continue
                
                ratio = off_mean / peak_mean
                
                if ratio > 0.4:
                    tariff = self.constants["electricity_tariff_rs_per_kwh"]
                    potential_saving = off_mean * 0.3 * 8  # 30% reduction, 8 off-hours
                    
                    self._add_recommendation(
                        title=f"High off-hours {name} consumption",
                        building=bid,
                        evidence=f"Off-hours (22:00-06:00) average: {off_mean:.1f} kWh/hr, which is {ratio*100:.0f}% of peak-hours average ({peak_mean:.1f} kWh/hr).",
                        suggested_action=f"Review {name} schedules for off-hours. Check HVAC, lighting, and equipment shutdown procedures.",
                        expected_impact=f"30% reduction in off-hours could save ~{potential_saving:.0f} kWh/night (₹{potential_saving*tariff:.0f}/night). This is an estimate.",
                        confidence="medium",
                        priority="medium",
                        calculation_method=f"Off-hours ratio = off_hours_mean / peak_hours_mean. Saving = 30% × off_mean × 8 hours × ₹{tariff}.",
                    )
    
    def _recommendations_from_anomalies(self, anomalies: list):
        """Generate recommendations from detected anomalies."""
        high_severity = [a for a in anomalies if a.get("severity") == "high"]
        
        for alert in high_severity[:5]:
            self._add_recommendation(
                title=f"Investigate {alert.get('resource', 'resource')} anomaly",
                building=alert.get("building_id", ""),
                evidence=alert.get("evidence", "Anomaly detected"),
                suggested_action=alert.get("recommended_investigation", "Investigate on site."),
                expected_impact="Preventing resource waste from undetected issues.",
                confidence="high",
                priority="high",
                calculation_method="Based on anomaly detection flagging.",
            )
