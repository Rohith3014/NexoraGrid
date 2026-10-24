"""
EcoGrid AI - Circular Economy Accounting Module
Calculates resource costs, emissions, recycling rates, and compost output.
"""
import pandas as pd
import numpy as np
from .config import get_campus_constants
from .utils import get_logger, safe_divide

logger = get_logger("circular_accounting")


class CircularAccountant:
    """Transparent circular-economy accounting with documented formulas."""
    
    def __init__(self):
        self.constants = get_campus_constants()
    
    def calculate_campus_metrics(self, daily: pd.DataFrame, buildings: pd.DataFrame = None) -> dict:
        """Calculate campus-wide circular economy metrics."""
        if daily is None or daily.empty:
            return self._empty_metrics()
        
        metrics = {}
        
        # === ELECTRICITY ===
        total_elec = daily["electricity_kwh"].sum()
        tariff = self.constants["electricity_tariff_rs_per_kwh"]
        carbon = self.constants["grid_carbon_factor_kg_per_kwh"]
        
        metrics["electricity"] = {
            "total_kwh": round(float(total_elec), 2),
            "estimated_cost_rs": round(float(total_elec * tariff), 2),
            "estimated_emissions_kg_co2": round(float(total_elec * carbon), 2),
            "daily_avg_kwh": round(float(daily.groupby("date")["electricity_kwh"].sum().mean()), 2),
            "formula": f"Cost = total_kwh × ₹{tariff}/kWh; Emissions = total_kwh × {carbon} kg CO₂/kWh",
            "note": "Emissions are ESTIMATED based on grid carbon factor. Actual emissions depend on real-time grid mix.",
        }
        
        # === SOLAR ===
        total_solar = daily["solar_generation_kwh"].sum()
        metrics["solar"] = {
            "total_generation_kwh": round(float(total_solar), 2),
            "estimated_offset_cost_rs": round(float(total_solar * tariff), 2),
            "estimated_avoided_emissions_kg_co2": round(float(total_solar * carbon), 2),
            "daily_avg_kwh": round(float(daily.groupby("date")["solar_generation_kwh"].sum().mean()), 2),
            "formula": f"Offset cost = solar_kwh × ₹{tariff}/kWh; Avoided emissions = solar_kwh × {carbon} kg CO₂/kWh",
            "note": "Avoided emissions assume solar displaces grid electricity 1:1. This may not hold during export/curtailment.",
        }
        
        # === WATER ===
        total_water = daily["water_litres"].sum()
        water_energy = self.constants["water_energy_intensity_kwh_per_kl"]
        water_energy_total = total_water / 1000 * water_energy
        
        metrics["water"] = {
            "total_litres": round(float(total_water), 2),
            "total_kl": round(float(total_water / 1000), 2),
            "estimated_treatment_energy_kwh": round(float(water_energy_total), 2),
            "daily_avg_litres": round(float(daily.groupby("date")["water_litres"].sum().mean()), 2),
            "formula": f"Treatment energy = total_kL × {water_energy} kWh/kL",
            "note": "Treatment energy is an ESTIMATE of pumping and treatment energy, NOT measured electricity consumption.",
        }
        
        # === WASTE ===
        waste_cols = ["wet_organic_kg", "dry_recyclable_kg", "ewaste_kg", "hazardous_kg"]
        total_waste = {col: round(float(daily[col].sum()), 2) for col in waste_cols}
        total_waste_kg = sum(total_waste.values())
        
        composted = daily["composted_kg"].sum()
        recycled = daily["recycled_kg"].sum()
        landfill = daily["landfill_kg"].sum()
        
        diverted = composted + recycled
        diversion_rate = safe_divide(diverted, total_waste_kg) * 100
        
        compost_ratio = self.constants["composting_ratio"]
        estimated_compost_output = composted / compost_ratio
        
        avg_contamination = daily["contamination_pct"].mean()
        
        metrics["waste"] = {
            "total_waste_kg": round(float(total_waste_kg), 2),
            "by_category": total_waste,
            "composted_kg": round(float(composted), 2),
            "recycled_kg": round(float(recycled), 2),
            "landfill_kg": round(float(landfill), 2),
            "diversion_rate_pct": round(float(diversion_rate), 2),
            "estimated_compost_output_kg": round(float(estimated_compost_output), 2),
            "avg_contamination_pct": round(float(avg_contamination), 2),
            "formula": f"Diversion rate = (composted + recycled) / total_waste × 100; Compost output = composted_kg / {compost_ratio} (4:1 ratio)",
            "note": "Compost output assumes all diverted organic waste is successfully composted at a 4:1 input-to-output ratio. Actual yield depends on composting conditions.",
        }
        
        metrics["summary"] = {
            "monitoring_days": int(daily["date"].nunique()),
            "buildings_monitored": int(daily["building_id"].nunique()),
            "total_records": len(daily),
        }
        
        return metrics
    
    def calculate_building_metrics(self, daily: pd.DataFrame, building_id: str) -> dict:
        """Calculate metrics for a single building."""
        bdata = daily[daily["building_id"] == building_id]
        if bdata.empty:
            return self._empty_metrics()
        return self.calculate_campus_metrics(bdata)
    
    def compare_buildings(self, daily: pd.DataFrame, buildings: pd.DataFrame = None) -> pd.DataFrame:
        """Compare buildings on key circular economy metrics."""
        if daily is None or daily.empty:
            return pd.DataFrame()
        
        comparisons = []
        for bid in daily["building_id"].unique():
            bdata = daily[daily["building_id"] == bid]
            
            total_waste = (bdata["wet_organic_kg"] + bdata["dry_recyclable_kg"] +
                          bdata["ewaste_kg"] + bdata["hazardous_kg"]).sum()
            diverted = bdata["composted_kg"].sum() + bdata["recycled_kg"].sum()
            
            capacity = 1
            if buildings is not None:
                b_row = buildings[buildings["building_id"] == bid]
                if not b_row.empty:
                    capacity = b_row["design_capacity"].iloc[0]
            
            comparisons.append({
                "building_id": bid,
                "avg_daily_electricity_kwh": round(float(bdata["electricity_kwh"].mean()), 1),
                "avg_daily_water_litres": round(float(bdata["water_litres"].mean()), 1),
                "avg_daily_solar_kwh": round(float(bdata["solar_generation_kwh"].mean()), 1),
                "total_waste_kg": round(float(total_waste), 1),
                "diversion_rate_pct": round(safe_divide(diverted, total_waste) * 100, 1),
                "avg_contamination_pct": round(float(bdata["contamination_pct"].mean()), 1),
                "electricity_per_capita": round(safe_divide(bdata["electricity_kwh"].mean(), capacity), 2),
                "water_per_capita": round(safe_divide(bdata["water_litres"].mean(), capacity), 2),
            })
        
        return pd.DataFrame(comparisons)
    
    def _empty_metrics(self) -> dict:
        return {
            "electricity": {"total_kwh": 0, "estimated_cost_rs": 0, "estimated_emissions_kg_co2": 0},
            "solar": {"total_generation_kwh": 0},
            "water": {"total_litres": 0},
            "waste": {"total_waste_kg": 0, "diversion_rate_pct": 0},
            "summary": {"monitoring_days": 0, "buildings_monitored": 0},
        }
