"""
EcoGrid AI - Sensor Fusion Module
Time-aware joining of async data streams with building-safe merges.
"""
import pandas as pd
import numpy as np
from .utils import get_logger

logger = get_logger("sensor_fusion")


def fuse_hourly_with_calendar(hourly: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Merge hourly telemetry with calendar data (weather, events)."""
    if hourly is None or calendar is None:
        return hourly
    
    hourly = hourly.copy()
    hourly["date"] = hourly["timestamp"].dt.date.astype(str)
    calendar_prep = calendar.copy()
    calendar_prep["date"] = calendar_prep["date"].dt.date.astype(str)
    
    merged = hourly.merge(
        calendar_prep[["date", "day_type", "event", "avg_temp_c", "rainfall_mm",
                       "humidity_pct", "is_weekend", "is_holiday", "is_exam", "has_event"]],
        on="date",
        how="left"
    )
    merged.drop(columns=["date"], inplace=True)
    logger.info(f"Fused hourly with calendar: {len(merged)} rows")
    return merged


def fuse_daily_with_calendar(daily: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Merge daily summaries with calendar data."""
    if daily is None or calendar is None:
        return daily
    
    daily = daily.copy()
    daily["date_str"] = daily["date"].dt.date.astype(str)
    calendar_prep = calendar.copy()
    calendar_prep["date_str"] = calendar_prep["date"].dt.date.astype(str)
    
    merged = daily.merge(
        calendar_prep[["date_str", "day_type", "event", "avg_temp_c", "rainfall_mm",
                       "humidity_pct", "is_weekend", "is_holiday", "is_exam", "has_event"]],
        on="date_str",
        how="left"
    )
    merged.drop(columns=["date_str"], inplace=True)
    logger.info(f"Fused daily with calendar: {len(merged)} rows")
    return merged


def fuse_daily_with_buildings(daily: pd.DataFrame, buildings: pd.DataFrame) -> pd.DataFrame:
    """Merge daily summaries with building metadata."""
    if daily is None or buildings is None:
        return daily
    
    merged = daily.merge(buildings, on="building_id", how="left")
    logger.info(f"Fused daily with buildings: {len(merged)} rows")
    return merged


def aggregate_hourly_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly streams to daily level for comparison with daily_summary."""
    if hourly is None:
        return None
    
    hourly = hourly.copy()
    hourly["date"] = hourly["timestamp"].dt.date
    
    agg = hourly.groupby(["date", "building_id"]).agg(
        electricity_kwh_hourly_sum=("electricity_kwh", "sum"),
        water_litres_hourly_sum=("water_litres", "sum"),
        solar_generation_kwh_hourly_sum=("solar_generation_kwh", "sum"),
        footfall_total=("footfall", "sum"),
        hourly_records_count=("timestamp", "count"),
        missing_electricity=("electricity_kwh", lambda x: x.isnull().sum()),
        missing_water=("water_litres", lambda x: x.isnull().sum()),
    ).reset_index()
    
    agg["date"] = pd.to_datetime(agg["date"])
    logger.info(f"Aggregated hourly to daily: {len(agg)} rows")
    return agg


def create_feature_matrix(hourly: pd.DataFrame, calendar: pd.DataFrame = None) -> pd.DataFrame:
    """Create feature matrix for forecasting from hourly data."""
    if hourly is None:
        return None
    
    df = hourly.copy()
    
    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_of_month"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["is_night"] = df["hour"].apply(lambda h: 1 if h < 6 or h >= 22 else 0)
    df["is_peak"] = df["hour"].apply(lambda h: 1 if 9 <= h <= 17 else 0)
    
    # Merge calendar if available
    if calendar is not None:
        df = fuse_hourly_with_calendar(df, calendar)
    
    return df
