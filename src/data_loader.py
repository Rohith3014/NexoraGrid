"""
EcoGrid AI - Data Loader Module
Handles loading, cleaning, and normalizing all dataset files.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from .config import get_config, get_data_path, get_building_aliases, PROJECT_ROOT
from .utils import get_logger, generate_stable_id

logger = get_logger("data_loader")


def normalize_building_id(name: str, aliases: dict = None) -> str:
    """Normalize building names/aliases to standard building_id (B01-B08)."""
    if aliases is None:
        aliases = get_building_aliases()
    name_str = str(name).strip()
    # Already a standard ID
    if name_str in [f"B0{i}" for i in range(1, 9)]:
        return name_str
    # Check aliases (case-insensitive)
    for alias, bid in aliases.items():
        if name_str.lower() == alias.lower():
            return bid
    # Try partial matching
    name_lower = name_str.lower()
    partial_map = {
        'academic block a': 'B01', 'block a': 'B01', 'acad a': 'B01',
        'academic block b': 'B02', 'block b': 'B02', 'acad b': 'B02',
        'library': 'B03', 'central lib': 'B03', 'lib': 'B03',
        'eng lab': 'B04', 'engineering': 'B04', 'lab': 'B04', 'innovation': 'B04',
        'boys': 'B05', 'bh': 'B05', 'boys hostel': 'B05', 'boys hostle': 'B05',
        'girls': 'B06', 'gh': 'B06', 'girls hostel': 'B06', 'girls hostle': 'B06',
        'food': 'B07', 'fc': 'B07', 'mess': 'B07', 'food court': 'B07', 'canteen': 'B07',
        'admin': 'B08', 'administration': 'B08',
    }
    for key, bid in partial_map.items():
        if key in name_lower:
            return bid
    logger.warning(f"Could not resolve building alias: '{name_str}'")
    return name_str


def load_buildings(data_dir: Path = None) -> pd.DataFrame:
    """Load buildings metadata."""
    if data_dir is None:
        data_dir = get_data_path("raw")
    fp = data_dir / "buildings.xlsx"
    if not fp.exists():
        return None
    df = pd.read_excel(fp)
    logger.info(f"Loaded buildings: {len(df)} rows")
    return df


def load_calendar(data_dir: Path = None) -> pd.DataFrame:
    """Load calendar with weather and events."""
    if data_dir is None:
        data_dir = get_data_path("raw")
    fp = data_dir / "calendar.xlsx"
    if not fp.exists():
        return None
    df = pd.read_excel(fp)
    df["date"] = pd.to_datetime(df["date"])
    df["is_weekend"] = df["day_type"] == "weekend"
    df["is_holiday"] = df["day_type"].isin(["holiday", "semester_break"])
    df["is_exam"] = df["day_type"] == "exam"
    df["has_event"] = df["event"].notna()
    logger.info(f"Loaded calendar: {len(df)} days from {df['date'].min()} to {df['date'].max()}")
    return df


def load_daily_summary(data_dir: Path = None) -> pd.DataFrame:
    """Load daily building-level summaries."""
    if data_dir is None:
        data_dir = get_data_path("raw")
    fp = data_dir / "daily_summary.xlsx"
    if not fp.exists():
        return None
    df = pd.read_excel(fp)
    df["date"] = pd.to_datetime(df["date"])
    # Validate building IDs
    valid_ids = [f"B0{i}" for i in range(1, 9)]
    invalid = df[~df["building_id"].isin(valid_ids)]
    if len(invalid) > 0:
        logger.warning(f"Found {len(invalid)} rows with invalid building_id in daily_summary")
    logger.info(f"Loaded daily_summary: {len(df)} rows, {df['building_id'].nunique()} buildings")
    return df


def load_hourly_streams(data_dir: Path = None) -> pd.DataFrame:
    """Load hourly telemetry streams with dedup and timestamp handling."""
    if data_dir is None:
        data_dir = get_data_path("raw")
    fp = data_dir / "hourly_streams.xlsx"
    if not fp.exists():
        return None
    df = pd.read_excel(fp)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["received_at"] = pd.to_datetime(df["received_at"])
    
    # Detect and remove duplicates (same building, same timestamp)
    before_dedup = len(df)
    df["_dedup_key"] = df.apply(
        lambda r: generate_stable_id(r["building_id"], r["timestamp"]), axis=1
    )
    df = df.drop_duplicates(subset=["_dedup_key"], keep="first")
    df = df.drop(columns=["_dedup_key"])
    dupes_removed = before_dedup - len(df)
    if dupes_removed > 0:
        logger.info(f"Removed {dupes_removed} duplicate hourly records")
    
    # Sort by measurement timestamp
    df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    
    # Detect late-arriving data (received_at >> timestamp)
    df["ingestion_lag_min"] = (df["received_at"] - df["timestamp"]).dt.total_seconds() / 60
    
    logger.info(f"Loaded hourly_streams: {len(df)} rows after dedup")
    return df


def load_manual_logs(data_dir: Path = None) -> pd.DataFrame:
    """Load and normalize manual field logs."""
    if data_dir is None:
        data_dir = get_data_path("raw")
    fp = data_dir / "manual_logs.xlsx"
    if not fp.exists():
        return None
    df = pd.read_excel(fp)
    df["logged_at"] = pd.to_datetime(df["logged_at"])
    
    # Normalize building aliases
    aliases = get_building_aliases()
    df["building_id"] = df["building"].apply(lambda x: normalize_building_id(x, aliases))
    
    # Normalize units to standard values
    config = get_config()
    unit_conv = config["unit_conversions"]
    
    def normalize_value(row):
        """Convert value to standard units based on stream type."""
        val = row.get("value")
        unit = str(row.get("unit", "")).strip()
        stream = str(row.get("stream", "")).strip().lower()
        
        if pd.isna(val):
            return np.nan, "unknown"
        
        if stream == "electricity":
            factor = unit_conv["electricity"].get(unit, 1.0)
            return float(val) * factor, "kWh"
        elif stream == "water":
            factor = unit_conv["water"].get(unit, 1.0)
            return float(val) * factor, "L"
        elif stream == "waste":
            factor = unit_conv["waste"].get(unit, 1.0)
            return float(val) * factor, "kg"
        return val, unit
    
    normalized = df.apply(normalize_value, axis=1, result_type="expand")
    df["normalized_value"] = normalized[0]
    df["normalized_unit"] = normalized[1]
    
    logger.info(f"Loaded manual_logs: {len(df)} records")
    return df


def load_waste_train(data_dir: Path = None) -> pd.DataFrame:
    """Load labeled waste training data."""
    if data_dir is None:
        data_dir = get_data_path("raw")
    fp = data_dir / "waste_events_train.xlsx"
    if not fp.exists():
        return None
    df = pd.read_excel(fp)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    logger.info(f"Loaded waste_events_train: {len(df)} events, {df['category'].nunique()} classes")
    logger.info(f"Class distribution: {df['category'].value_counts().to_dict()}")
    return df


def load_waste_test(data_dir: Path = None) -> pd.DataFrame:
    """Load unlabeled waste test data."""
    if data_dir is None:
        data_dir = get_data_path("raw")
    fp = data_dir / "waste_events_test.xlsx"
    if not fp.exists():
        return None
    df = pd.read_excel(fp)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    logger.info(f"Loaded waste_events_test: {len(df)} events (unlabeled)")
    return df


def load_all_data(data_dir: Path = None) -> dict:
    """Load all datasets and return as a dictionary."""
    if data_dir is None:
        data_dir = get_data_path("raw")
    
    data = {
        "buildings": load_buildings(data_dir),
        "calendar": load_calendar(data_dir),
        "daily_summary": load_daily_summary(data_dir),
        "hourly_streams": load_hourly_streams(data_dir),
        "manual_logs": load_manual_logs(data_dir),
        "waste_train": load_waste_train(data_dir),
        "waste_test": load_waste_test(data_dir),
    }
    
    loaded = [k for k, v in data.items() if v is not None]
    missing = [k for k, v in data.items() if v is None]
    
    logger.info(f"Loaded datasets: {loaded}")
    if missing:
        logger.warning(f"Missing datasets: {missing}")
    
    return data


def generate_demo_data(seed: int = 42) -> dict:
    """Generate synthetic demo data when real data is unavailable."""
    np.random.seed(seed)
    
    buildings = pd.DataFrame({
        "building_id": [f"B0{i}" for i in range(1, 9)],
        "building_name": [
            "Academic Block A", "Academic Block B", "Central Library",
            "Engineering Labs", "Boys Hostel", "Girls Hostel",
            "Food Court", "Admin Block"
        ],
        "building_type": ["academic", "academic", "library", "labs",
                          "hostel", "hostel", "food", "admin"],
        "floors": [4, 4, 3, 3, 5, 5, 2, 3],
        "area_sqm": [9000, 7500, 5000, 6000, 8500, 7000, 3500, 2800],
        "design_capacity": [1200, 900, 600, 500, 800, 600, 1500, 150],
        "has_solar": [1, 0, 0, 1, 0, 0, 1, 0],
        "solar_capacity_kw": [120, 0, 0, 80, 0, 0, 60, 0],
        "year_built": [2012, 2016, 2014, 2019, 2013, 2017, 2015, 2012],
        "has_composter": [0, 0, 0, 0, 1, 1, 1, 0],
    })
    
    dates = pd.date_range("2026-06-20", periods=120, freq="D")
    calendar = pd.DataFrame({
        "date": dates,
        "day_of_week": dates.day_name(),
        "day_type": ["weekend" if d.weekday() >= 5 else "working" for d in dates],
        "event": [None] * 120,
        "avg_temp_c": np.random.normal(28, 4, 120).round(1),
        "rainfall_mm": np.maximum(0, np.random.exponential(2, 120)).round(1),
        "humidity_pct": np.random.normal(65, 10, 120).clip(30, 95).round(1),
    })
    calendar["is_weekend"] = calendar["day_type"] == "weekend"
    calendar["is_holiday"] = False
    calendar["is_exam"] = False
    calendar["has_event"] = False
    
    # Daily summary
    daily_rows = []
    for d in dates:
        for bid in buildings["building_id"]:
            daily_rows.append({
                "date": d,
                "building_id": bid,
                "occupancy_avg": np.random.randint(50, 500),
                "electricity_kwh": np.random.uniform(200, 1500),
                "water_litres": np.random.uniform(1000, 8000),
                "solar_generation_kwh": np.random.uniform(0, 100),
                "wet_organic_kg": np.random.uniform(5, 50),
                "dry_recyclable_kg": np.random.uniform(3, 30),
                "ewaste_kg": np.random.uniform(0, 5),
                "hazardous_kg": np.random.uniform(0, 2),
                "composted_kg": np.random.uniform(0, 15),
                "recycled_kg": np.random.uniform(2, 20),
                "contamination_pct": np.random.uniform(2, 15),
                "landfill_kg": np.random.uniform(5, 30),
            })
    daily_summary = pd.DataFrame(daily_rows)
    daily_summary["date"] = pd.to_datetime(daily_summary["date"])
    
    # Hourly streams
    hourly_rows = []
    for d in dates:
        for h in range(24):
            for bid in buildings["building_id"]:
                ts = d + pd.Timedelta(hours=h)
                hourly_rows.append({
                    "timestamp": ts,
                    "received_at": ts + pd.Timedelta(minutes=np.random.randint(0, 10)),
                    "building_id": bid,
                    "electricity_kwh": np.random.uniform(5, 80),
                    "water_litres": np.random.uniform(50, 500),
                    "solar_generation_kwh": max(0, np.random.normal(10, 5)) if 6 <= h <= 18 else 0,
                    "footfall": np.random.randint(0, 100),
                })
    hourly_streams = pd.DataFrame(hourly_rows)
    hourly_streams["timestamp"] = pd.to_datetime(hourly_streams["timestamp"])
    hourly_streams["received_at"] = pd.to_datetime(hourly_streams["received_at"])
    hourly_streams["ingestion_lag_min"] = 0
    
    # Waste events
    categories = ["food_organic", "paper_cardboard", "plastic", "glass", "metal", "ewaste"]
    n_train = 500
    waste_train = pd.DataFrame({
        "event_id": [f"WE{i:05d}" for i in range(n_train)],
        "timestamp": pd.date_range("2026-06-20", periods=n_train, freq="2h"),
        "building_id": np.random.choice(buildings["building_id"], n_train),
        "bin_id": np.random.choice([f"BIN{i:03d}" for i in range(1, 20)], n_train),
        "weight_g": np.random.uniform(50, 2000, n_train),
        "moisture_pct": np.random.uniform(0, 80, n_train),
        "metal_signal": np.random.uniform(0, 1, n_train),
        "transparency": np.random.uniform(0, 1, n_train),
        "acoustic_hardness": np.random.uniform(0, 1, n_train),
        "category": np.random.choice(categories, n_train),
    })
    
    n_test = 100
    waste_test = pd.DataFrame({
        "event_id": [f"WT{i:05d}" for i in range(n_test)],
        "timestamp": pd.date_range("2026-09-01", periods=n_test, freq="2h"),
        "building_id": np.random.choice(buildings["building_id"], n_test),
        "bin_id": np.random.choice([f"BIN{i:03d}" for i in range(1, 20)], n_test),
        "weight_g": np.random.uniform(50, 2000, n_test),
        "moisture_pct": np.random.uniform(0, 80, n_test),
        "metal_signal": np.random.uniform(0, 1, n_test),
        "transparency": np.random.uniform(0, 1, n_test),
        "acoustic_hardness": np.random.uniform(0, 1, n_test),
    })
    
    manual_logs = pd.DataFrame({
        "log_id": range(1, 21),
        "logged_at": pd.date_range("2026-06-20", periods=20, freq="6D"),
        "logger_role": np.random.choice(["warden", "technician", "supervisor"], 20),
        "building": np.random.choice(["BH", "Food Court", "B03", "Admin"], 20),
        "building_id": np.random.choice(buildings["building_id"], 20),
        "stream": np.random.choice(["water", "electricity", "waste", "note"], 20),
        "value": np.random.uniform(10, 500, 20),
        "unit": np.random.choice(["kWh", "L", "kg"], 20),
        "note": ["Demo log entry"] * 20,
        "normalized_value": np.random.uniform(10, 500, 20),
        "normalized_unit": np.random.choice(["kWh", "L", "kg"], 20),
    })
    
    return {
        "buildings": buildings,
        "calendar": calendar,
        "daily_summary": daily_summary,
        "hourly_streams": hourly_streams,
        "manual_logs": manual_logs,
        "waste_train": waste_train,
        "waste_test": waste_test,
        "_is_demo": True,
    }
