"""
EcoGrid AI - Comprehensive Test Suite
Tests data loading, waste classification, forecasting, anomaly detection,
circular accounting, root-cause attribution, and database operations.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import json

from src.config import PROJECT_ROOT
from src.data_loader import (
    normalize_building_id, load_buildings, load_calendar,
    load_daily_summary, load_hourly_streams, load_manual_logs,
    load_waste_train, load_waste_test
)
from src.waste_classifier import WasteClassifier
from src.forecasting import ResourceForecaster, create_lag_features
from src.anomaly_detection import AnomalyDetector
from src.root_cause import RootCauseAnalyzer
from src.circular_accounting import CircularAccountant
from src.incentives import SustainabilityScorer
from src.database import Database


# ==========================================
# 1. DATA LOADER & NORMALIZATION TESTS
# ==========================================

def test_building_alias_normalization():
    """Verify alias normalization for common facilities logging variations."""
    assert normalize_building_id("Boys Hostle") == "B05"
    assert normalize_building_id("BH") == "B05"
    assert normalize_building_id("Girls hostle") == "B06"
    assert normalize_building_id("Eng Labs") == "B04"
    assert normalize_building_id("Food Court") == "B07"
    assert normalize_building_id("Canteen") == "B07"
    assert normalize_building_id("Central Library") == "B03"
    assert normalize_building_id("Admin") == "B08"
    assert normalize_building_id("B02") == "B02"


def test_data_loader_real_dataset():
    """Verify real benchmark dataset files load correctly with valid schemas."""
    buildings = load_buildings()
    assert buildings is not None
    assert len(buildings) == 8
    assert "building_id" in buildings.columns
    assert "design_capacity" in buildings.columns
    
    calendar = load_calendar()
    assert calendar is not None
    assert len(calendar) == 120
    assert "date" in calendar.columns
    assert "avg_temp_c" in calendar.columns
    
    daily = load_daily_summary()
    assert daily is not None
    assert len(daily) == 960  # 120 days * 8 buildings
    assert "electricity_kwh" in daily.columns
    assert "water_litres" in daily.columns
    
    hourly = load_hourly_streams()
    assert hourly is not None
    assert len(hourly) > 20000  # 23,008 after dedup
    assert "timestamp" in hourly.columns
    
    waste_train = load_waste_train()
    assert waste_train is not None
    assert len(waste_train) == 6314
    assert "category" in waste_train.columns
    assert waste_train["category"].nunique() == 6


# ==========================================
# 2. WASTE CLASSIFIER & LATENCY TESTS
# ==========================================

def test_waste_classifier_accuracy_and_latency():
    """Verify smart-bin classifier meets accuracy (>85%) and edge latency (<2s) constraints."""
    clf = WasteClassifier()
    model_file = PROJECT_ROOT / "models" / "waste_classifier.joblib"
    assert model_file.exists(), "Trained model file missing. Run pipeline first."
    
    loaded = clf.load(model_file)
    assert loaded is True
    
    meta = clf.model_metadata
    best_metrics = meta.get("best_model_metrics", {})
    
    # Target accuracy constraint
    accuracy = best_metrics.get("accuracy", 0)
    assert accuracy >= 0.85, f"Accuracy {accuracy*100:.2f}% did not meet target > 85%"
    
    # Target edge latency constraint (< 2000 ms)
    latency_s = best_metrics.get("prediction_latency_per_event_s", 0)
    assert latency_s < 2.0, f"Edge latency {latency_s*1000:.2f} ms exceeded 2000 ms"


def test_single_waste_event_prediction():
    """Test real-time edge prediction on a single smart-bin sensor reading."""
    clf = WasteClassifier()
    clf.load()
    
    sample_event = {
        "weight_g": 35.0,
        "moisture_pct": 12.0,
        "metal_signal": 0.0,
        "transparency": 0.85,
        "acoustic_hardness": 0.35,
    }
    
    res = clf.predict_single(sample_event)
    assert "predicted_class" in res
    assert res["predicted_class"] in ["plastic", "glass", "paper_cardboard", "food_organic", "metal", "ewaste"]
    assert res["confidence"] is not None


# ==========================================
# 3. FORECASTING & BACKTESTING TESTS
# ==========================================

def test_lag_feature_creation():
    """Verify lag and rolling feature engineering for time-series forecasting."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=50, freq="h"),
        "electricity_kwh": np.linspace(10, 50, 50)
    })
    
    lagged = create_lag_features(df, "electricity_kwh", lags=[1, 2, 6], rolling_windows=[6])
    assert "electricity_kwh_lag_1" in lagged.columns
    assert "electricity_kwh_lag_2" in lagged.columns
    assert "electricity_kwh_rolling_mean_6" in lagged.columns
    assert lagged["electricity_kwh_lag_1"].iloc[1] == 10.0


def test_forecasting_inference():
    """Test 6-hour and 12-hour recursive forecasting generation."""
    hourly = load_hourly_streams()
    forecaster = ResourceForecaster()
    loaded = forecaster.load()
    assert loaded is True
    
    # Generate 6-hour forecast
    fc_6h = forecaster.forecast(hourly, building_id="B01", resource="electricity", horizon_hours=6)
    assert fc_6h is not None
    assert len(fc_6h) == 6
    assert (fc_6h["predicted_value"] >= 0).all()  # Non-negative resource load
    
    # Generate 12-hour forecast
    fc_12h = forecaster.forecast(hourly, building_id="B01", resource="electricity", horizon_hours=12)
    assert fc_12h is not None
    assert len(fc_12h) == 12


# ==========================================
# 4. ANOMALY DETECTION & ROOT CAUSE TESTS
# ==========================================

def test_anomaly_detection_schema():
    """Test that anomaly alerts follow the standardized explainable schema."""
    hourly = load_hourly_streams()
    detector = AnomalyDetector()
    
    # Test on a single building
    alerts = detector.detect_statistical_anomalies(hourly, building_id="B05", resource_col="water_litres", resource_name="water")
    assert isinstance(alerts, list)
    if alerts:
        a = alerts[0]
        assert "alert_id" in a
        assert "building_id" in a
        assert "observed_value" in a
        assert "expected_value" in a
        assert "evidence" in a
        assert "recommended_investigation" in a


def test_root_cause_attribution():
    """Verify root cause analyzer attributes anomalous events with evidence."""
    data = {
        "daily_summary": load_daily_summary(),
        "calendar": load_calendar(),
        "manual_logs": load_manual_logs(),
        "hourly_streams": load_hourly_streams(),
    }
    
    analyzer = RootCauseAnalyzer()
    dummy_alert = {
        "alert_id": "TEST-ALRT-001",
        "building_id": "B07",
        "resource": "electricity",
        "detection_timestamp": "2026-08-15 13:00:00",
        "observed_value": 85.0,
        "expected_value": 40.0,
        "deviation_pct": 112.5,
    }
    
    analysis = analyzer.analyze(dummy_alert, data)
    assert "alert_id" in analysis
    assert "ranked_explanations" in analysis
    assert "recommended_action" in analysis


# ==========================================
# 5. CIRCULAR ACCOUNTING & SCORING TESTS
# ==========================================

def test_circular_accounting_mass_conservation():
    """Verify circular accounting formulas and 4:1 composting mass ratio."""
    daily = load_daily_summary()
    buildings = load_buildings()
    accountant = CircularAccountant()
    
    metrics = accountant.calculate_campus_metrics(daily, buildings)
    
    assert "electricity" in metrics
    assert "water" in metrics
    assert "waste" in metrics
    
    waste = metrics["waste"]
    composted = waste["composted_kg"]
    compost_output = waste["estimated_compost_output_kg"]
    
    # 4:1 composting ratio
    assert np.isclose(compost_output, composted / 4.0, atol=0.1)
    
    # Diversion rate between 0 and 100%
    assert 0 <= waste["diversion_rate_pct"] <= 100


def test_sustainability_scoring():
    """Verify transparent building sustainability scoring system."""
    daily = load_daily_summary()
    buildings = load_buildings()
    scorer = SustainabilityScorer()
    
    scores = scorer.calculate_scores(daily, buildings)
    assert len(scores) == 8
    assert "total_score" in scores.columns
    assert (scores["total_score"] >= 0).all()
    assert (scores["total_score"] <= 100).all()


# ==========================================
# 6. DATABASE & OFFLINE BUFFERING TESTS
# ==========================================

def test_database_idempotent_ingestion(tmp_path):
    """Verify SQLite database handles duplicates and offline buffering properly."""
    db_file = tmp_path / "test_ecogrid.db"
    db = Database(db_file)
    db.initialize()
    
    # 1. Insert sensor reading
    reading = {
        "id": "R001",
        "timestamp": "2026-06-20 10:00:00",
        "building_id": "B01",
        "electricity_kwh": 25.5,
        "water_litres": 150.0,
        "solar_generation_kwh": 12.0,
        "footfall": 45,
    }
    res1 = db.insert_sensor_reading(reading)
    assert res1 is True
    
    # 2. Duplicate insertion must be rejected idempotently without crash
    res2 = db.insert_sensor_reading(reading)
    stats = db.get_stats()
    assert stats["sensor_readings"] == 1
    
    # 3. Anomaly insertion and status update
    alert = {
        "alert_id": "ALT-001",
        "building_id": "B01",
        "resource": "water",
        "detection_timestamp": "2026-06-20 10:00:00",
        "observed_value": 500.0,
        "expected_value": 150.0,
        "deviation_pct": 233.3,
        "severity": "high",
        "status": "active"
    }
    db.insert_anomaly(alert)
    db.update_anomaly_status("ALT-001", "acknowledged")
    anomalies = db.get_anomalies(building_id="B01")
    assert len(anomalies) == 1
    assert anomalies[0]["status"] == "acknowledged"
