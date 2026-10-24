"""
EcoGrid AI - Main Pipeline Runner
Loads datasets, trains waste classifier & forecasting models,
runs anomaly detection, circular accounting, and populates SQLite.
"""
import sys
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROJECT_ROOT
from src.data_loader import load_all_data
from src.waste_classifier import WasteClassifier
from src.forecasting import ResourceForecaster
from src.anomaly_detection import AnomalyDetector
from src.root_cause import RootCauseAnalyzer
from src.recommendations import RecommendationEngine
from src.circular_accounting import CircularAccountant
from src.incentives import SustainabilityScorer
from src.database import get_db
from src.utils import get_logger

logger = get_logger("train_and_process")


def run_pipeline(force_retrain: bool = False):
    print("=" * 60)
    print("ECOGRID AI - FULL DATA & MODEL PIPELINE")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[Phase 1] Loading Datasets from data/raw/P2P_DATASET...")
    data = load_all_data()
    for name, df in data.items():
        if df is not None:
            print(f"  [OK] {name}: {len(df)} records")
        else:
            print(f"  [X] {name}: Failed to load")
            
    # Create output directories
    (PROJECT_ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "models" / "forecasts").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "predictions").mkdir(parents=True, exist_ok=True)

    # 2. Waste Classifier
    print("\n[Phase 2] Smart-Bin Edge Waste Classifier...")
    clf = WasteClassifier()
    waste_model_file = PROJECT_ROOT / "models" / "waste_classifier.joblib"
    
    if waste_model_file.exists() and not force_retrain:
        clf.load(waste_model_file)
        waste_metrics = clf.model_metadata
        print("  [OK] Loaded existing trained model from disk")
    else:
        waste_metrics = clf.train(data["waste_train"], model_dir=PROJECT_ROOT / "models")
        with open(PROJECT_ROOT / "reports" / "waste_classification_metrics.json", "w", encoding="utf-8") as f:
            json.dump(waste_metrics, f, indent=2)
        
    best_metrics = waste_metrics.get("best_model_metrics", {})
    latency_ms = best_metrics.get("prediction_latency_per_event_s", 0) * 1000
    print(f"  [OK] Selected Model: {waste_metrics.get('selected_model')}")
    print(f"  [OK] Val Accuracy: {best_metrics.get('accuracy', 0)*100:.2f}% (Target: >85%, Target Met: {waste_metrics.get('accuracy_target_met')})")
    print(f"  [OK] Macro F1: {best_metrics.get('macro_f1', 0):.4f}")
    print(f"  [OK] Edge Latency per event: {latency_ms:.3f} ms (Target: < 2000 ms, Target Met: {waste_metrics.get('latency_target_met')})")
    
    # Predict on test set if available
    test_pred_path = PROJECT_ROOT / "data" / "predictions" / "waste_events_test_predictions.csv"
    if data["waste_test"] is not None and not test_pred_path.exists():
        preds = clf.generate_test_predictions(data["waste_test"])
        print(f"  [OK] Generated predictions for {len(preds)} test events")
    elif test_pred_path.exists():
        print(f"  [OK] Test predictions already generated at {test_pred_path}")

    # 3. Forecasting Models
    print("\n[Phase 3] Multi-Building Resource Forecasting Models (6h & 12h)...")
    forecaster = ResourceForecaster()
    forecast_model_file = PROJECT_ROOT / "models" / "forecasters.joblib"
    
    if forecast_model_file.exists() and not force_retrain:
        forecaster.load(forecast_model_file)
        print("  [OK] Loaded 24 forecasting models from disk")
    else:
        forecaster.train_all(data["hourly_streams"])
        forecaster.save()
        print("  [OK] Trained & saved 24 forecasting models")
        
    print(f"  [OK] Forecasting ready for electricity, water, and solar generation across all campus blocks")

    # 4. Anomaly Detection & Root-Cause Attribution
    print("\n[Phase 4] Detecting Anomalies & Attributing Root Causes...")
    detector = AnomalyDetector()
    alerts = detector.detect_all(data["hourly_streams"])
    print(f"  [OK] Total Anomaly Alerts Detected: {len(alerts)}")
    
    analyzer = RootCauseAnalyzer()
    enriched_alerts = []
    for alert in alerts[:40]:  # Enrich top 40 alerts with multi-source evidence
        analysis = analyzer.analyze(alert, data)
        enriched_alert = alert.copy()
        enriched_alert["root_cause"] = analysis.get("ranked_explanations", [])
        enriched_alert["recommended_action"] = analysis.get("recommended_action")
        enriched_alerts.append(enriched_alert)
        
    with open(PROJECT_ROOT / "reports" / "anomalies_and_root_causes.json", "w", encoding="utf-8") as f:
        json.dump(enriched_alerts, f, indent=2, default=str)
    print(f"  [OK] Root-Cause Attribution completed for active alerts")

    # 5. Circular Accounting & Incentives Scoring
    print("\n[Phase 5] Calculating Circular Accounting & Sustainability Scores...")
    accountant = CircularAccountant()
    campus_metrics = accountant.calculate_campus_metrics(data["daily_summary"], data["buildings"])
    
    with open(PROJECT_ROOT / "reports" / "circular_accounting_summary.json", "w", encoding="utf-8") as f:
        json.dump(campus_metrics, f, indent=2, default=str)
        
    scorer = SustainabilityScorer()
    scores_df = scorer.calculate_scores(data["daily_summary"], data["buildings"], alerts)
    scores_df.to_csv(PROJECT_ROOT / "reports" / "sustainability_scores.csv", index=False)
    
    rec_engine = RecommendationEngine()
    recs = rec_engine.generate_all(data, alerts)
    with open(PROJECT_ROOT / "reports" / "recommendations.json", "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=2, default=str)
        
    print(f"  [OK] Sustainability Scores calculated for {len(scores_df)} buildings")
    print(f"  [OK] Campus Diversion Rate: {campus_metrics.get('waste', {}).get('diversion_rate_pct', 0)}%")
    print(f"  [OK] Compost Generated: {campus_metrics.get('waste', {}).get('estimated_compost_output_kg', 0)} kg")
    print(f"  [OK] Recommendations Generated: {len(recs)} prioritized actions")

    # 6. SQLite Database Ingestion
    print("\n[Phase 6] Populating SQLite Database (data/ecogrid.db)...")
    db = get_db()
    db.ingest_buildings(data["buildings"])
    db.ingest_sensor_readings(data["hourly_streams"])
    db.ingest_waste_events(data["waste_train"])
    db.ingest_anomalies(alerts)
    for r in recs:
        db.insert_recommendation(r)
        
    stats = db.get_stats()
    print(f"  [OK] SQLite database populated successfully at data/ecogrid.db")
    print(f"    - Sensor readings: {stats.get('sensor_readings', 0):,}")
    print(f"    - Waste events: {stats.get('waste_events', 0):,}")
    print(f"    - Anomalies: {stats.get('anomalies', 0):,}")
    print(f"    - Recommendations: {stats.get('recommendations', 0):,}")

    print("\n" + "=" * 60)
    print("ALL PIPELINE PHASES COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()
