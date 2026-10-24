"""
EcoGrid AI - Waste Classifier Module
ML-based waste classification from smart-bin sensor readings.
"""
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    HAS_SKLEARN_ENSEMBLE = True
except Exception:
    HAS_SKLEARN_ENSEMBLE = False
    RandomForestClassifier, GradientBoostingClassifier = None, None

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import joblib
from .config import PROJECT_ROOT, get_config
from .utils import get_logger, timer

logger = get_logger("waste_classifier")

FEATURE_COLS = ["weight_g", "moisture_pct", "metal_signal", "transparency", "acoustic_hardness"]
TARGET_COL = "category"


class WasteClassifier:
    """Waste classification model with training, evaluation, and prediction."""
    
    def __init__(self):
        self.pipeline = None
        self.label_encoder = None
        self.model_metadata = {}
        self.is_trained = False
        self.classes = []
    
    @timer
    def train(self, train_df: pd.DataFrame, model_dir: Path = None) -> dict:
        """Train the waste classifier and return evaluation metrics."""
        if model_dir is None:
            model_dir = PROJECT_ROOT / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        df = train_df.dropna(subset=[TARGET_COL]).copy()
        X = df[FEATURE_COLS].copy()
        y = df[TARGET_COL].copy()
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes = list(self.label_encoder.classes_)
        
        logger.info(f"Training on {len(X)} samples, {len(self.classes)} classes")
        logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # Stratified train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Build pipelines for comparison
        models = {}
        
        # XGBoost (primary edge classifier)
        if HAS_XGBOOST:
            models["XGBoost"] = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", XGBClassifier(
                    n_estimators=200, max_depth=8, learning_rate=0.1,
                    random_state=42, eval_metric="mlogloss"
                ))
            ])
            
        if HAS_SKLEARN_ENSEMBLE and RandomForestClassifier is not None:
            models["RandomForest"] = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(
                    n_estimators=100, max_depth=12, min_samples_leaf=4,
                    random_state=42, n_jobs=-1, class_weight="balanced"
                ))
            ])
        
        # Train and evaluate each
        results = {}
        best_model_name = None
        best_accuracy = 0
        
        for name, pipeline in models.items():
            start_time = time.perf_counter()
            pipeline.fit(X_train, y_train)
            train_time = time.perf_counter() - start_time
            
            # Predict
            pred_start = time.perf_counter()
            y_pred = pipeline.predict(X_val)
            pred_time = time.perf_counter() - pred_start
            latency_per_event = pred_time / len(X_val) if len(X_val) > 0 else 0
            
            acc = accuracy_score(y_val, y_pred)
            macro_f1 = f1_score(y_val, y_pred, average="macro")
            weighted_f1 = f1_score(y_val, y_pred, average="weighted")
            
            results[name] = {
                "accuracy": round(acc, 4),
                "macro_f1": round(macro_f1, 4),
                "weighted_f1": round(weighted_f1, 4),
                "training_time_s": round(train_time, 3),
                "prediction_latency_per_event_s": round(latency_per_event, 6),
                "val_samples": len(X_val),
            }
            
            logger.info(f"{name}: accuracy={acc:.4f}, macro_f1={macro_f1:.4f}, time={train_time:.2f}s")
            
            if acc > best_accuracy:
                best_accuracy = acc
                best_model_name = name
        
        # Select best model
        self.pipeline = models[best_model_name]
        self.is_trained = True
        
        # Detailed evaluation for best model
        y_pred_best = self.pipeline.predict(X_val)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred_best)
        y_val_labels = self.label_encoder.inverse_transform(y_val)
        
        cm = confusion_matrix(y_val_labels, y_pred_labels, labels=self.classes)
        cr = classification_report(y_val_labels, y_pred_labels, output_dict=True)
        
        per_class = {}
        for cls in self.classes:
            if cls in cr:
                per_class[cls] = {
                    "precision": round(cr[cls]["precision"], 4),
                    "recall": round(cr[cls]["recall"], 4),
                    "f1": round(cr[cls]["f1-score"], 4),
                    "support": cr[cls]["support"],
                }
        
        self.model_metadata = {
            "selected_model": best_model_name,
            "classes": self.classes,
            "feature_columns": FEATURE_COLS,
            "comparison_results": results,
            "best_model_metrics": results[best_model_name],
            "per_class_metrics": per_class,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_labels": self.classes,
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "total_samples": len(df),
            "accuracy_target": 0.85,
            "latency_target_s": 2.0,
            "accuracy_target_met": best_accuracy >= 0.85,
            "latency_target_met": results[best_model_name]["prediction_latency_per_event_s"] < 2.0,
            "trained_at": pd.Timestamp.now().isoformat(),
        }
        
        # Save model and metadata
        model_path = model_dir / "waste_classifier.joblib"
        joblib.dump({
            "pipeline": self.pipeline,
            "label_encoder": self.label_encoder,
            "classes": self.classes,
            "feature_columns": FEATURE_COLS,
            "metadata": self.model_metadata,
        }, model_path)
        
        meta_path = model_dir / "model_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.model_metadata, f, indent=2, default=str)
        
        # Save evaluation report
        reports_dir = PROJECT_ROOT / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        with open(reports_dir / "model_evaluation.json", "w") as f:
            json.dump(self.model_metadata, f, indent=2, default=str)
        
        logger.info(f"Best model: {best_model_name} (accuracy={best_accuracy:.4f})")
        logger.info(f"Model saved to {model_path}")
        
        return self.model_metadata
    
    def load(self, model_path: Path = None):
        """Load a trained model from disk."""
        if model_path is None:
            model_path = PROJECT_ROOT / "models" / "waste_classifier.joblib"
        
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            return False
        
        saved = joblib.load(model_path)
        self.pipeline = saved["pipeline"]
        self.label_encoder = saved["label_encoder"]
        self.classes = saved["classes"]
        self.model_metadata = saved.get("metadata", {})
        self.is_trained = True
        logger.info(f"Model loaded from {model_path}")
        return True
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict waste categories for new events."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        
        X = df[FEATURE_COLS].copy()
        
        start = time.perf_counter()
        predictions = self.pipeline.predict(X)
        pred_labels = self.label_encoder.inverse_transform(predictions)
        elapsed = time.perf_counter() - start
        
        result = df.copy()
        result["predicted_class"] = pred_labels
        
        # Confidence scores if supported
        if hasattr(self.pipeline, "predict_proba"):
            try:
                proba = self.pipeline.predict_proba(X)
                result["confidence"] = proba.max(axis=1).round(4)
            except Exception:
                result["confidence"] = np.nan
        
        result["prediction_latency_s"] = round(elapsed / len(df), 6) if len(df) > 0 else 0
        
        logger.info(f"Predicted {len(df)} events in {elapsed:.3f}s ({elapsed/len(df)*1000:.1f}ms/event)")
        return result
    
    def predict_single(self, features: dict) -> dict:
        """Predict a single waste event from sensor readings."""
        df = pd.DataFrame([features])
        result = self.predict(df)
        row = result.iloc[0]
        return {
            "predicted_class": row["predicted_class"],
            "confidence": row.get("confidence", None),
            "features": features,
        }
    
    def generate_test_predictions(self, test_df: pd.DataFrame, output_path: Path = None) -> pd.DataFrame:
        """Generate predictions for the test set and save to CSV."""
        if output_path is None:
            output_path = PROJECT_ROOT / "data" / "predictions" / "waste_events_test_predictions.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result = self.predict(test_df)
        
        # Save with event_id and predicted_class
        output_cols = ["event_id", "predicted_class"]
        if "confidence" in result.columns:
            output_cols.append("confidence")
        
        result[output_cols].to_csv(output_path, index=False)
        logger.info(f"Test predictions saved to {output_path}")
        return result
