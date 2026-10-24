"""
EcoGrid AI - Database Module
SQLite-backed persistence with offline buffering and idempotent ingestion.
"""
import sqlite3
import json
import uuid
from pathlib import Path
from contextlib import contextmanager
import pandas as pd
from .config import get_db_path
from .utils import get_logger

logger = get_logger("database")

SCHEMA_VERSION = 1

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS buildings (
    building_id TEXT PRIMARY KEY,
    building_name TEXT,
    building_type TEXT,
    floors INTEGER,
    area_sqm REAL,
    design_capacity INTEGER,
    has_solar INTEGER,
    solar_capacity_kw REAL,
    year_built INTEGER,
    has_composter INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sensor_readings (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    received_at TEXT,
    building_id TEXT NOT NULL,
    electricity_kwh REAL,
    water_litres REAL,
    solar_generation_kwh REAL,
    footfall INTEGER,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source TEXT DEFAULT 'telemetry',
    UNIQUE(building_id, timestamp)
);

CREATE TABLE IF NOT EXISTS waste_events (
    event_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    building_id TEXT NOT NULL,
    bin_id TEXT,
    weight_g REAL,
    moisture_pct REAL,
    metal_signal REAL,
    transparency REAL,
    acoustic_hardness REAL,
    category TEXT,
    predicted_class TEXT,
    confidence REAL,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    building_id TEXT NOT NULL,
    resource TEXT NOT NULL,
    predicted_value REAL,
    model_type TEXT,
    horizon_hours INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS anomalies (
    alert_id TEXT PRIMARY KEY,
    building_id TEXT NOT NULL,
    resource TEXT,
    detection_timestamp TEXT,
    observed_value TEXT,
    expected_value TEXT,
    deviation_pct REAL,
    severity TEXT,
    method TEXT,
    evidence TEXT,
    recommended_investigation TEXT,
    status TEXT DEFAULT 'active',
    acknowledged_at TIMESTAMP,
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS recommendations (
    rec_id TEXT PRIMARY KEY,
    title TEXT,
    building TEXT,
    evidence TEXT,
    suggested_action TEXT,
    expected_impact TEXT,
    confidence TEXT,
    priority TEXT,
    calculation_method TEXT,
    verification_status TEXT DEFAULT 'pending',
    user_feedback TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS incentive_scores (
    id TEXT PRIMARY KEY,
    building_id TEXT NOT NULL,
    total_score REAL,
    electricity_score REAL,
    water_score REAL,
    waste_diversion_score REAL,
    anomaly_response_score REAL,
    data_quality_score REAL,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ingestion_queue (
    queue_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    event_type TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    retries INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY,
    action TEXT NOT NULL,
    entity_type TEXT,
    entity_id TEXT,
    details TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sync_status (
    id TEXT PRIMARY KEY,
    last_sync_at TIMESTAMP,
    records_synced INTEGER,
    status TEXT,
    details TEXT
);

CREATE INDEX IF NOT EXISTS idx_sensor_building_ts ON sensor_readings(building_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_waste_building ON waste_events(building_id);
CREATE INDEX IF NOT EXISTS idx_anomalies_building ON anomalies(building_id, status);
CREATE INDEX IF NOT EXISTS idx_predictions_building ON predictions(building_id, resource);
CREATE INDEX IF NOT EXISTS idx_ingestion_status ON ingestion_queue(status);
"""


class Database:
    """SQLite database manager with offline buffering."""
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or get_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False
    
    @contextmanager
    def get_connection(self):
        """Get a database connection with proper error handling."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def initialize(self):
        """Create tables and apply schema."""
        with self.get_connection() as conn:
            conn.executescript(CREATE_TABLES)
            # Check/set schema version
            cursor = conn.execute("SELECT MAX(version) FROM schema_version")
            current = cursor.fetchone()[0]
            if current is None or current < SCHEMA_VERSION:
                conn.execute(
                    "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                    (SCHEMA_VERSION,)
                )
            self._initialized = True
            logger.info(f"Database initialized at {self.db_path}")
    
    def insert_sensor_reading(self, reading: dict) -> bool:
        """Insert a sensor reading with duplicate detection."""
        rid = reading.get("id", str(uuid.uuid4())[:16])
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO sensor_readings
                    (id, timestamp, received_at, building_id, electricity_kwh,
                     water_litres, solar_generation_kwh, footfall, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rid, reading["timestamp"], reading.get("received_at"),
                    reading["building_id"], reading.get("electricity_kwh"),
                    reading.get("water_litres"), reading.get("solar_generation_kwh"),
                    reading.get("footfall"), reading.get("source", "telemetry")
                ))
            return True
        except sqlite3.IntegrityError:
            logger.debug(f"Duplicate sensor reading: {rid}")
            return False
        except Exception as e:
            logger.error(f"Failed to insert sensor reading: {e}")
            self._queue_for_retry("sensor_reading", reading)
            return False
    
    def insert_waste_event(self, event: dict) -> bool:
        """Insert a waste event idempotently."""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO waste_events
                    (event_id, timestamp, building_id, bin_id, weight_g,
                     moisture_pct, metal_signal, transparency, acoustic_hardness,
                     category, predicted_class, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event["event_id"], event["timestamp"], event["building_id"],
                    event.get("bin_id"), event.get("weight_g"), event.get("moisture_pct"),
                    event.get("metal_signal"), event.get("transparency"),
                    event.get("acoustic_hardness"), event.get("category"),
                    event.get("predicted_class"), event.get("confidence")
                ))
            return True
        except Exception as e:
            logger.error(f"Failed to insert waste event: {e}")
            return False
    
    def insert_anomaly(self, alert: dict) -> bool:
        """Insert an anomaly alert."""
        try:
            with self.get_connection() as conn:
                observed = json.dumps(alert.get("observed_value")) if isinstance(alert.get("observed_value"), dict) else str(alert.get("observed_value", ""))
                expected = json.dumps(alert.get("expected_value")) if isinstance(alert.get("expected_value"), dict) else str(alert.get("expected_value", ""))
                conn.execute("""
                    INSERT OR REPLACE INTO anomalies
                    (alert_id, building_id, resource, detection_timestamp,
                     observed_value, expected_value, deviation_pct, severity,
                     method, evidence, recommended_investigation, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert["alert_id"], alert["building_id"], alert.get("resource"),
                    alert.get("detection_timestamp"), observed, expected,
                    alert.get("deviation_pct"), alert.get("severity"),
                    alert.get("method"), alert.get("evidence"),
                    alert.get("recommended_investigation"), alert.get("status", "active")
                ))
            return True
        except Exception as e:
            logger.error(f"Failed to insert anomaly: {e}")
            return False
    
    def update_anomaly_status(self, alert_id: str, status: str):
        """Update anomaly status (active/acknowledged/resolved)."""
        ts_field = "acknowledged_at" if status == "acknowledged" else "resolved_at"
        with self.get_connection() as conn:
            conn.execute(f"""
                UPDATE anomalies SET status = ?, {ts_field} = CURRENT_TIMESTAMP
                WHERE alert_id = ?
            """, (status, alert_id))
    
    def insert_recommendation(self, rec: dict) -> bool:
        """Insert a recommendation."""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO recommendations
                    (rec_id, title, building, evidence, suggested_action,
                     expected_impact, confidence, priority, calculation_method,
                     verification_status, user_feedback)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rec["rec_id"], rec["title"], rec.get("building"),
                    rec.get("evidence"), rec.get("suggested_action"),
                    rec.get("expected_impact"), rec.get("confidence"),
                    rec.get("priority"), rec.get("calculation_method"),
                    rec.get("verification_status", "pending"),
                    rec.get("user_feedback")
                ))
            return True
        except Exception as e:
            logger.error(f"Failed to insert recommendation: {e}")
            return False
    
    def insert_prediction(self, prediction: dict) -> bool:
        """Insert a forecast prediction."""
        try:
            pid = str(uuid.uuid4())[:16]
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT INTO predictions
                    (id, timestamp, building_id, resource, predicted_value,
                     model_type, horizon_hours)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    pid, prediction["timestamp"], prediction["building_id"],
                    prediction["resource"], prediction["predicted_value"],
                    prediction.get("model_type"), prediction.get("horizon_hours")
                ))
            return True
        except Exception as e:
            logger.error(f"Failed to insert prediction: {e}")
            return False
    
    def get_anomalies(self, building_id: str = None, status: str = None) -> list:
        """Get anomalies, optionally filtered."""
        query = "SELECT * FROM anomalies WHERE 1=1"
        params = []
        if building_id:
            query += " AND building_id = ?"
            params.append(building_id)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC"
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_recommendations(self) -> list:
        """Get all recommendations."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM recommendations ORDER BY priority, created_at DESC")
            return [dict(row) for row in cursor.fetchall()]
    
    def _queue_for_retry(self, event_type: str, payload: dict):
        """Queue an event for retry when offline."""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT INTO ingestion_queue (queue_id, payload, event_type)
                    VALUES (?, ?, ?)
                """, (str(uuid.uuid4())[:16], json.dumps(payload, default=str), event_type))
        except Exception as e:
            logger.error(f"Failed to queue event: {e}")
    
    def process_queue(self) -> int:
        """Process pending items in the ingestion queue."""
        processed = 0
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM ingestion_queue WHERE status = 'pending' ORDER BY created_at LIMIT 100"
            )
            items = cursor.fetchall()
            
            for item in items:
                item_dict = dict(item)
                payload = json.loads(item_dict["payload"])
                event_type = item_dict["event_type"]
                
                success = False
                if event_type == "sensor_reading":
                    success = self.insert_sensor_reading(payload)
                elif event_type == "waste_event":
                    success = self.insert_waste_event(payload)
                
                if success:
                    conn.execute(
                        "UPDATE ingestion_queue SET status = 'processed', processed_at = CURRENT_TIMESTAMP WHERE queue_id = ?",
                        (item_dict["queue_id"],)
                    )
                    processed += 1
                else:
                    conn.execute(
                        "UPDATE ingestion_queue SET retries = retries + 1 WHERE queue_id = ?",
                        (item_dict["queue_id"],)
                    )
        
        if processed > 0:
            logger.info(f"Processed {processed} queued items")
        return processed
    
    def get_stats(self) -> dict:
        """Get database statistics."""
        with self.get_connection() as conn:
            stats = {}
            for table in ["sensor_readings", "waste_events", "predictions",
                         "anomalies", "recommendations", "ingestion_queue"]:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM ingestion_queue WHERE status = 'pending'")
            stats["pending_queue"] = cursor.fetchone()[0]
            
            return stats
    
    def log_audit(self, action: str, entity_type: str = None,
                  entity_id: str = None, details: str = None):
        """Log an audit event."""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO audit_log (id, action, entity_type, entity_id, details)
                VALUES (?, ?, ?, ?, ?)
            """, (str(uuid.uuid4())[:16], action, entity_type, entity_id, details))

    def ingest_buildings(self, buildings_df):
        """Bulk ingest buildings metadata."""
        if buildings_df is None or len(buildings_df) == 0:
            return 0
        self.initialize()
        records = []
        for _, row in buildings_df.iterrows():
            records.append((
                str(row["building_id"]), str(row.get("building_name", "")),
                str(row.get("building_type", "")), int(row.get("floors", 1)),
                float(row.get("area_sqm", 0.0)), int(row.get("design_capacity", 0)),
                int(row.get("has_solar", 0)), float(row.get("solar_capacity_kw", 0.0)),
                int(row.get("year_built", 2020)), int(row.get("has_composter", 0))
            ))
        with self.get_connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO buildings
                (building_id, building_name, building_type, floors, area_sqm, design_capacity, has_solar, solar_capacity_kw, year_built, has_composter)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
        return len(records)

    def ingest_sensor_readings(self, hourly_df):
        """Bulk ingest hourly telemetry records."""
        if hourly_df is None or len(hourly_df) == 0:
            return 0
        self.initialize()
        records = []
        for _, row in hourly_df.iterrows():
            rid = f"{row['building_id']}_{row['timestamp']}"
            records.append((
                rid, str(row["timestamp"]), str(row.get("received_at", "")),
                str(row["building_id"]),
                float(row["electricity_kwh"]) if pd.notna(row.get("electricity_kwh")) else None,
                float(row["water_litres"]) if pd.notna(row.get("water_litres")) else None,
                float(row["solar_generation_kwh"]) if pd.notna(row.get("solar_generation_kwh")) else None,
                int(row["footfall"]) if pd.notna(row.get("footfall")) else None,
                "telemetry"
            ))
        with self.get_connection() as conn:
            conn.executemany("""
                INSERT OR IGNORE INTO sensor_readings
                (id, timestamp, received_at, building_id, electricity_kwh, water_litres, solar_generation_kwh, footfall, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
        return len(records)

    def ingest_waste_events(self, waste_df):
        """Bulk ingest waste events."""
        if waste_df is None or len(waste_df) == 0:
            return 0
        self.initialize()
        records = []
        for idx, row in waste_df.iterrows():
            eid = str(row.get("event_id", f"WE-{idx}"))
            records.append((
                eid, str(row["timestamp"]), str(row["building_id"]),
                str(row.get("bin_id", "")),
                float(row["weight_g"]) if pd.notna(row.get("weight_g")) else None,
                float(row["moisture_pct"]) if pd.notna(row.get("moisture_pct")) else None,
                float(row["metal_signal"]) if pd.notna(row.get("metal_signal")) else None,
                float(row["transparency"]) if pd.notna(row.get("transparency")) else None,
                float(row["acoustic_hardness"]) if pd.notna(row.get("acoustic_hardness")) else None,
                str(row.get("category", "")),
                str(row.get("predicted_class", "")),
                float(row["confidence"]) if pd.notna(row.get("confidence")) else None
            ))
        with self.get_connection() as conn:
            conn.executemany("""
                INSERT OR IGNORE INTO waste_events
                (event_id, timestamp, building_id, bin_id, weight_g, moisture_pct, metal_signal, transparency, acoustic_hardness, category, predicted_class, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
        return len(records)

    def ingest_anomalies(self, alerts):
        """Bulk ingest anomaly alerts."""
        if not alerts:
            return 0
        self.initialize()
        records = []
        for alert in alerts:
            observed = json.dumps(alert.get("observed_value")) if isinstance(alert.get("observed_value"), dict) else str(alert.get("observed_value", ""))
            expected = json.dumps(alert.get("expected_value")) if isinstance(alert.get("expected_value"), dict) else str(alert.get("expected_value", ""))
            records.append((
                alert["alert_id"], str(alert["building_id"]), alert.get("resource"),
                str(alert.get("detection_timestamp")), observed, expected,
                float(alert.get("deviation_pct", 0.0)), alert.get("severity", "medium"),
                alert.get("method", "statistical"), alert.get("evidence", ""),
                alert.get("recommended_investigation", ""), alert.get("status", "active")
            ))
        with self.get_connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO anomalies
                (alert_id, building_id, resource, detection_timestamp, observed_value, expected_value, deviation_pct, severity, method, evidence, recommended_investigation, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
        return len(records)


_db_instance = None

def get_db(db_path: Path = None) -> Database:
    """Get database singleton instance."""
    global _db_instance
    if _db_instance is None or db_path is not None:
        _db_instance = Database(db_path)
    return _db_instance
