"""
Database utilities for anomaly detection system.
"""

import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from src.utils.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manager for PostgreSQL database operations."""
    
    def __init__(self):
        self.connection_params = {
            'host': config.get('database', 'host'),
            'port': config.get('database', 'port'),
            'database': config.get('database', 'name'),
            'user': config.get('database', 'user'),
            'password': config.get('database', 'password'),
        }
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = psycopg2.connect(**self.connection_params)
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute a SELECT query and return results."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]
    
    def execute_update(self, query: str, params: tuple = None) -> int:
        """Execute an INSERT/UPDATE/DELETE query."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.rowcount
    
    def insert_anomalies(self, anomalies: List[Dict[str, Any]]):
        """Batch insert anomalies."""
        if not anomalies:
            return
        
        query = """
            INSERT INTO anomalies 
            (log_id, timestamp, log_level, log_type, message, 
             anomaly_score, model_name, model_version, features)
            VALUES %s
            ON CONFLICT (log_id) DO UPDATE SET
                anomaly_score = EXCLUDED.anomaly_score,
                model_name = EXCLUDED.model_name,
                model_version = EXCLUDED.model_version
        """
        
        values = [
            (
                a['log_id'],
                a['timestamp'],
                a.get('log_level', 'INFO'),
                a['log_type'],
                a['message'],
                a['anomaly_score'],
                a['model_name'],
                a.get('model_version', 'v1.0'),
                psycopg2.extras.Json(a.get('features', {}))
            )
            for a in anomalies
        ]
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                execute_values(cur, query, values)
        
        logger.info(f"Inserted {len(anomalies)} anomalies")
    
    def create_alert(self, anomaly_id: int, severity: str, 
                     alert_type: str, description: str):
        """Create an alert for an anomaly."""
        query = """
            INSERT INTO alerts (anomaly_id, severity, alert_type, description)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (anomaly_id, severity, alert_type, description))
                alert_id = cur.fetchone()[0]
        
        logger.info(f"Created alert {alert_id} for anomaly {anomaly_id}")
        return alert_id
    
    def get_recent_anomalies(self, hours: int = 24, 
                           min_score: float = 0.7) -> List[Dict]:
        """Get recent high-score anomalies."""
        query = """
            SELECT 
                id, log_id, timestamp, log_type, message,
                anomaly_score, model_name, detected_at
            FROM anomalies
            WHERE detected_at > NOW() - INTERVAL '%s hours'
              AND anomaly_score >= %s
            ORDER BY anomaly_score DESC
            LIMIT 100
        """
        return self.execute_query(query, (hours, min_score))
    
    def record_model_performance(self, model_name: str, model_version: str,
                                metrics: Dict[str, float], dataset_size: int):
        """Record model performance metrics."""
        query = """
            INSERT INTO model_performance 
            (model_name, model_version, metric_name, metric_value, 
             evaluation_date, dataset_size)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        values = [
            (model_name, model_version, metric_name, metric_value,
             datetime.now().date(), dataset_size)
            for metric_name, metric_value in metrics.items()
        ]
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                execute_values(cur, query, values)
        
        logger.info(f"Recorded {len(metrics)} metrics for {model_name}")
    
    def update_feature_statistics(self, feature_stats: Dict[str, Dict[str, float]],
                                 sample_size: int):
        """Update feature statistics for drift monitoring."""
        query = """
            INSERT INTO feature_statistics 
            (feature_name, stat_type, stat_value, calculation_date, sample_size)
            VALUES %s
        """
        
        values = []
        for feature_name, stats in feature_stats.items():
            for stat_type, stat_value in stats.items():
                values.append((
                    feature_name,
                    stat_type,
                    stat_value,
                    datetime.now().date(),
                    sample_size
                ))
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                execute_values(cur, query, values)
        
        logger.info(f"Updated statistics for {len(feature_stats)} features")


# Singleton instance
db_manager = DatabaseManager()


if __name__ == "__main__":
    # Test database connection
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                print(f"Connected to PostgreSQL: {version}")
        
        # Test query
        anomalies = db_manager.get_recent_anomalies(hours=24, min_score=0.5)
        print(f"Found {len(anomalies)} recent anomalies")
        
    except Exception as e:
        print(f"Database connection failed: {e}")
        print("Make sure PostgreSQL is running: docker-compose up -d postgres")
