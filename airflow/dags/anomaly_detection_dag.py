"""
Airflow DAG for Anomaly Detection Pipeline.

This DAG orchestrates the end-to-end anomaly detection workflow:
1. Feature engineering
2. Model training (scheduled)
3. Batch inference
4. Monitoring and alerting
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project to path
sys.path.append('/opt/airflow')

default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}


def check_data_quality(**context):
    """Check data quality before processing."""
    from src.utils.config import config
    import pandas as pd
    
    # Get latest raw data
    raw_path = Path(config.raw_data_path)
    parquet_files = list(raw_path.glob("*.parquet"))
    
    if not parquet_files:
        raise ValueError("No data files found")
    
    latest_file = max(parquet_files, key=lambda p: p.stat().st_mtime)
    df = pd.read_parquet(latest_file)
    
    # Quality checks
    assert len(df) > 0, "Empty dataset"
    assert df['log_id'].is_unique, "Duplicate log IDs found"
    assert df['timestamp'].notna().all(), "Missing timestamps"
    
    print(f"✓ Data quality checks passed: {len(df)} records")
    return str(latest_file)


def monitor_model_performance(**context):
    """Monitor model performance and trigger retraining if needed."""
    from src.utils.db_utils import db_manager
    
    # Get recent model performance
    query = """
        SELECT model_name, metric_name, metric_value, evaluation_date
        FROM model_performance
        WHERE evaluation_date > NOW() - INTERVAL '7 days'
        ORDER BY evaluation_date DESC
    """
    
    results = db_manager.execute_query(query)
    
    # Check if F1 score dropped below threshold
    for row in results:
        if row['metric_name'] == 'f1_score' and row['metric_value'] < 0.7:
            print(f"⚠ Warning: {row['model_name']} F1 score dropped to {row['metric_value']}")
            # In production, this would trigger retraining
    
    print("✓ Model monitoring complete")


# Define DAG
with DAG(
    'anomaly_detection_pipeline',
    default_args=default_args,
    description='End-to-end anomaly detection pipeline',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    catchup=False,
    tags=['anomaly-detection', 'ml-pipeline'],
) as dag:
    
    # Task 1: Data quality check
    data_quality_check = PythonOperator(
        task_id='check_data_quality',
        python_callable=check_data_quality,
    )
    
    # Task 2: Feature engineering
    feature_engineering = BashOperator(
        task_id='feature_engineering',
        bash_command="""
            cd /opt/airflow && \
            spark-submit \
                --master local[*] \
                --driver-memory 4g \
                src/processing/feature_engineering.py
        """,
    )
    
    # Task 3: Batch inference
    batch_inference = BashOperator(
        task_id='batch_inference',
        bash_command="""
            cd /opt/airflow && \
            spark-submit \
                --master local[*] \
                --driver-memory 4g \
                src/processing/batch_inference.py
        """,
    )
    
    # Task 4: Model monitoring
    model_monitoring = PythonOperator(
        task_id='monitor_model_performance',
        python_callable=monitor_model_performance,
    )
    
    # Task 5: Generate daily report
    generate_report = PostgresOperator(
        task_id='generate_daily_report',
        postgres_conn_id='postgres_default',
        sql="""
            INSERT INTO logs_metadata (batch_id, total_logs, start_timestamp, end_timestamp)
            SELECT 
                '{{ ds }}',
                COUNT(*),
                MIN(timestamp),
                MAX(timestamp)
            FROM anomalies
            WHERE DATE(detected_at) = '{{ ds }}';
        """,
    )
    
    # Task 6: Clean old data
    cleanup = PostgresOperator(
        task_id='cleanup_old_data',
        postgres_conn_id='postgres_default',
        sql="""
            -- Archive anomalies older than 90 days
            DELETE FROM anomalies 
            WHERE detected_at < NOW() - INTERVAL '90 days';
            
            -- Clean old feature statistics
            DELETE FROM feature_statistics 
            WHERE calculation_date < NOW() - INTERVAL '180 days';
        """,
    )
    
    # Define task dependencies
    data_quality_check >> feature_engineering >> batch_inference
    batch_inference >> [model_monitoring, generate_report]
    generate_report >> cleanup


# Weekly model retraining DAG
with DAG(
    'model_retraining',
    default_args=default_args,
    description='Weekly model retraining',
    schedule_interval='0 3 * * 0',  # Sunday at 3 AM
    catchup=False,
    tags=['anomaly-detection', 'ml-training'],
) as retraining_dag:
    
    # Train Isolation Forest
    train_if = BashOperator(
        task_id='train_isolation_forest',
        bash_command="""
            cd /opt/airflow && \
            python src/models/train_isolation_forest.py
        """,
    )
    
    # Train LSTM Autoencoder
    train_lstm = BashOperator(
        task_id='train_lstm_autoencoder',
        bash_command="""
            cd /opt/airflow && \
            python src/models/train_lstm_autoencoder.py
        """,
    )
    
    # Parallel training
    [train_if, train_lstm]
