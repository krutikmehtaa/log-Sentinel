"""
Batch Inference Pipeline using PySpark.

Loads trained models and applies them to new data in batch mode.
"""

import sys
from pathlib import Path
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, FloatType, TimestampType
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config
from src.utils.db_utils import db_manager


class BatchInference:
    """Batch inference pipeline for anomaly detection."""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.isolation_forest = None
        self.lstm_detector = None
    
    def load_models(self):
        """Load trained models."""
        print("Loading trained models...")
        
        models_path = Path(config.models_path)
        
        # Find latest Isolation Forest model
        if_models = list(models_path.glob("isolation_forest_*_model.pkl"))
        if if_models:
            latest_if = str(max(if_models, key=lambda p: p.stat().st_mtime)).replace('_model.pkl', '')
            print(f"Loading Isolation Forest from: {latest_if}")
            
            self.isolation_forest = {
                'model': joblib.load(f"{latest_if}_model.pkl"),
                'scaler': joblib.load(f"{latest_if}_scaler.pkl"),
                'features': joblib.load(f"{latest_if}_features.pkl")
            }
            print("✓ Isolation Forest loaded")
        else:
            print("⚠ No Isolation Forest model found")
        
        # LSTM would be loaded similarly if needed
        # For simplicity, we'll focus on Isolation Forest for batch inference
    
    def load_data(self, input_path: str) -> DataFrame:
        """Load feature data."""
        print(f"Loading data from {input_path}...")
        df = self.spark.read.parquet(input_path)
        record_count = df.count()
        print(f"Loaded {record_count} records")
        return df
    
    def predict_with_isolation_forest(self, df: DataFrame) -> DataFrame:
        """Apply Isolation Forest predictions."""
        if self.isolation_forest is None:
            print("⚠ Isolation Forest not loaded, skipping...")
            return df
        
        print("Running Isolation Forest predictions...")
        
        # Convert to Pandas for sklearn prediction
        feature_cols = self.isolation_forest['features']
        pdf = df.select(feature_cols + ['log_id', 'timestamp', 'message']).toPandas()
        
        # Prepare features
        X = pdf[feature_cols].fillna(0).values
        X_scaled = self.isolation_forest['scaler'].transform(X)
        
        # Predict
        predictions = self.isolation_forest['model'].predict(X_scaled)
        predictions = (predictions == -1).astype(int)  # Convert to 0/1
        
        # Get anomaly scores
        scores = self.isolation_forest['model'].decision_function(X_scaled)
        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min())
        scores_normalized = 1 - scores_normalized  # Invert so 1 = anomaly
        
        # Add predictions to dataframe
        pdf['if_prediction'] = predictions
        pdf['if_score'] = scores_normalized
        
        # Convert back to Spark
        schema = df.schema
        schema = schema.add(StructField("if_prediction", FloatType(), True))
        schema = schema.add(StructField("if_score", FloatType(), True))
        
        result_df = self.spark.createDataFrame(pdf[['log_id', 'if_prediction', 'if_score']])
        
        # Join with original data
        df = df.join(result_df, on='log_id', how='left')
        
        print(f"✓ Predictions complete")
        print(f"  Anomalies detected: {predictions.sum()} ({predictions.mean()*100:.2f}%)")
        
        return df
    
    def combine_predictions(self, df: DataFrame) -> DataFrame:
        """Combine predictions from multiple models."""
        print("Combining model predictions...")
        
        # For now, we only have IF, but this is where we'd ensemble
        df = df.withColumn(
            'combined_score',
            F.coalesce(F.col('if_score'), F.lit(0.0))
        )
        
        # Threshold for anomaly
        threshold = config.get('inference', 'anomaly_threshold')
        df = df.withColumn(
            'is_anomaly_predicted',
            F.when(F.col('combined_score') > threshold, 1).otherwise(0)
        )
        
        return df
    
    def save_anomalies_to_db(self, df: DataFrame):
        """Save detected anomalies to PostgreSQL."""
        print("Saving anomalies to database...")
        
        # Filter anomalies
        anomalies_df = df.filter(F.col('is_anomaly_predicted') == 1)
        anomalies_pdf = anomalies_df.select(
            'log_id',
            'timestamp',
            F.col('log_level').alias('log_level'),
            F.col('log_type').alias('log_type'),
            F.col('message').alias('message'),
            F.col('combined_score').alias('anomaly_score')
        ).toPandas()
        
        if len(anomalies_pdf) == 0:
            print("No anomalies detected")
            return
        
        # Prepare for database insertion
        anomalies_list = []
        for _, row in anomalies_pdf.iterrows():
            anomalies_list.append({
                'log_id': row['log_id'],
                'timestamp': row['timestamp'],
                'log_level': row.get('log_level', 'INFO'),
                'log_type': row['log_type'],
                'message': row['message'],
                'anomaly_score': float(row['anomaly_score']),
                'model_name': 'ensemble',
                'model_version': 'v1.0',
                'features': {}
            })
        
        # Insert to database
        db_manager.insert_anomalies(anomalies_list)
        
        print(f"✓ Saved {len(anomalies_list)} anomalies to database")
        
        # Create alerts for high-severity anomalies
        high_severity = anomalies_pdf[anomalies_pdf['anomaly_score'] > 0.8]
        print(f"Creating alerts for {len(high_severity)} high-severity anomalies...")
        
        # This would be expanded in production
        
    def save_results(self, df: DataFrame, output_path: str):
        """Save inference results."""
        print(f"Saving results to {output_path}...")
        
        # Select relevant columns
        output_df = df.select(
            'log_id',
            'timestamp',
            'log_type',
            'message',
            'combined_score',
            'is_anomaly_predicted',
            F.col('is_anomaly').alias('is_anomaly_actual')  # Ground truth if available
        )
        
        output_df.write.mode('overwrite').parquet(output_path)
        print("✓ Results saved")
    
    def run_pipeline(self, input_path: str, output_path: str):
        """Run complete inference pipeline."""
        print("\n" + "=" * 80)
        print("BATCH INFERENCE PIPELINE")
        print("=" * 80 + "\n")
        
        # Load models
        self.load_models()
        
        # Load data
        df = self.load_data(input_path)
        
        # Run predictions
        df = self.predict_with_isolation_forest(df)
        df = self.combine_predictions(df)
        
        # Cache for multiple operations
        df.cache()
        
        # Save anomalies to database
        self.save_anomalies_to_db(df)
        
        # Save full results
        self.save_results(df, output_path)
        
        # Print summary
        total = df.count()
        anomalies = df.filter(F.col('is_anomaly_predicted') == 1).count()
        
        print("\n" + "=" * 80)
        print("INFERENCE SUMMARY")
        print("=" * 80)
        print(f"Total logs processed: {total}")
        print(f"Anomalies detected:   {anomalies} ({anomalies/total*100:.2f}%)")
        
        # Show sample anomalies
        print("\nTop 10 anomalies by score:")
        df.filter(F.col('is_anomaly_predicted') == 1) \
          .orderBy(F.col('combined_score').desc()) \
          .select('timestamp', 'log_type', 'message', 'combined_score') \
          .show(10, truncate=True)
        
        print("\n✓ Inference complete!")


def main():
    """Main entry point."""
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("AnomalyDetection-BatchInference") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    try:
        # Get latest feature data
        processed_path = Path(config.processed_data_path)
        feature_files = list(processed_path.glob("features_*"))
        
        if not feature_files:
            print("Error: No feature files found. Run feature_engineering.py first.")
            sys.exit(1)
        
        input_path = str(max(feature_files, key=lambda p: p.stat().st_mtime))
        output_path = f"{config.anomalies_path}/batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}\n")
        
        # Run inference
        inference = BatchInference(spark)
        inference.run_pipeline(input_path, output_path)
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
