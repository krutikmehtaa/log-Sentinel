"""
Feature Engineering Pipeline using PySpark.

This module transforms raw log data into features suitable for ML models.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config


class FeatureEngineer:
    """PySpark-based feature engineering for log data."""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
    
    def load_data(self, input_path: str) -> DataFrame:
        """Load raw log data from Parquet."""
        print(f"Loading data from {input_path}...")
        df = self.spark.read.parquet(input_path)
        print(f"Loaded {df.count()} records")
        return df
    
    def extract_temporal_features(self, df: DataFrame) -> DataFrame:
        """Extract time-based features."""
        print("Extracting temporal features...")
        
        df = df.withColumn("hour", F.hour("timestamp"))
        df = df.withColumn("day_of_week", F.dayofweek("timestamp"))
        df = df.withColumn("is_weekend", 
                          F.when(F.col("day_of_week").isin([1, 7]), 1).otherwise(0))
        df = df.withColumn("is_business_hours",
                          F.when((F.col("hour") >= 9) & (F.col("hour") <= 17), 1).otherwise(0))
        
        return df
    
    def compute_rolling_statistics(self, df: DataFrame) -> DataFrame:
        """Compute rolling window statistics."""
        print("Computing rolling statistics...")
        
        # Define windows for different time intervals
        window_5min = Window.partitionBy("log_type").orderBy("timestamp").rangeBetween(-300, 0)
        window_15min = Window.partitionBy("log_type").orderBy("timestamp").rangeBetween(-900, 0)
        window_1hour = Window.partitionBy("log_type").orderBy("timestamp").rangeBetween(-3600, 0)
        
        # Convert timestamp to unix timestamp for window calculations
        df = df.withColumn("timestamp_unix", F.unix_timestamp("timestamp"))
        
        # Count of logs in different windows
        df = df.withColumn("count_5min", F.count("log_id").over(window_5min))
        df = df.withColumn("count_15min", F.count("log_id").over(window_15min))
        df = df.withColumn("count_1hour", F.count("log_id").over(window_1hour))
        
        # For API requests, compute response time statistics
        df = df.withColumn(
            "avg_response_time_5min",
            F.avg(F.when(F.col("log_type") == "API_REQUEST", 
                        F.col("response_time_ms")).otherwise(None)).over(window_5min)
        )
        
        df = df.withColumn(
            "max_response_time_5min",
            F.max(F.when(F.col("log_type") == "API_REQUEST", 
                        F.col("response_time_ms")).otherwise(None)).over(window_5min)
        )
        
        # Fill nulls with 0 for missing values
        df = df.fillna({
            "avg_response_time_5min": 0,
            "max_response_time_5min": 0
        })
        
        return df
    
    def encode_categorical_features(self, df: DataFrame) -> DataFrame:
        """Encode categorical variables."""
        print("Encoding categorical features...")
        
        # Log level encoding
        log_level_mapping = {
            "DEBUG": 0,
            "INFO": 1,
            "WARN": 2,
            "ERROR": 3,
            "CRITICAL": 4
        }
        
        mapping_expr = F.create_map([F.lit(x) for x in sum(log_level_mapping.items(), ())])
        df = df.withColumn("log_level_encoded", mapping_expr[F.col("log_level")])
        
        # String indexer for log_type
        indexer = StringIndexer(inputCol="log_type", outputCol="log_type_indexed")
        indexer_model = indexer.fit(df)
        df = indexer_model.transform(df)
        
        return df
    
    def compute_statistical_features(self, df: DataFrame) -> DataFrame:
        """Compute statistical anomaly indicators."""
        print("Computing statistical features...")
        
        # Group statistics by log_type
        stats_df = df.groupBy("log_type").agg(
            F.mean("response_time_ms").alias("mean_response_time"),
            F.stddev("response_time_ms").alias("std_response_time"),
            F.percentile_approx("response_time_ms", 0.95).alias("p95_response_time")
        )
        
        df = df.join(stats_df, on="log_type", how="left")
        
        # Z-score for response time
        df = df.withColumn(
            "response_time_zscore",
            F.when(
                (F.col("response_time_ms").isNotNull()) & 
                (F.col("std_response_time") > 0),
                (F.col("response_time_ms") - F.col("mean_response_time")) / F.col("std_response_time")
            ).otherwise(0)
        )
        
        # Is response time above 95th percentile?
        df = df.withColumn(
            "is_slow_response",
            F.when(F.col("response_time_ms") > F.col("p95_response_time"), 1).otherwise(0)
        )
        
        return df
    
    def create_sequence_features(self, df: DataFrame) -> DataFrame:
        """Create features for sequence-based models (LSTM)."""
        print("Creating sequence features...")
        
        # Lag features
        window_spec = Window.partitionBy("log_type").orderBy("timestamp")
        
        for i in range(1, 4):  # Previous 3 time steps
            df = df.withColumn(
                f"response_time_lag_{i}",
                F.lag("response_time_ms", i).over(window_spec)
            )
            df = df.withColumn(
                f"count_lag_{i}",
                F.lag("count_5min", i).over(window_spec)
            )
        
        # Fill nulls in lag features with forward fill
        df = df.fillna(0)
        
        return df
    
    def select_features(self, df: DataFrame) -> DataFrame:
        """Select final feature set."""
        print("Selecting features for ML...")
        
        feature_columns = [
            # Identifiers
            "log_id",
            "timestamp",
            "log_type",
            
            # Temporal features
            "hour",
            "day_of_week",
            "is_weekend",
            "is_business_hours",
            
            # Encoded features
            "log_level_encoded",
            "log_type_indexed",
            
            # Rolling statistics
            "count_5min",
            "count_15min",
            "count_1hour",
            "avg_response_time_5min",
            "max_response_time_5min",
            
            # Statistical features
            "response_time_zscore",
            "is_slow_response",
            
            # Sequence features
            "response_time_lag_1",
            "response_time_lag_2",
            "response_time_lag_3",
            "count_lag_1",
            "count_lag_2",
            "count_lag_3",
            
            # Ground truth (for training)
            "is_anomaly",
            
            # Response time (for analysis)
            "response_time_ms",
            "message"
        ]
        
        # Select columns that exist
        available_cols = df.columns
        selected_cols = [col for col in feature_columns if col in available_cols]
        
        return df.select(*selected_cols)
    
    def run_pipeline(self, input_path: str, output_path: str):
        """Run the complete feature engineering pipeline."""
        print("=" * 80)
        print("FEATURE ENGINEERING PIPELINE")
        print("=" * 80)
        
        # Load data
        df = self.load_data(input_path)
        
        # Feature engineering steps
        df = self.extract_temporal_features(df)
        df = self.compute_rolling_statistics(df)
        df = self.encode_categorical_features(df)
        df = self.compute_statistical_features(df)
        df = self.create_sequence_features(df)
        df = self.select_features(df)
        
        # Cache for better performance
        df.cache()
        
        print(f"\nFinal feature count: {len(df.columns)}")
        print(f"Total records: {df.count()}")
        
        # Show sample
        print("\nSample of engineered features:")
        df.show(5, truncate=False)
        
        # Save to parquet
        print(f"\nSaving to {output_path}...")
        df.write.mode("overwrite").parquet(output_path)
        
        print("\n✓ Feature engineering complete!")
        
        return df


def main():
    """Main entry point."""
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("AnomalyDetection-FeatureEngineering") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    try:
        # Get latest raw data file
        raw_data_path = Path(config.raw_data_path)
        parquet_files = list(raw_data_path.glob("*.parquet"))
        
        if not parquet_files:
            print("Error: No data files found. Run log_generator.py first.")
            sys.exit(1)
        
        # Use most recent file
        input_file = str(max(parquet_files, key=lambda p: p.stat().st_mtime))
        output_path = f"{config.processed_data_path}/features_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"Input: {input_file}")
        print(f"Output: {output_path}")
        
        # Run pipeline
        engineer = FeatureEngineer(spark)
        engineer.run_pipeline(input_file, output_path)
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
