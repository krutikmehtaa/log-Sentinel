/**
 * Scala Utility for Log Processing
 * 
 * This utility demonstrates Scala/Spark integration for high-performance
 * data processing in the anomaly detection pipeline.
 * 
 * Compile with: scalac -classpath "$(find ~/.ivy2 -name 'spark-sql*.jar'):$(find ~/.ivy2 -name 'scala-library*.jar')" LogProcessor.scala
 * Run with: spark-submit --class LogProcessor target/log-processor.jar
 */

package com.anomalydetection

import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types._

object LogProcessor {
  
  /**
   * Advanced log parsing and enrichment
   */
  def parseAndEnrichLogs(df: DataFrame): DataFrame = {
    println("Parsing and enriching logs...")
    
    // Extract HTTP status code patterns
    val statusCodePattern = "\\b(\\d{3})\\b"
    
    val enrichedDF = df
      .withColumn("extracted_status_code", 
        regexp_extract(col("message"), statusCodePattern, 1))
      .withColumn("is_error_status", 
        when(col("extracted_status_code").cast("int") >= 400, 1)
          .otherwise(0))
      
      // Calculate inter-arrival time
      .withColumn("prev_timestamp", 
        lag(col("timestamp"), 1)
          .over(Window.partitionBy("log_type").orderBy("timestamp")))
      .withColumn("inter_arrival_seconds",
        when(col("prev_timestamp").isNotNull,
          unix_timestamp(col("timestamp")) - unix_timestamp(col("prev_timestamp")))
          .otherwise(0))
    
    enrichedDF
  }
  
  /**
   * Compute advanced statistical features
   */
  def computeAdvancedFeatures(df: DataFrame): DataFrame = {
    println("Computing advanced statistical features...")
    
    // Define time windows
    val hourWindow = Window
      .partitionBy("log_type")
      .orderBy(col("timestamp").cast("long"))
      .rangeBetween(-3600, 0)
    
    // Exponentially weighted moving average
    val alpha = 0.3
    val ewmaWindow = Window
      .partitionBy("log_type")
      .orderBy("timestamp")
      .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    
    df
      .withColumn("count_per_hour", count("*").over(hourWindow))
      .withColumn("variance_response_time", 
        variance("response_time_ms").over(hourWindow))
      .withColumn("coefficient_variation",
        when(avg("response_time_ms").over(hourWindow) =!= 0,
          sqrt(col("variance_response_time")) / avg("response_time_ms").over(hourWindow))
          .otherwise(0))
      
      // Entropy of log levels (diversity measure)
      .withColumn("log_level_entropy", 
        -sum(
          when(col("log_level").isNotNull, 1.0 / count("*").over(hourWindow))
            .otherwise(0) * 
          log2(1.0 / count("*").over(hourWindow))
        ).over(hourWindow))
  }
  
  /**
   * Detect burst patterns
   */
  def detectBursts(df: DataFrame, threshold: Double = 2.0): DataFrame = {
    println("Detecting burst patterns...")
    
    val burstWindow = Window
      .partitionBy("log_type")
      .orderBy("timestamp")
      .rowsBetween(-10, 0)
    
    df
      .withColumn("recent_count", count("*").over(burstWindow))
      .withColumn("avg_count", 
        avg(col("recent_count")).over(Window.partitionBy("log_type")))
      .withColumn("std_count",
        stddev(col("recent_count")).over(Window.partitionBy("log_type")))
      .withColumn("is_burst",
        when(
          col("recent_count") > (col("avg_count") + threshold * col("std_count")),
          1
        ).otherwise(0))
  }
  
  /**
   * Aggregate anomaly scores
   */
  def aggregateAnomalyScores(df: DataFrame): DataFrame = {
    println("Aggregating anomaly scores...")
    
    df
      .groupBy(
        window(col("timestamp"), "1 hour"),
        col("log_type")
      )
      .agg(
        count("*").alias("log_count"),
        avg("anomaly_score").alias("avg_anomaly_score"),
        max("anomaly_score").alias("max_anomaly_score"),
        sum(when(col("is_anomaly_predicted") === 1, 1).otherwise(0))
          .alias("anomaly_count"),
        countDistinct("user_id").alias("unique_users")
      )
      .orderBy(desc("max_anomaly_score"))
  }
  
  /**
   * Main processing pipeline
   */
  def processPipeline(
    spark: SparkSession, 
    inputPath: String, 
    outputPath: String
  ): Unit = {
    
    println("=" * 80)
    println("SCALA LOG PROCESSOR")
    println("=" * 80)
    
    // Read data
    println(s"Reading from: $inputPath")
    var df = spark.read.parquet(inputPath)
    
    // Processing steps
    df = parseAndEnrichLogs(df)
    df = computeAdvancedFeatures(df)
    df = detectBursts(df)
    
    // Cache for multiple operations
    df.cache()
    
    // Show statistics
    println("\nProcessing Statistics:")
    println(s"Total records: ${df.count()}")
    println("\nBurst detection summary:")
    df.groupBy("is_burst")
      .count()
      .show()
    
    // Save results
    println(s"\nWriting to: $outputPath")
    df.write
      .mode("overwrite")
      .partitionBy("log_type")
      .parquet(outputPath)
    
    // Generate summary report
    val summaryDF = aggregateAnomalyScores(df)
    println("\nTop anomaly windows:")
    summaryDF.show(10, truncate = false)
    
    println("\n✓ Processing complete!")
  }
  
  /**
   * Main entry point
   */
  def main(args: Array[String]): Unit = {
    // Initialize Spark
    val spark = SparkSession.builder()
      .appName("AnomalyDetection-ScalaProcessor")
      .config("spark.driver.memory", "4g")
      .config("spark.sql.shuffle.partitions", "8")
      .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    try {
      val inputPath = if (args.length > 0) args(0) else "data/processed/features_*"
      val outputPath = if (args.length > 1) args(1) else "data/processed/scala_enriched"
      
      processPipeline(spark, inputPath, outputPath)
      
    } catch {
      case e: Exception =>
        println(s"Error: ${e.getMessage}")
        e.printStackTrace()
        System.exit(1)
    } finally {
      spark.stop()
    }
  }
}

/**
 * Helper object for utility functions
 */
object FeatureUtils {
  
  /**
   * Custom UDF for complex feature computation
   */
  def computeEntropy(values: Seq[String]): Double = {
    if (values.isEmpty) return 0.0
    
    val counts = values.groupBy(identity).mapValues(_.size.toDouble)
    val total = values.size.toDouble
    
    -counts.values.map { count =>
      val p = count / total
      p * math.log(p) / math.log(2)
    }.sum
  }
  
  /**
   * Seasonal decomposition helper
   */
  def isSeasonalAnomaly(
    value: Double,
    historicalMean: Double,
    historicalStd: Double,
    threshold: Double = 3.0
  ): Boolean = {
    if (historicalStd == 0) return false
    val zScore = math.abs((value - historicalMean) / historicalStd)
    zScore > threshold
  }
}

/**
 * Case classes for typed data structures
 */
case class LogRecord(
  log_id: String,
  timestamp: java.sql.Timestamp,
  log_level: String,
  log_type: String,
  message: String,
  anomaly_score: Option[Double]
)

case class AnomalySummary(
  window_start: java.sql.Timestamp,
  window_end: java.sql.Timestamp,
  log_type: String,
  total_logs: Long,
  anomaly_count: Long,
  avg_anomaly_score: Double,
  max_anomaly_score: Double
)
