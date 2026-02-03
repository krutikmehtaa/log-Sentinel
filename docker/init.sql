-- Initialize Database Schema for Anomaly Detection System

-- Create logs table (metadata)
CREATE TABLE IF NOT EXISTS logs_metadata (
    id SERIAL PRIMARY KEY,
    batch_id VARCHAR(100),
    total_logs INTEGER,
    start_timestamp TIMESTAMP,
    end_timestamp TIMESTAMP,
    processing_time_seconds FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create anomalies table
CREATE TABLE IF NOT EXISTS anomalies (
    id SERIAL PRIMARY KEY,
    log_id VARCHAR(100) UNIQUE,
    timestamp TIMESTAMP NOT NULL,
    log_level VARCHAR(20),
    log_type VARCHAR(50),
    message TEXT,
    anomaly_score FLOAT,
    model_name VARCHAR(50),
    model_version VARCHAR(20),
    features JSONB,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_confirmed BOOLEAN DEFAULT FALSE,
    feedback VARCHAR(20),  -- 'true_positive', 'false_positive', 'unknown'
    notes TEXT
);

-- Create model_performance table
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50),
    model_version VARCHAR(20),
    metric_name VARCHAR(50),
    metric_value FLOAT,
    evaluation_date DATE,
    dataset_size INTEGER,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create feature_statistics table for monitoring drift
CREATE TABLE IF NOT EXISTS feature_statistics (
    id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100),
    stat_type VARCHAR(20),  -- 'mean', 'std', 'min', 'max', 'percentile_95'
    stat_value FLOAT,
    calculation_date DATE,
    sample_size INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    anomaly_id INTEGER REFERENCES anomalies(id),
    severity VARCHAR(20),  -- 'low', 'medium', 'high', 'critical'
    alert_type VARCHAR(50),
    description TEXT,
    status VARCHAR(20) DEFAULT 'open',  -- 'open', 'acknowledged', 'resolved'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP,
    resolved_at TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_anomalies_timestamp ON anomalies(timestamp);
CREATE INDEX IF NOT EXISTS idx_anomalies_model ON anomalies(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_anomalies_score ON anomalies(anomaly_score);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);

-- Create view for recent high-severity anomalies
CREATE OR REPLACE VIEW recent_critical_anomalies AS
SELECT 
    a.id,
    a.timestamp,
    a.log_type,
    a.message,
    a.anomaly_score,
    a.model_name,
    al.severity,
    al.status
FROM anomalies a
JOIN alerts al ON a.id = al.anomaly_id
WHERE al.severity IN ('high', 'critical')
  AND al.status = 'open'
  AND a.detected_at > NOW() - INTERVAL '24 hours'
ORDER BY a.anomaly_score DESC;

-- Insert sample model performance records
INSERT INTO model_performance (model_name, model_version, metric_name, metric_value, evaluation_date, dataset_size)
VALUES 
    ('isolation_forest', 'v1.0', 'precision', 0.85, CURRENT_DATE, 10000),
    ('isolation_forest', 'v1.0', 'recall', 0.78, CURRENT_DATE, 10000),
    ('isolation_forest', 'v1.0', 'f1_score', 0.81, CURRENT_DATE, 10000);

COMMENT ON TABLE anomalies IS 'Stores detected anomalies with scores and metadata';
COMMENT ON TABLE model_performance IS 'Tracks model performance metrics over time';
COMMENT ON TABLE feature_statistics IS 'Monitors feature distributions for drift detection';
COMMENT ON TABLE alerts IS 'Manages anomaly alerts and their lifecycle';
