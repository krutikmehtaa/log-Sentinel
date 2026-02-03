"""
Synthetic log generator for anomaly detection system.

Generates realistic application logs with:
- Normal patterns (API calls, DB queries, user actions)
- Temporal patterns (peak hours, weekday/weekend)
- Anomalies (errors, spikes, unusual patterns)
"""

import random
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
from pathlib import Path
import click

from src.utils.config import config, ensure_directories


class LogGenerator:
    """Generate synthetic application logs."""
    
    LOG_TYPES = ["API_REQUEST", "DATABASE_QUERY", "USER_ACTION", "SYSTEM_EVENT", "ERROR"]
    LOG_LEVELS = ["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"]
    
    # Normal API endpoints
    API_ENDPOINTS = [
        "/api/v1/users",
        "/api/v1/products",
        "/api/v1/orders",
        "/api/v1/payments",
        "/api/v1/analytics",
        "/health",
    ]
    
    # Database operations
    DB_OPERATIONS = [
        "SELECT * FROM users WHERE id = ?",
        "SELECT * FROM products WHERE category = ?",
        "INSERT INTO orders VALUES (?, ?, ?)",
        "UPDATE inventory SET quantity = ?",
        "DELETE FROM cart WHERE user_id = ?",
    ]
    
    # Error messages
    ERROR_MESSAGES = [
        "Connection timeout to database",
        "Authentication failed for user",
        "Payment gateway unavailable",
        "Rate limit exceeded",
        "Invalid input parameters",
        "Internal server error",
    ]
    
    def __init__(self, start_date: datetime, num_logs: int, anomaly_ratio: float = 0.05):
        """
        Initialize log generator.
        
        Args:
            start_date: Starting timestamp for logs
            num_logs: Total number of logs to generate
            anomaly_ratio: Proportion of anomalous logs (default 5%)
        """
        self.start_date = start_date
        self.num_logs = num_logs
        self.anomaly_ratio = anomaly_ratio
        self.num_anomalies = int(num_logs * anomaly_ratio)
    
    def _generate_timestamp(self, base_time: datetime, log_idx: int) -> datetime:
        """Generate timestamp with realistic temporal patterns."""
        # Add sequential time progression (1-10 seconds between logs)
        time_delta = timedelta(seconds=random.uniform(1, 10))
        timestamp = base_time + (time_delta * log_idx)
        
        # Add peak hour patterns (higher activity 9am-5pm)
        hour = timestamp.hour
        if 9 <= hour <= 17:
            # Peak hours - more frequent logs
            return timestamp
        else:
            # Off-peak - add extra gaps
            return timestamp + timedelta(seconds=random.randint(5, 20))
    
    def _generate_normal_log(self, timestamp: datetime, log_idx: int) -> Dict[str, Any]:
        """Generate a normal log entry."""
        log_type = random.choices(
            self.LOG_TYPES,
            weights=[50, 25, 15, 8, 2],  # API requests most common
            k=1
        )[0]
        
        if log_type == "API_REQUEST":
            endpoint = random.choice(self.API_ENDPOINTS)
            status_code = random.choices([200, 201, 204, 400, 404], 
                                        weights=[85, 5, 5, 3, 2])[0]
            response_time = random.gauss(150, 50)  # ms, normal distribution
            
            message = f"{endpoint} returned {status_code} in {response_time:.0f}ms"
            log_level = "INFO" if status_code < 400 else "WARN"
            
            extra_data = {
                "endpoint": endpoint,
                "status_code": status_code,
                "response_time_ms": round(response_time, 2),
                "user_id": f"user_{random.randint(1, 1000)}",
            }
        
        elif log_type == "DATABASE_QUERY":
            query = random.choice(self.DB_OPERATIONS)
            execution_time = random.gauss(50, 20)  # ms
            
            message = f"Query executed in {execution_time:.0f}ms"
            log_level = "DEBUG"
            
            extra_data = {
                "query_type": query.split()[0],
                "execution_time_ms": round(execution_time, 2),
                "rows_affected": random.randint(1, 100),
            }
        
        elif log_type == "USER_ACTION":
            actions = ["login", "logout", "view_product", "add_to_cart", "checkout"]
            action = random.choice(actions)
            
            message = f"User performed action: {action}"
            log_level = "INFO"
            
            extra_data = {
                "action": action,
                "user_id": f"user_{random.randint(1, 1000)}",
                "session_id": f"session_{random.randint(1, 500)}",
            }
        
        else:  # SYSTEM_EVENT or ERROR (rare)
            if log_type == "ERROR":
                message = random.choice(self.ERROR_MESSAGES)
                log_level = "ERROR"
            else:
                message = "System health check completed"
                log_level = "INFO"
            
            extra_data = {
                "service": random.choice(["web", "api", "database", "cache"]),
            }
        
        return {
            "log_id": f"log_{log_idx}",
            "timestamp": timestamp.isoformat(),
            "log_level": log_level,
            "log_type": log_type,
            "message": message,
            "extra_data": extra_data,
        }
    
    def _generate_anomalous_log(self, timestamp: datetime, log_idx: int) -> Dict[str, Any]:
        """Generate an anomalous log entry."""
        anomaly_type = random.choice([
            "error_spike",
            "slow_response",
            "unusual_endpoint",
            "authentication_failure",
            "resource_exhaustion"
        ])
        
        if anomaly_type == "error_spike":
            message = random.choice(self.ERROR_MESSAGES)
            extra_data = {
                "status_code": random.choice([500, 502, 503, 504]),
                "error_count": random.randint(10, 100),
                "service": "api",
            }
            log_level = "ERROR"
            log_type = "ERROR"
        
        elif anomaly_type == "slow_response":
            endpoint = random.choice(self.API_ENDPOINTS)
            response_time = random.gauss(5000, 1000)  # Very slow
            
            message = f"{endpoint} returned 200 in {response_time:.0f}ms"
            extra_data = {
                "endpoint": endpoint,
                "status_code": 200,
                "response_time_ms": round(response_time, 2),
                "user_id": f"user_{random.randint(1, 1000)}",
            }
            log_level = "WARN"
            log_type = "API_REQUEST"
        
        elif anomaly_type == "unusual_endpoint":
            message = "/api/admin/delete_all returned 200 in 50ms"
            extra_data = {
                "endpoint": "/api/admin/delete_all",
                "status_code": 200,
                "response_time_ms": 50,
                "user_id": f"user_{random.randint(1, 1000)}",
            }
            log_level = "WARN"
            log_type = "API_REQUEST"
        
        elif anomaly_type == "authentication_failure":
            message = "Multiple authentication failures detected"
            extra_data = {
                "failed_attempts": random.randint(10, 50),
                "user_id": f"user_{random.randint(1, 1000)}",
                "ip_address": f"192.168.{random.randint(0, 255)}.{random.randint(0, 255)}",
            }
            log_level = "ERROR"
            log_type = "ERROR"
        
        else:  # resource_exhaustion
            message = "High memory usage detected"
            extra_data = {
                "memory_usage_percent": random.uniform(85, 99),
                "cpu_usage_percent": random.uniform(80, 95),
                "service": random.choice(["web", "api", "database"]),
            }
            log_level = "CRITICAL"
            log_type = "SYSTEM_EVENT"
        
        return {
            "log_id": f"log_{log_idx}",
            "timestamp": timestamp.isoformat(),
            "log_level": log_level,
            "log_type": log_type,
            "message": message,
            "extra_data": extra_data,
            "is_anomaly": True,  # Ground truth label
        }
    
    def generate(self) -> pd.DataFrame:
        """Generate complete dataset of logs."""
        logs = []
        
        # Generate indices for anomalies
        anomaly_indices = set(random.sample(range(self.num_logs), self.num_anomalies))
        
        current_time = self.start_date
        
        for i in range(self.num_logs):
            timestamp = self._generate_timestamp(current_time, i)
            current_time = timestamp
            
            if i in anomaly_indices:
                log = self._generate_anomalous_log(timestamp, i)
            else:
                log = self._generate_normal_log(timestamp, i)
                log["is_anomaly"] = False
            
            logs.append(log)
        
        df = pd.DataFrame(logs)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Expand extra_data into columns
        extra_data_df = pd.json_normalize(df['extra_data'])
        df = pd.concat([df.drop('extra_data', axis=1), extra_data_df], axis=1)
        
        return df


@click.command()
@click.option('--num-logs', default=100000, help='Number of logs to generate')
@click.option('--anomaly-ratio', default=0.05, help='Ratio of anomalous logs')
@click.option('--days-ago', default=7, help='Start date (days ago)')
@click.option('--output', default=None, help='Output file path')
def main(num_logs: int, anomaly_ratio: float, days_ago: int, output: str):
    """Generate synthetic log data."""
    ensure_directories()
    
    # Set start date
    start_date = datetime.now() - timedelta(days=days_ago)
    
    print(f"Generating {num_logs} logs starting from {start_date}...")
    print(f"Anomaly ratio: {anomaly_ratio * 100}%")
    
    # Generate logs
    generator = LogGenerator(start_date, num_logs, anomaly_ratio)
    df = generator.generate()
    
    # Save to parquet
    if output is None:
        output = f"{config.raw_data_path}/logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    
    print(f"\n✓ Generated {len(df)} logs")
    print(f"✓ Anomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.2f}%)")
    print(f"✓ Saved to: {output}")
    print(f"\nLog type distribution:")
    print(df['log_type'].value_counts())


if __name__ == "__main__":
    main()
