"""
Configuration utilities for the anomaly detection system.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration loader and manager."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.load_config()
    
    def load_config(self, config_path: str = None):
        """Load configuration from YAML file."""
        if config_path is None:
            # Default to config/config.yaml
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        # Override with environment variables if present
        self._override_from_env()
    
    def _override_from_env(self):
        """Override config values from environment variables."""
        env_mappings = {
            'DB_HOST': ('database', 'host'),
            'DB_PORT': ('database', 'port'),
            'DB_NAME': ('database', 'name'),
            'DB_USER': ('database', 'user'),
            'DB_PASSWORD': ('database', 'password'),
            'SPARK_MASTER': ('spark', 'master'),
            'MLFLOW_TRACKING_URI': ('paths', 'mlflow_tracking'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if section in self._config:
                    self._config[section][key] = value
    
    def get(self, *keys, default=None) -> Any:
        """Get configuration value by nested keys."""
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value
    
    def get_database_url(self) -> str:
        """Get PostgreSQL connection URL."""
        db_config = self._config['database']
        return (f"postgresql://{db_config['user']}:{db_config['password']}"
                f"@{db_config['host']}:{db_config['port']}/{db_config['name']}")
    
    def get_spark_config(self) -> Dict[str, str]:
        """Get Spark configuration dictionary."""
        spark_config = self._config['spark']
        return {
            'spark.app.name': spark_config['app_name'],
            'spark.driver.memory': spark_config['driver_memory'],
            'spark.executor.memory': spark_config['executor_memory'],
        }
    
    @property
    def raw_data_path(self) -> str:
        return self._config['paths']['raw_data']
    
    @property
    def processed_data_path(self) -> str:
        return self._config['paths']['processed_data']
    
    @property
    def models_path(self) -> str:
        return self._config['paths']['models']
    
    @property
    def anomalies_path(self) -> str:
        return self._config['paths']['anomalies']


# Singleton instance
config = Config()


def ensure_directories():
    """Ensure all required directories exist."""
    paths = [
        config.raw_data_path,
        config.processed_data_path,
        config.models_path,
        config.anomalies_path,
        'mlruns',
    ]
    
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Test configuration loading
    ensure_directories()
    print("Configuration loaded successfully:")
    print(f"Database URL: {config.get_database_url()}")
    print(f"Raw data path: {config.raw_data_path}")
    print(f"Spark config: {config.get_spark_config()}")
