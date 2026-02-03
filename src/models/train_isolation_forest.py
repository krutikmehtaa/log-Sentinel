"""
Isolation Forest Anomaly Detection Model.

Uses scikit-learn's Isolation Forest for unsupervised anomaly detection.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config
from src.utils.db_utils import db_manager


class IsolationForestDetector:
    """Isolation Forest-based anomaly detector."""
    
    def __init__(self, contamination=0.05, n_estimators=100, random_state=42):
        """
        Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees
            random_state: Random seed
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load feature-engineered data."""
        print(f"Loading data from {data_path}...")
        df = pd.read_parquet(data_path)
        print(f"Loaded {len(df)} records")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for training."""
        print("Preparing features...")
        
        # Select numerical features for training
        feature_cols = [
            'hour',
            'day_of_week',
            'is_weekend',
            'is_business_hours',
            'log_level_encoded',
            'log_type_indexed',
            'count_5min',
            'count_15min',
            'count_1hour',
            'avg_response_time_5min',
            'max_response_time_5min',
            'response_time_zscore',
            'is_slow_response',
            'response_time_lag_1',
            'response_time_lag_2',
            'response_time_lag_3',
            'count_lag_1',
            'count_lag_2',
            'count_lag_3',
        ]
        
        # Filter to existing columns
        self.feature_names = [col for col in feature_cols if col in df.columns]
        
        X = df[self.feature_names].fillna(0).values
        y = df['is_anomaly'].values if 'is_anomaly' in df.columns else None
        
        print(f"Feature shape: {X.shape}")
        print(f"Features: {self.feature_names}")
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray = None):
        """Train the Isolation Forest model."""
        print("\n" + "=" * 80)
        print("TRAINING ISOLATION FOREST")
        print("=" * 80)
        
        # Scale features
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        print(f"\nTraining with {len(X)} samples...")
        self.model.fit(X_scaled)
        
        # Evaluate if ground truth available
        if y is not None:
            predictions = self.predict(X)
            self._evaluate(y, predictions)
        
        print("\n✓ Training complete!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        # Convert -1 (anomaly) to 1, and 1 (normal) to 0
        return (predictions == -1).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores."""
        X_scaled = self.scaler.transform(X)
        # decision_function returns anomaly scores (lower = more anomalous)
        scores = self.model.decision_function(X_scaled)
        # Normalize to [0, 1] where 1 = most anomalous
        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min())
        return 1 - scores_normalized  # Invert so 1 = anomaly
    
    def _evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Evaluate model performance."""
        print("\n" + "-" * 80)
        print("MODEL EVALUATION")
        print("-" * 80)
        
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Normal  Anomaly")
        print(f"Actual Normal   {cm[0][0]:6d}   {cm[0][1]:6d}")
        print(f"       Anomaly  {cm[1][0]:6d}   {cm[1][1]:6d}")
        
        # Log to database
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }
        
        db_manager.record_model_performance(
            model_name='isolation_forest',
            model_version='v1.0',
            metrics=metrics,
            dataset_size=len(y_true)
        )
        
        return metrics
    
    def save_model(self, output_path: str):
        """Save model and scaler."""
        print(f"\nSaving model to {output_path}...")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model components
        joblib.dump(self.model, f"{output_path}_model.pkl")
        joblib.dump(self.scaler, f"{output_path}_scaler.pkl")
        joblib.dump(self.feature_names, f"{output_path}_features.pkl")
        
        print("✓ Model saved!")
    
    def load_model(self, model_path: str):
        """Load trained model."""
        print(f"Loading model from {model_path}...")
        
        self.model = joblib.load(f"{model_path}_model.pkl")
        self.scaler = joblib.load(f"{model_path}_scaler.pkl")
        self.feature_names = joblib.load(f"{model_path}_features.pkl")
        
        print("✓ Model loaded!")


def main():
    """Main training script."""
    print("=" * 80)
    print("ISOLATION FOREST ANOMALY DETECTION TRAINING")
    print("=" * 80)
    
    # Start MLflow run
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("anomaly_detection")
    
    with mlflow.start_run(run_name=f"isolation_forest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Get latest feature data
        processed_path = Path(config.processed_data_path)
        feature_files = list(processed_path.glob("features_*"))
        
        if not feature_files:
            print("Error: No feature files found. Run feature_engineering.py first.")
            sys.exit(1)
        
        # Use most recent file
        input_path = str(max(feature_files, key=lambda p: p.stat().st_mtime))
        print(f"Using features from: {input_path}\n")
        
        # Initialize detector
        detector = IsolationForestDetector(
            contamination=config.get('models', 'isolation_forest', 'contamination'),
            n_estimators=config.get('models', 'isolation_forest', 'n_estimators'),
            random_state=config.get('models', 'isolation_forest', 'random_state')
        )
        
        # Load and prepare data
        df = detector.load_data(input_path)
        X, y = detector.prepare_features(df)
        
        # Log parameters
        mlflow.log_param("n_estimators", detector.n_estimators)
        mlflow.log_param("contamination", detector.contamination)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_samples", X.shape[0])
        
        # Train model
        detector.train(X, y)
        
        # Save model
        model_path = f"{config.models_path}/isolation_forest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        detector.save_model(model_path)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(detector.model, "model")
        mlflow.log_artifact(f"{model_path}_scaler.pkl")
        
        print(f"\n✓ Training complete!")
        print(f"Model saved to: {model_path}")
        print(f"MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
