"""
LSTM Autoencoder for Sequence-based Anomaly Detection.

Uses PyTorch to build an LSTM autoencoder that learns normal patterns.
Anomalies are detected based on reconstruction error.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import mlflow
import mlflow.pytorch
from datetime import datetime
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config
from src.utils.db_utils import db_manager


class TimeSeriesDataset(Dataset):
    """Dataset for sequence-based anomaly detection."""
    
    def __init__(self, sequences, labels=None):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels) if labels is not None else None
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        return self.sequences[idx]


class LSTMAutoencoder(nn.Module):
    """LSTM-based Autoencoder for anomaly detection."""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMAutoencoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        # Encode
        _, (hidden, cell) = self.encoder(x)
        
        # Decode
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
        
        # Reconstruct
        reconstruction = self.output_layer(decoder_output)
        
        return reconstruction


class LSTMDetector:
    """LSTM Autoencoder-based anomaly detector."""
    
    def __init__(self, sequence_length=50, hidden_size=64, num_layers=2, 
                 dropout=0.2, learning_rate=0.001, batch_size=32, epochs=20):
        """
        Initialize LSTM detector.
        
        Args:
            sequence_length: Length of input sequences
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.feature_names = None
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load feature-engineered data."""
        print(f"Loading data from {data_path}...")
        df = pd.read_parquet(data_path)
        print(f"Loaded {len(df)} records")
        return df
    
    def create_sequences(self, data: np.ndarray, labels: np.ndarray = None) -> tuple:
        """Create sequences for LSTM input."""
        print(f"Creating sequences of length {self.sequence_length}...")
        
        sequences = []
        sequence_labels = []
        
        for i in range(len(data) - self.sequence_length + 1):
            seq = data[i:i + self.sequence_length]
            sequences.append(seq)
            
            if labels is not None:
                # Label is anomaly if ANY point in sequence is anomaly
                label = int(np.any(labels[i:i + self.sequence_length]))
                sequence_labels.append(label)
        
        sequences = np.array(sequences)
        sequence_labels = np.array(sequence_labels) if labels is not None else None
        
        print(f"Created {len(sequences)} sequences")
        if sequence_labels is not None:
            print(f"Anomalies: {sequence_labels.sum()} ({sequence_labels.mean()*100:.2f}%)")
        
        return sequences, sequence_labels
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for training."""
        print("Preparing features...")
        
        # Select numerical features
        feature_cols = [
            'hour',
            'day_of_week',
            'log_level_encoded',
            'log_type_indexed',
            'count_5min',
            'avg_response_time_5min',
            'max_response_time_5min',
            'response_time_zscore',
        ]
        
        self.feature_names = [col for col in feature_cols if col in df.columns]
        
        # Sort by timestamp for sequential data
        df = df.sort_values('timestamp')
        
        X = df[self.feature_names].fillna(0).values
        y = df['is_anomaly'].values if 'is_anomaly' in df.columns else None
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y)
        
        print(f"Feature shape: {X_seq.shape}")
        
        return X_seq, y_seq
    
    def train(self, X: np.ndarray, y: np.ndarray = None):
        """Train the LSTM Autoencoder."""
        print("\n" + "=" * 80)
        print("TRAINING LSTM AUTOENCODER")
        print("=" * 80)
        
        # Initialize model
        input_size = X.shape[2]
        self.model = LSTMAutoencoder(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        print(f"\nModel architecture:")
        print(self.model)
        print(f"\nTotal parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        # Create data loaders
        dataset = TimeSeriesDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        train_losses = []
        
        print(f"\nTraining for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            epoch_loss = 0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for batch in progress_bar:
                if y is not None:
                    sequences, labels = batch
                else:
                    sequences = batch
                
                sequences = sequences.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                reconstructed = self.model(sequences)
                loss = criterion(reconstructed, sequences)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_loss = epoch_loss / len(dataloader)
            train_losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{self.epochs} - Average Loss: {avg_loss:.6f}")
        
        # Set threshold based on reconstruction error
        self.model.eval()
        with torch.no_grad():
            all_errors = []
            for batch in DataLoader(dataset, batch_size=self.batch_size):
                if y is not None:
                    sequences, _ = batch
                else:
                    sequences = batch
                
                sequences = sequences.to(self.device)
                reconstructed = self.model(sequences)
                
                # Compute reconstruction error
                errors = torch.mean((sequences - reconstructed) ** 2, dim=(1, 2))
                all_errors.extend(errors.cpu().numpy())
            
            all_errors = np.array(all_errors)
            self.threshold = np.percentile(all_errors, 95)  # 95th percentile
            print(f"\nAnomaly threshold set to: {self.threshold:.6f}")
        
        # Evaluate if ground truth available
        if y is not None:
            predictions = self.predict(X)
            self._evaluate(y, predictions)
        
        print("\n✓ Training complete!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies based on reconstruction error."""
        self.model.eval()
        
        dataset = TimeSeriesDataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        
        predictions = []
        
        with torch.no_grad():
            for sequences in dataloader:
                sequences = sequences.to(self.device)
                reconstructed = self.model(sequences)
                
                # Compute reconstruction error
                errors = torch.mean((sequences - reconstructed) ** 2, dim=(1, 2))
                
                # Threshold
                pred = (errors > self.threshold).cpu().numpy().astype(int)
                predictions.extend(pred)
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores based on reconstruction error."""
        self.model.eval()
        
        dataset = TimeSeriesDataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        
        scores = []
        
        with torch.no_grad():
            for sequences in dataloader:
                sequences = sequences.to(self.device)
                reconstructed = self.model(sequences)
                
                # Compute reconstruction error
                errors = torch.mean((sequences - reconstructed) ** 2, dim=(1, 2))
                scores.extend(errors.cpu().numpy())
        
        scores = np.array(scores)
        # Normalize to [0, 1]
        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return scores_normalized
    
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
        
        # Log to database
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }
        
        db_manager.record_model_performance(
            model_name='lstm_autoencoder',
            model_version='v1.0',
            metrics=metrics,
            dataset_size=len(y_true)
        )
        
        return metrics
    
    def save_model(self, output_path: str):
        """Save model and preprocessing objects."""
        print(f"\nSaving model to {output_path}...")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), f"{output_path}_model.pth")
        
        # Save other components
        metadata = {
            'input_size': self.model.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'sequence_length': self.sequence_length,
            'threshold': self.threshold,
        }
        
        joblib.dump(metadata, f"{output_path}_metadata.pkl")
        joblib.dump(self.scaler, f"{output_path}_scaler.pkl")
        joblib.dump(self.feature_names, f"{output_path}_features.pkl")
        
        print("✓ Model saved!")
    
    def load_model(self, model_path: str):
        """Load trained model."""
        print(f"Loading model from {model_path}...")
        
        # Load metadata
        metadata = joblib.load(f"{model_path}_metadata.pkl")
        self.scaler = joblib.load(f"{model_path}_scaler.pkl")
        self.feature_names = joblib.load(f"{model_path}_features.pkl")
        
        # Recreate model
        self.model = LSTMAutoencoder(
            input_size=metadata['input_size'],
            hidden_size=metadata['hidden_size'],
            num_layers=metadata['num_layers'],
            dropout=metadata['dropout']
        ).to(self.device)
        
        # Load state
        self.model.load_state_dict(torch.load(f"{model_path}_model.pth"))
        self.threshold = metadata['threshold']
        self.sequence_length = metadata['sequence_length']
        
        self.model.eval()
        print("✓ Model loaded!")


def main():
    """Main training script."""
    print("=" * 80)
    print("LSTM AUTOENCODER ANOMALY DETECTION TRAINING")
    print("=" * 80)
    
    # Start MLflow run
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("anomaly_detection")
    
    with mlflow.start_run(run_name=f"lstm_autoencoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Get latest feature data
        processed_path = Path(config.processed_data_path)
        feature_files = list(processed_path.glob("features_*"))
        
        if not feature_files:
            print("Error: No feature files found. Run feature_engineering.py first.")
            sys.exit(1)
        
        input_path = str(max(feature_files, key=lambda p: p.stat().st_mtime))
        print(f"Using features from: {input_path}\n")
        
        # Initialize detector
        lstm_config = config.get('models', 'lstm_autoencoder')
        detector = LSTMDetector(
            sequence_length=lstm_config['sequence_length'],
            hidden_size=lstm_config['hidden_size'],
            num_layers=lstm_config['num_layers'],
            dropout=lstm_config['dropout'],
            learning_rate=lstm_config['learning_rate'],
            batch_size=lstm_config['batch_size'],
            epochs=lstm_config['epochs']
        )
        
        # Load and prepare data
        df = detector.load_data(input_path)
        X, y = detector.prepare_features(df)
        
        # Log parameters
        mlflow.log_param("sequence_length", detector.sequence_length)
        mlflow.log_param("hidden_size", detector.hidden_size)
        mlflow.log_param("num_layers", detector.num_layers)
        mlflow.log_param("n_sequences", X.shape[0])
        
        # Train model
        detector.train(X, y)
        
        # Save model
        model_path = f"{config.models_path}/lstm_autoencoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        detector.save_model(model_path)
        
        # Log model to MLflow
        mlflow.pytorch.log_model(detector.model, "model")
        
        print(f"\n✓ Training complete!")
        print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
