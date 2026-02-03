#!/bin/bash

# Quick Start Script for Anomaly Detection System
# This script sets up the entire environment and runs a sample pipeline

set -e  # Exit on error

echo "=============================================================================="
echo "  ANOMALY DETECTION SYSTEM - QUICK START"
echo "=============================================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi
print_status "Python 3 found"

if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
    exit 1
fi
print_status "Docker found"

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed"
    exit 1
fi
print_status "Docker Compose found"

# Check Java for Spark
if ! command -v java &> /dev/null; then
    print_warning "Java not found. Spark will not work without Java 11+"
else
    print_status "Java found"
fi

echo ""
echo "=============================================================================="
echo "  STEP 1: Environment Setup"
echo "=============================================================================="

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
print_status "Virtual environment activated"

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip setuptools wheel -q
pip install -r requirements.txt -q
print_status "Dependencies installed"

echo ""
echo "=============================================================================="
echo "  STEP 2: Infrastructure Setup"
echo "=============================================================================="

# Start Docker containers
echo "Starting PostgreSQL and MLflow..."
docker-compose up -d postgres mlflow

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
sleep 5

# Check if PostgreSQL is running
if docker-compose ps postgres | grep -q "Up"; then
    print_status "PostgreSQL is running"
else
    print_error "PostgreSQL failed to start"
    exit 1
fi

# Check if MLflow is running
if docker-compose ps mlflow | grep -q "Up"; then
    print_status "MLflow is running"
else
    print_warning "MLflow failed to start (optional)"
fi

echo ""
echo "=============================================================================="
echo "  STEP 3: Generate Sample Data"
echo "=============================================================================="

echo "Generating 100,000 synthetic log entries..."
python src/ingestion/log_generator.py --num-logs 100000 --anomaly-ratio 0.05

print_status "Sample data generated"

echo ""
echo "=============================================================================="
echo "  STEP 4: Feature Engineering (PySpark)"
echo "=============================================================================="

echo "Running feature engineering pipeline..."
spark-submit \
    --master local[*] \
    --driver-memory 4g \
    src/processing/feature_engineering.py

print_status "Feature engineering complete"

echo ""
echo "=============================================================================="
echo "  STEP 5: Train ML Models"
echo "=============================================================================="

echo "Training Isolation Forest..."
python src/models/train_isolation_forest.py

print_status "Isolation Forest trained"

echo ""
echo "Training LSTM Autoencoder (this may take a few minutes)..."
python src/models/train_lstm_autoencoder.py

print_status "LSTM Autoencoder trained"

echo ""
echo "=============================================================================="
echo "  STEP 6: Batch Inference"
echo "=============================================================================="

echo "Running batch inference..."
spark-submit \
    --master local[*] \
    --driver-memory 4g \
    src/processing/batch_inference.py

print_status "Batch inference complete"

echo ""
echo "=============================================================================="
echo "  SETUP COMPLETE!"
echo "=============================================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. View MLflow tracking:"
echo "   Open http://localhost:5000 in your browser"
echo ""
echo "2. Query anomalies in PostgreSQL:"
echo "   docker exec -it anomaly_postgres psql -U postgres -d anomaly_detection"
echo "   SELECT * FROM anomalies ORDER BY anomaly_score DESC LIMIT 10;"
echo ""
echo "3. Explore the Jupyter notebooks:"
echo "   jupyter notebook notebooks/"
echo ""
echo "4. (Optional) Start Airflow:"
echo "   docker-compose up -d airflow"
echo "   Open http://localhost:8080 (admin/admin)"
echo ""
echo "5. Re-run the pipeline:"
echo "   ./quickstart.sh"
echo ""
print_status "All done! Your anomaly detection system is ready."
