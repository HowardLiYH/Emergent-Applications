#!/bin/bash
# One-click reproduction script

echo "=================================="
echo "SI Trading Analysis - Reproduction"
echo "=================================="

# Build Docker image
echo "Building Docker image..."
docker build -t si-trading .

# Run analysis
echo "Running analysis..."
docker run -v $(pwd)/results:/app/results -v $(pwd)/data:/app/data:ro si-trading

echo "=================================="
echo "Results saved to ./results/"
echo "=================================="
