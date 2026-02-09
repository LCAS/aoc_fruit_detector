#!/bin/bash

# Deployment script for AOC Fruit Detector Nuclio Function

set -e

echo "AOC Fruit Detector Nuclio Function Deployment Script"
echo "=================================================="

# Configuration
FUNCTION_NAME="aoc-fruit-detector"
NAMESPACE="nuclio"
IMAGE_TAG="aoc-fruit-detector:latest"

# Check if nuclio CLI is installed
if ! command -v nuctl &> /dev/null; then
    echo "Error: nuclio CLI (nuctl) is not installed"
    echo "Please install nuclio CLI first: https://nuclio.io/docs/latest/setup/k8s/getting-started-k8s/"
    exit 1
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not available"
    exit 1
fi

echo "Building Docker image..."
docker build -t $IMAGE_TAG .

echo "Deploying nuclio function..."
nuctl deploy $FUNCTION_NAME \
    --namespace $NAMESPACE \
    --path . \
    --file function.yaml \
    --registry "" \
    --run-image $IMAGE_TAG

echo "Checking function status..."
nuctl get function $FUNCTION_NAME --namespace $NAMESPACE

echo ""
echo "Deployment completed!"
echo "Function endpoint should be available at the URL shown above."
echo ""
echo "To test the function, you can use:"
echo "curl -X POST -H 'Content-Type: application/json' \\"
echo "  -d '{\"image\": \"<base64_encoded_image>\"}' \\"
echo "  http://<function-url>"