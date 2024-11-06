#!/bin/bash

# Exit if a command exists with a non-zero status
set -e

# Load environment variables from .env file
# Request the .env file from code owner
if [ -f .env ]; then
  set -a
  source .env
  set +a
else
    echo "Error: .env file not found"
    exit 1
fi

# Check if the required environment variables are set
if [ -z "$PROJECT_ID" ] || [ -z "$REGION" ] || [ -z "$MEMORY" ]; then
    echo "Error: Required environment variables are not set"
    exit 1
fi

echo "Project ID=$PROJECT_ID"
echo "Region=$REGION"
echo "Memory allocation=$MEMORY"

# Variables
DOCKER_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/brookyapi-repo/brooky-api"

# Deploy API to Cloud Run

echo "Building docker image for brooky-api"
docker build \
    -t $DOCKER_IMAGE \
    -f brookyapi/dockerfile .

echo "Pushing docker image to Google Container Registry"
docker push $DOCKER_IMAGE

echo "Deploying brooky-api to Cloud Run"
gcloud run deploy brooky-api \
    --image $DOCKER_IMAGE \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory $MEMORY

echo "API deployed successfully"
