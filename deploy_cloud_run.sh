#!/bin/bash

IMAGE_NAME="eu.gcr.io/blank-gpt/blank-gpt:1.1.1"
REGION="europe-west1"
PORT=9000
MEMORY_LIMIT=4Gi
CPU=1
MAX_INSTANCES=5


source .env

gcloud builds submit \
-t $IMAGE_NAME \
. 

gcloud run deploy blank-gpt --image=$IMAGE_NAME \
--platform=managed --allow-unauthenticated --region=$REGION  --port=$PORT \
--memory=$MEMORY_LIMIT --cpu=$CPU --timeout=300 --max-instances $MAX_INSTANCES --min-instances 0
