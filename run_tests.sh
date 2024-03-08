#!/bin/bash

echo "Hello"
# Get container ID
container_id=$(docker ps -qf "name=chat-langchain-backend")

# Check if container ID is not empty
if [ -z "$container_id" ]; then
    echo "Error: Container not found."
    exit 1
fi

# Run command in the container
docker exec -ti "$container_id" /bin/bash

docker exec -t $(docker ps -qf "name=chat-langchain-backend") /bin/bash
