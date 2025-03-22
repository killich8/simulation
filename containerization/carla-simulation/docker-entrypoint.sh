#!/bin/bash
# Docker entrypoint script for CARLA simulation service

set -e

# Check if CARLA server is already running
if pgrep -x "CarlaUE4-Linux-" > /dev/null
then
    echo "CARLA server is already running"
else
    echo "Starting CARLA server..."
    # Start CARLA server in the background
    /opt/carla/CarlaUE4.sh -opengl -quality-level=Low -carla-rpc-port=2000 -carla-streaming-port=2001 &
    
    # Wait for CARLA server to start
    echo "Waiting for CARLA server to start..."
    sleep 10
fi

# Execute the command passed to the container
echo "Executing: $@"
exec "$@"
