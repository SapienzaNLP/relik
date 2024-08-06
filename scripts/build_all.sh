#!/bin/bash


# get version from version.py file
VERSION=$(python -c "import relik; print(relik.__version__)")
LATEST_VERSION=$(echo "$VERSION" | tail -n 1)
echo "Building version: $LATEST_VERSION"

echo "==== Building CPU images ===="
# docker build -f dockerfiles/ray/Dockerfile.cpu -t relik:$VERSION-cpu-ray .
docker build -f dockerfiles/fastapi/Dockerfile.cpu -t relik:$LATEST_VERSION-cpu-fastapi .

echo "==== Building GPU images ===="
# docker build -f dockerfiles/ray/Dockerfile.cuda -t relik:$VERSION-cuda-ray .
docker build -f dockerfiles/fastapi/Dockerfile.cuda -t relik:$LATEST_VERSION-cuda-fastapi .
