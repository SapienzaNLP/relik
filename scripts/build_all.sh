#!/bin/bash


# get version from version.py file
VERSION=$(python -c "import relik; print(relik.__version__)")
echo "Building version: $VERSION"

# echo "==== Building CPU images ===="
docker build -f dockerfiles/ray/Dockerfile.cpu -t relik:$VERSION-cpu-ray .
# docker build -f dockerfiles/fastapi/Dockerfile.cpu -t relik:$VERSION-cpu-fastapi .

# echo "==== Building GPU images ===="
# docker build -f dockerfiles/ray/Dockerfile.cuda -t relik:$VERSION-cuda-ray .
# docker build -f dockerfiles/fastapi/Dockerfile.cuda -t relik:$VERSION-cuda-fastapi .
