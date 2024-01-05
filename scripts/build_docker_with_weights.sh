#!/bin/bash

# get version from version.py file
VERSION=$(python -c "import relik; print(relik.__version__)")
echo "Building version: $VERSION"

# if relik-model exists, delete it
REPO_MODEL_PATH="relik-model"
if [ -d "$REPO_MODEL_PATH" ]; then
    echo "Deleting $REPO_MODEL_PATH"
    rm -r "$REPO_MODEL_PATH"
fi

# create relik-model directory and copy model files
echo "Copying model files to $REPO_MODEL_PATH"
mkdir -p "$REPO_MODEL_PATH"
# copy model files
cp -r "$MODEL_PATH"/* "$REPO_MODEL_PATH"

docker build -f dockerfiles/ray/Dockerfile.cuda -t relik:$VERSION-bsc-cuda-ray .

# clean up
if [ -d "$REPO_MODEL_PATH" ]; then
    echo "Deleting $REPO_MODEL_PATH"
    rm -r "$REPO_MODEL_PATH"
fi
