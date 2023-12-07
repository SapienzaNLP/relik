#!/bin/bash

# setup conda
CONDA_BASE=$(conda info --base)
# check if conda is installed
if [ -z "$CONDA_BASE" ]; then
  echo "Conda is not installed. Please install conda first."
  exit 1
fi
source "$CONDA_BASE"/etc/profile.d/conda.sh

# create conda env
read -rp "Enter environment name or prefix: " ENV_NAME
read -rp "Enter python version (default 3.10): " PYTHON_VERSION
if [ -z "$PYTHON_VERSION" ]; then
  PYTHON_VERSION="3.10"
fi

# check if ENV_NAME is a full path
if [[ "$ENV_NAME" == /* ]]; then
  CONDA_NEW_ARG="--prefix"
else
  CONDA_NEW_ARG="--name"
fi

conda create -y "$CONDA_NEW_ARG" "$ENV_NAME" python="$PYTHON_VERSION"
conda activate "$ENV_NAME"

# replace placeholder env with $ENV_NAME in scripts/train.sh
# NEW_CONDA_LINE="source \$CONDA_BASE/bin/activate $ENV_NAME"
# sed -i.bak -e "s,.*bin/activate.*,$NEW_CONDA_LINE,g" scripts/train.sh

# install torch
read -rp "Enter cuda version (e.g. '11.8', default no cuda support): " CUDA_VERSION
read -rp "Enter PyTorch version (e.g. '2.1', default latest): " PYTORCH_VERSION
if [ -n "$PYTORCH_VERSION" ]; then
  PYTORCH_VERSION="=$PYTORCH_VERSION"
fi
if [ -z "$CUDA_VERSION" ]; then
  conda install -y pytorch"$PYTORCH_VERSION" cpuonly -c pytorch
else
  conda install -y pytorch"$PYTORCH_VERSION" pytorch-cuda="$CUDA_VERSION" -c pytorch -c nvidia
fi

# install python requirements
pip install -e .[all]
