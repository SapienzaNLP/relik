#!/bin/bash
set -e

# Pre-start
# checkmark font for fancy log
CHECK_MARK="\033[0;32m\xE2\x9C\x94\033[0m"
# usage text
USAGE="$(basename "$0") [-h --help] [-c --config] [-p --precision] [-d --device] [--retriever] [--retriever-device] 
[--retriever-precision] [--index-device] [--index-precision] [--reader] [--reader-device] [--reader-precision] 
[--annotation-type] [--frontend] [--workers] -- start the FastAPI server for the RElik model

where:
    -h --help               Show this help text
    -c --config             Pretrained ReLiK config name (from HuggingFace) or path
    -p --precision          Precision, default '32'.
    -d --device             Device to use, default 'cpu'.
    --retriever             Override retriever model name.
    --retriever-device      Override retriever device.
    --retriever-precision   Override retriever precision.
    --index-device          Override index device.
    --index-precision       Override index precision.
    --reader                Override reader model name.
    --reader-device         Override reader device.
    --reader-precision      Override reader precision.
    --annotation-type       Annotation type ('char', 'word'), default 'char'.
    --frontend              Whether to start the frontend server.
    --workers               Number of workers to use.
"

# Transform long options to short ones
for arg in "$@"; do
  shift
  case "$arg" in
  '--help') set -- "$@" '-h' ;;
  '--config') set -- "$@" '-c' ;;
  '--precision') set -- "$@" '-p' ;;
  '--device') set -- "$@" '-d' ;;
  '--retriever') set -- "$@" '-q' ;;
  '--retriever-device') set -- "$@" '-w' ;;
  '--retriever-precision') set -- "$@" '-e' ;;
  '--index-device') set -- "$@" '-r' ;;
  '--index-precision') set -- "$@" '-t' ;;
  '--reader') set -- "$@" '-y' ;;
  '--reader-device') set -- "$@" '-a' ;;
  '--reader-precision') set -- "$@" '-b' ;;
  '--annotation-type') set -- "$@" '-f' ;;
  '--frontend') set -- "$@" '-v' ;;
  '--workers') set -- "$@" '-z' ;;
  *) set -- "$@" "$arg" ;;
  esac
done

# check for named params
#while [ $OPTIND -le "$#" ]; do
while getopts ":hc:p:d:q:w:e:r:t:y:a:b:f:vz:" opt; do
  case $opt in
  h)
    printf "%s$USAGE" && exit 0
    ;;
  c)
    export RELIK_PRETRAINED=$OPTARG
    ;;
  p)
    export PRECISION=$OPTARG
    ;;
  d)
    export DEVICE=$OPTARG
    ;;
  q)
    export RETRIEVER_MODEL_NAME=$OPTARG
    ;;
  w)
    export RETRIEVER_DEVICE=$OPTARG
    ;;
  e)
    export RETRIEVER_PRECISION=$OPTARG
    ;;
  r)
    export INDEX_DEVICE=$OPTARG
    ;;
  t)
    export INDEX_PRECISION=$OPTARG
    ;;
  y)
    export READER_MODEL_NAME=$OPTARG
    ;;
  a)
    export READER_DEVICE=$OPTARG
    ;;
  b)
    export READER_PRECISION=$OPTARG
    ;;
  f)
    export ANNOTATION_TYPE=$OPTARG
    ;;
  v)
    export FRONTEND=true
    ;;
  z)
    export WORKERS=$OPTARG
    ;;
  \?)
    echo "Invalid option -$OPTARG" >&2 && echo "$USAGE" && exit 0
    ;;
  esac
done

if [ -z "$RELIK_PRETRAINED" ]; then
  echo "No ReLiK model specified. Please provide a model name or path. Exiting." &&
  exit 1
fi

# Device
if [ -z "$DEVICE" ]; then
  # echo "DEVICE not set, using default"
  export DEVICE=cpu
fi

# Retriever device
if [ -z "$RETRIEVER_DEVICE" ]; then
  # echo "RETRIEVER_DEVICE not set, using default"
  export RETRIEVER_DEVICE=$DEVICE
fi

# Index device
if [ -z "$INDEX_DEVICE" ]; then
  # echo "INDEX_DEVICE not set, using default"
  export INDEX_DEVICE=$DEVICE
fi

# Reader device
if [ -z "$READER_DEVICE" ]; then
  # echo "READER_DEVICE not set, using default"
  export READER_DEVICE=$DEVICE
fi

if [ -z "$PRECISION" ]; then
  # echo "PRECISION not set, using default"
  export PRECISION=32
fi

if [ -z "$RETRIEVER_PRECISION" ]; then
  # echo "RETRIEVER_PRECISION not set, using default"
  export RETRIEVER_PRECISION=$PRECISION
fi

if [ -z "$INDEX_PRECISION" ]; then
  # echo "INDEX_PRECISION not set, using default"
  export INDEX_PRECISION=$PRECISION
fi

if [ -z "$READER_PRECISION" ]; then
  # echo "READER_PRECISION not set, using default"
  export READER_PRECISION=$PRECISION
fi

if [ -z "$ANNOTATION_TYPE" ]; then
  # echo "ANNOTATION_TYPE not set, using default"
  export ANNOTATION_TYPE=char
fi

if [ -z "$FRONTEND" ]; then
  # echo "FRONTEND not set, using default"
  export FRONTEND=false
fi

if [ -z "$WORKERS" ]; then
  # echo "WORKERS not set, using default"
  WORKERS=1
fi


# FastAPI app location
if [ -z "$APP_MODULE" ]; then
  # echo "APP_MODULE not set, using default"
  export APP_MODULE=relik.inference.serve.backend.fastapi:app
fi
# echo "APP_MODULE set to $APP_MODULE"

# If there's a prestart.sh script in the /app directory, run it before starting
if [ -z "$PRE_START_PATH" ]; then
  # echo "PRE_START_PATH not set, using default"
  PRE_START_PATH=scripts/docker/pre-start.sh
fi
# echo "PRE_START_PATH set to $PRE_START_PATH"

if [ -f $PRE_START_PATH ]; then
  . "$PRE_START_PATH" $RELIK_PRETRAINED
else
  echo "There is no script $PRE_START_PATH"
fi

if [ -z "$GUNICORN_CONF" ]; then
  # echo "PRE_START_PATH not set, using default"
  GUNICORN_CONF=scripts/docker/gunicorn_conf.py
fi

# Start Ray Serve with the app
# exec gunicorn -k uvicorn.workers.UvicornWorker -c "$GUNICORN_CONF" "$APP_MODULE" -b 0.0.0.0:8000
exec relik serve $RELIK_PRETRAINED \
  --device $DEVICE \
  --retriever-device $RETRIEVER_DEVICE \
  --index-device $INDEX_DEVICE \
  --reader-device $READER_DEVICE \
  --precision $PRECISION \
  --retriever-precision $RETRIEVER_PRECISION \
  --index-precision $INDEX_PRECISION \
  --reader-precision $READER_PRECISION \
  --annotation-type $ANNOTATION_TYPE \
  --frontend $FRONTEND \
  --workers $WORKERS \
  --host 0.0.0.0 \
  --port 8000
