#!/bin/bash
set -e

# Pre-start
# checkmark font for fancy log
CHECK_MARK="\033[0;32m\xE2\x9C\x94\033[0m"
# usage text
USAGE="$(basename "$0") [-h --help] [-c --config] [-p --precision] [-d --device] [--retriever] [--retriever-device] 
[--retriever-precision] [--index-device] [--index-precision] [--reader] [--reader-device] [--reader-precision]

where:
    -h --help               Show this help text
    -c --config             Config name (from HuggingFace) or path
    -p --precision          Training precision, default '32'.
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
  *) set -- "$@" "$arg" ;;
  esac
done

# check for named params
#while [ $OPTIND -le "$#" ]; do
while getopts ":hc:p:d:q:w:e:r:t:y:a:b:f:" opt; do
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
  \?)
    echo "Invalid option -$OPTARG" >&2 && echo "$USAGE" && exit 0
    ;;
  esac
done

# FastAPI app location
if [ -z "$APP_MODULE" ]; then
  # echo "APP_MODULE not set, using default"
  export APP_MODULE=relik.inference.serve.backend.ray:server
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

# Start Ray Serve with the app
exec serve run "$APP_MODULE" --host 0.0.0.0 --port 8000
# micromamba run -n base serve run "$APP_MODULE" --host 0.0.0.0 --port 8000
