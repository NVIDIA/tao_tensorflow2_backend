#!/usr/bin/env bash

set -eo pipefail

# Parse command line.
while [[ $# -gt 0 ]]
    do
    key="$1"

    case $key in
        -b|--build)
        BUILD_DOCKER="1"
        RUN_DOCKER="0"
        shift # past argument
        ;;
        -r|--run)
        RUN_DOCKER="1"
        BUILD_DOCKER="0"
        shift # past argument
        ;;
        --default)
        BUILD_DOCKER="0"
        RUN_DOCKER="1"
        FORCE="0"
        shift # past argument
        ;;
        *)    # unknown option
        POSITIONAL+=("$1") # save it in an array for later
        shift # past argument
        ;;
    esac
done


if [ $BUILD_DOCKER = "1" ]; then
    echo "Building base docker ..."
    if [ $FORCE = "1" ]; then
        echo "Forcing docker build without cache ..."
        NO_CACHE="--no-cache"
    else
        NO_CACHE=""
    fi
    
    docker build --rm -t tao-tf2:v0.1 . -f Dockerfile

elif [ $RUN_DOCKER = "1" ]; then
    echo "Running docker interatively..."
    docker run --gpus all -v /home/obaba/workspace/tao-tf2:/workspace \
                          -v /media/scratch.metropolis3:/home/scratch.p3 \
                          -v /media/projects.metropolis2:/home/projects2_metropolis \
                          --net=host --shm-size=30g --ulimit memlock=-1 --ulimit stack=67108864 \
                          --rm -it tao-tf2:v0.1 /bin/bash
else
    echo "Usage: ./build_and_run.sh [--build] [--run] [--default]"
fi
