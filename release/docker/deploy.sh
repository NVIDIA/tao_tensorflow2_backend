#!/usr/bin/env bash

set -eo pipefail
# cd "$( dirname "${BASH_SOURCE[0]}" )"

registry="nvcr.io"
tensorflow_version="2.11.0"
tao_version="5.0.0"
repository="nvidia/tao/tao-toolkit"
build_id=${USER}
tag="${tao_version}-tf${tensorflow_version}-py3-${build_id}"

# Build parameters.
BUILD_DOCKER="0"
BUILD_WHEEL="0"
PUSH_DOCKER="0"
FORCE="0"


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
        -w|--wheel)
        BUILD_WHEEL="1"
        RUN_DOCKER="0"
        shift # past argument
        ;;
        -f|--force)
        FORCE=1
        shift
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
        PUSH_DOCKER="0"
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
    if [ $BUILD_WHEEL = "1" ]; then
        echo "Building source code wheel ..."
       tao_tf2 -- make build
    else
        echo "Skipping wheel builds ..."
    fi
    
    docker build -f $NV_TAO_TF2_TOP/release/docker/Dockerfile.release -t $registry/$repository:$tag $NO_CACHE --network=host $NV_TAO_TF2_TOP/.

    if [ $BUILD_WHEEL = "1" ]; then
        echo "Cleaning wheels ..."
        tao_tf2 -- make clean
    else
        echo "Skipping wheel cleaning ..."
    fi
    echo "The final release container was build at ${registry}/${repository}:${tag}"

elif [ $RUN_DOCKER = "1" ]; then
    echo "Running docker interactively..."
    docker run --gpus all -v /home/$USER/tlt-experiments:/workspace/tlt-experiments \
                          --net=host \
                          --shm-size=30g \
                          --ulimit memlock=-1 \
                          --ulimit stack=67108864 \
                          --rm -it $registry/$repository:$tag /bin/bash
else
    echo "Usage: ./deploy.sh [--build] [--wheel] [--run] [--default]"
fi
