#!/usr/bin/env bash

set -eo pipefail
# cd "$( dirname "${BASH_SOURCE[0]}" )"

registry="nvcr.io"
repository="nvstaging/tao/tao_tf2_base_image"

tag="$USER-$(date +%Y%m%d%H%M)"
local_tag="$USER"

# Build parameters.
BUILD_DOCKER="0"
PUSH_DOCKER="0"
FORCE="0"


# Parse command line.
while [[ $# -gt 0 ]]
    do
    key="$1"

    case $key in
        -b|--build)
        BUILD_DOCKER="1"
        shift # past argument
        ;;
        -p|--push)
        PUSH_DOCKER="1"
        shift # past argument
        ;;
        -f|--force)
        FORCE=1
        shift
        ;;
        --default)
        BUILD_DOCKER="1"
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

    docker build --pull -f $NV_TAO_TF2_TOP/docker/Dockerfile -t $registry/$repository:$local_tag $NO_CACHE \
            --network=host $NV_TAO_TF2_TOP/. \
            --build-arg EFF_TOKEN_NAME="$EFF_TOKEN_NAME" \
            --build-arg EFF_TOKEN_PASSWORD="$EFF_TOKEN_PASSWORD"

    if [ $PUSH_DOCKER = "1" ]; then
        echo "Pusing docker ..."
        docker tag $registry/$repository:$local_tag $registry/$repository:$tag
        docker push $registry/$repository:$tag
        echo $registry/$repository:$tag
        digest=$(docker inspect --format='{{index .RepoDigests 0}}' $registry/$repository:$tag)
        echo -e "\033[1;33mUpdate the digest in the manifest.json file to:\033[0m"
        echo $digest
    else
        echo "Skip pushing docker ..."
    fi
else
    echo "Usage: ./build.sh [--build] [--push] [--force] [--default]"
fi
