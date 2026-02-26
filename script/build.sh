#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
BUILD_DIR="$SCRIPT_DIR/../build/"
mkdir -p $BUILD_DIR

print_usage() {

    echo "Usage:"
    echo "    build [Debug | Release]"
}

if [ -z "$1" ]
then
    print_usage
    exit 2
elif [ "$1" != "Debug" ] && [ "$1" != "Release" ]
then
    print_usage
    exit 2
fi

variant="$1"

# run configuration first
bash "${SCRIPT_DIR}/configure.sh" ${variant}

pushd $BUILD_DIR > /dev/null

cmake --build .

popd > /dev/null
