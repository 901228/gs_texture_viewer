#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
BUILD_DIR="$SCRIPT_DIR/../build/"
mkdir -p $BUILD_DIR

print_usage() {

    echo "Usage:"
    echo "    configure [Debug | Release]"
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

pushd $BUILD_DIR > /dev/null

cmake .. -B . -DCMAKE_BUILD_TYPE=$variant -G Ninja

popd > /dev/null
