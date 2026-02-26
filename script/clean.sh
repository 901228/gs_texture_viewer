#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
BUILD_DIR="$SCRIPT_DIR/../build/"

pushd $BUILD_DIR > /dev/null

cmake --build . --target clean

popd > /dev/null
