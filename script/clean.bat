@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "BUILD_DIR=%SCRIPT_DIR%..\build\"

pushd "%BUILD_DIR%" >nul 2>&1

cmake --build . --target clean

popd >nul 2>&1
