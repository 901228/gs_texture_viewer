@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "BUILD_DIR=%SCRIPT_DIR%..\build\"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

if "%~1"=="" (
    call :print_usage
    exit /b 2
)

if /i not "%~1"=="Debug" if /i not "%~1"=="Release" (
    call :print_usage
    exit /b 2
)

set "variant=%~1"

pushd "%BUILD_DIR%" >nul 2>&1

cmake .. -B . -DCMAKE_BUILD_TYPE=%variant% -G Ninja

popd >nul 2>&1

exit /b 0

:print_usage
echo Usage:
echo     configure [Debug ^| Release]
exit /b 0
