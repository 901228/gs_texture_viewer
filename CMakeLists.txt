cmake_minimum_required(VERSION 3.16)

# project settings
project(texture_viewer LANGUAGES C CXX)

# CXX Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Show Compiler Information
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# Options
add_definitions("-D_XKEYCHECK_H")
add_definitions(-DPROJECT_DIR="${PROJECT_SOURCE_DIR}")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_definitions(-w)

if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
  # Ensure C++ exception handling is enabled (required by toml++ / try-catch usage).
  # CMAKE_CXX_FLAGS does not always include /EHsc under the Ninja generator.
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
  # WIN32/_WINDOWS are normally part of CMake's default MSVC flags, but are missing
  # here under Ninja; some third-party code (e.g. OpenMesh) relies on WIN32 for
  # platform detection.
  add_compile_definitions(WIN32 _WINDOWS)
endif()

# external libraries
add_subdirectory(extern)

# main source code
add_subdirectory(src)

set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "${PROJECT_NAME}")
