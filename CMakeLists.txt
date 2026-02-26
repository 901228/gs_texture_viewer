cmake_minimum_required(VERSION 3.16)

# project settings
project(gs_texture_viewer LANGUAGES C CXX CUDA)

# CXX Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA Standard
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA Architectures
# RTX 3060/3080: Ampere (sm_86)
# RTX 5090: Blackwell (sm_100)
# including downward compatible architectures to support more GPUs
set(CMAKE_CUDA_ARCHITECTURES 86 89 90 100)

# Show Compiler Information
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# Options
add_definitions("-D_XKEYCHECK_H")
add_definitions(-DPROJECT_DIR="${PROJECT_SOURCE_DIR}")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_definitions(-w)

if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()

# include
include_directories(extern/OpenMesh/src/)

add_subdirectory(extern)
add_subdirectory(src)

set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "${PROJECT_NAME}")

# install paths
# install(TARGETS ${PROJECT_NAME}
# CONFIGURATIONS Debug
# RUNTIME DESTINATION ${CMAKE_BINARY_DIR}/Debug)
# install(TARGETS ${PROJECT_NAME}
# CONFIGURATIONS Release MinSizeRel RelWithDebInfo
# RUNTIME DESTINATION ${CMAKE_BINARY_DIR}/Release)
