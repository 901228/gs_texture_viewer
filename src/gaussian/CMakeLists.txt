cmake_minimum_required(VERSION 3.16)

project(GAUSSIAN_LIB LANGUAGES CXX CUDA)

# CXX Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA Standard
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# CUDA Architectures
# RTX 3060/3080: Ampere (sm_86)
# RTX 5090: Blackwell (sm_100)
# including downward compatible architectures to support more GPUs
set(CMAKE_CUDA_ARCHITECTURES 86 89 90 100)

# Show CUDA information
message(STATUS "CUDA Version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# create CUDA static library
add_library(GAUSSIAN_LIB STATIC
  rasterizer/forward.hpp
  rasterizer/forward.cu
  rasterizer/auxiliary.h
  rasterizer/rasterizer.cu
  rasterizer/rasterizer.hpp

  rasterizer/texture_rasterizer.cu
  rasterizer/texture_rasterizer.hpp

  rasterizer/vector/matrix.hpp
  rasterizer/vector/vector.hpp

  renderer/ellipsoid_renderer.cpp
  renderer/point_renderer.cpp
  renderer/gaussian_renderer.cpp

  model/gs_model.cpp
  model/texture_gs_model.cpp

  view/gs_view.cpp
  view/texture_gs_view.cpp
)

# CUDA Separable Compilation
set_target_properties(GAUSSIAN_LIB PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

# link libraries
target_link_libraries(GAUSSIAN_LIB PRIVATE glad imgui OpenMeshCore OpenMeshTools magic_enum::magic_enum)

# link CUDA runtime
target_link_libraries(GAUSSIAN_LIB PUBLIC CUDA::cudart)
target_link_directories(GAUSSIAN_LIB PUBLIC ${CUDAToolkit_LIBRARY_DIR})

# include directories
target_include_directories(GAUSSIAN_LIB PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
