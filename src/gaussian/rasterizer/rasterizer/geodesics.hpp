#ifndef RASTERIZER_GEODESICS_HPP
#define RASTERIZER_GEODESICS_HPP
#pragma once

#include "defines.hpp"

namespace GaussianImplicit {

float3 project(float3 x, float threshold, const float *pos_cuda, const float *scale_cuda,
               const float *rot_cuda, const float *opacity_cuda, const int *gridData, const int *gridOffsets,
               int gridRes, float3 gridMin, float cellSize);

float3 normal(float3 x, const float *pos_cuda, const float *scale_cuda, const float *rot_cuda,
              const float *opacity_cuda, const int *gridData, const int *gridOffsets, int gridRes,
              float3 gridMin, float cellSize);

} // namespace GaussianImplicit

namespace CudaRasterizer {

void makeGeodesicsMask(const float *depth_raw, const float *t_final,
                       // LogMap
                       const float *pts3d, int nPts, const float *uvs, const int *gridData,
                       const int *gridOffsets, int gridRes, float3 gridMin, float cellSize, float R,
                       const float *lastPoints, int nLastPoints,
                       // camera
                       const float *colmap_projviewmatrix, int width, int height,
                       const float *inverse_colmap_viewmatrix, float tan_fovx, float tan_fovy,
                       // texture
                       cudaTextureObject_t model_basecolor_map_cuda,
                       cudaTextureObject_t model_normal_map_cuda, cudaTextureObject_t model_height_map_cuda,
                       // output
                       PixelMask *mask);

} // namespace CudaRasterizer

#endif // !RASTERIZER_GEODESICS_HPP
