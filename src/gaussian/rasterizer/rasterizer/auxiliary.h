
#ifndef RASTERIZER_AUXILIARY_HPP
#define RASTERIZER_AUXILIARY_HPP
#pragma once

#include "config.hpp"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
                                  -1.0925484305920792f, 0.5462742152960396f};
__device__ const float SH_C3[] = {-0.5900435899266435f, 2.890611442640554f,   -0.4570457994644658f,
                                  0.3731763325901154f,  -0.4570457994644658f, 1.445305721320277f,
                                  -0.5900435899266435f};

__forceinline__ __device__ float ndc2Pix(float v, int S) {
  return ((v + 1.0f) * static_cast<float>(S) - 1.0f) * 0.5f;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2 &rect_min, uint2 &rect_max,
                                        dim3 grid) {
  rect_min = {min(grid.x, max(0, static_cast<int>((p.x - static_cast<float>(max_radius)) / BLOCK_X))),
              min(grid.y, max(0, static_cast<int>((p.y - static_cast<float>(max_radius)) / BLOCK_Y)))};
  rect_max = {
      min(grid.x, max(0, static_cast<int>((p.x + static_cast<float>(max_radius) + BLOCK_X - 1) / BLOCK_X))),
      min(grid.y, max(0, static_cast<int>((p.y + static_cast<float>(max_radius) + BLOCK_Y - 1) / BLOCK_Y)))};
}

__forceinline__ __device__ void getRect(const float2 p, int2 ext_rect, uint2 &rect_min, uint2 &rect_max,
                                        dim3 grid) {
  rect_min = {min(grid.x, max(0, static_cast<int>((p.x - static_cast<float>(ext_rect.x)) / BLOCK_X))),
              min(grid.y, max(0, static_cast<int>((p.y - static_cast<float>(ext_rect.y)) / BLOCK_Y)))};
  rect_max = {
      min(grid.x, max(0, static_cast<int>((p.x + static_cast<float>(ext_rect.x) + BLOCK_X - 1) / BLOCK_X))),
      min(grid.y, max(0, static_cast<int>((p.y + static_cast<float>(ext_rect.y) + BLOCK_Y - 1) / BLOCK_Y)))};
}

__forceinline__ __device__ float3 transformPoint4x3(const float3 &p, const float *matrix) {
  float3 transformed = {
      matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
      matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
      matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
  };
  return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3 &p, const float *matrix) {
  float4 transformed = {matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
                        matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
                        matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
                        matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]};
  return transformed;
}

__forceinline__ __device__ bool in_frustum(int idx, const float *orig_points, const float *viewmatrix,
                                           const float *projmatrix, bool prefiltered, float3 &p_view) {
  float3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]};

  // Bring points to screen space
  float4 p_hom = transformPoint4x4(p_orig, projmatrix);
  float p_w = 1.0f / (p_hom.w + 0.0000001f);
  float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};
  p_view = transformPoint4x3(p_orig, viewmatrix);

  if (p_view.z <= 0.2f) // || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y <
                        // -1.3 || p_proj.y > 1.3)))
  {
    if (prefiltered) {
      printf("Point is filtered although prefiltered is set. This shouldn't "
             "happen!");
      __trap();
    }
    return false;
  }
  return true;
}

#endif // !RASTERIZER_AUXILIARY_HPP
