
#ifndef RASTERIZER_AUXILIARY_H
#define RASTERIZER_AUXILIARY_H
#pragma once

#include "config.hpp"

#include "gsm/gsm.cuh"

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

#define DEPTH_MIN 0.2f

__forceinline__ __device__ bool in_frustum(bool prefiltered, const gsm::vec3 &p_view) {

  if (p_view.z <= DEPTH_MIN) // || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y <
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

// =============================================================================

// determine whether p is on the left side(+), right side(-), or on the line(0) of the segment ab
__forceinline__ __device__ float _cross2D(gsm::vec2 a, gsm::vec2 b, gsm::vec2 p) {
  return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

__forceinline__ __device__ int _windingNumber(const gsm::vec2 *__restrict__ polygon, int n, gsm::vec2 p) {
  int winding = 0;

  for (int i = 0, j = n - 1; i < n; j = i++) {
    const gsm::vec2 &a = polygon[j];
    const gsm::vec2 &b = polygon[i];

    if (a.y <= p.y) {
      if (b.y > p.y && _cross2D(a, b, p) > 0)
        ++winding; // move upward through, on the left side
    } else {
      if (b.y <= p.y && _cross2D(a, b, p) < 0)
        --winding; // move downward through, on the right side
    }
  }
  return winding;
}

__forceinline__ __device__ bool pointInPolygon(const gsm::vec2 *__restrict__ polygon, int n, gsm::vec2 p) {
  return _windingNumber(polygon, n, p) != 0;
}

#endif // !RASTERIZER_AUXILIARY_H
