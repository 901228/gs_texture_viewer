#ifndef GSM_COMMON_CUH
#define GSM_COMMON_CUH
#pragma once

#include <cmath>
#include <cuda_runtime.h>

namespace gsm {

__host__ __device__ __forceinline__ float max(float a, float b) { return fmaxf(a, b); }
__host__ __device__ __forceinline__ int max(int a, int b) { return a > b ? a : b; }

__host__ __device__ __forceinline__ float min(float a, float b) { return fminf(a, b); }
__host__ __device__ __forceinline__ int min(int a, int b) { return a < b ? a : b; }

__host__ __device__ __forceinline__ float clamp(float a, float min, float max) {
  return gsm::max(min, gsm::min(a, max));
}
__host__ __device__ __forceinline__ int clamp(int a, int min, int max) {
  return gsm::max(min, gsm::min(a, max));
}

__host__ __device__ __forceinline__ int floor(float b) { return static_cast<int>(floorf(b)); }
__host__ __device__ __forceinline__ int ceil(float b) { return static_cast<int>(ceilf(b)); }

__host__ __device__ __forceinline__ float sqrt(float b) { return sqrtf(b); }

} // namespace gsm

#endif // !GSM_COMMON_CUH
