#ifndef GSM_VECTOR_OPERATOR_CUH
#define GSM_VECTOR_OPERATOR_CUH
#pragma once

#include <cuda_runtime.h>

#include "base.cuh"
#include "ivector.cuh"
#include "vector.cuh"

namespace gsm {

// Component-wise max/min
__host__ __device__ __forceinline__ vec2 max(const vec2 &a, const vec2 &b) {
  return {max(a.x, b.x), max(a.y, b.y)};
}
__host__ __device__ __forceinline__ vec2 max(const vec2 &a, float val) {
  return {max(a.x, val), max(a.y, val)};
}
__host__ __device__ __forceinline__ vec3 max(const vec3 &a, const vec3 &b) {
  return {max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)};
}
__host__ __device__ __forceinline__ vec3 max(const vec3 &a, float val) {
  return {max(a.x, val), max(a.y, val), max(a.z, val)};
}
__host__ __device__ __forceinline__ vec4 max(const vec4 &a, const vec4 &b) {
  return {max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w)};
}
__host__ __device__ __forceinline__ vec4 max(const vec4 &a, float val) {
  return {max(a.x, val), max(a.y, val), max(a.z, val), max(a.w, val)};
}

__host__ __device__ __forceinline__ vec2 min(const vec2 &a, const vec2 &b) {
  return {min(a.x, b.x), min(a.y, b.y)};
}
__host__ __device__ __forceinline__ vec2 min(const vec2 &a, float val) {
  return {min(a.x, val), min(a.y, val)};
}
__host__ __device__ __forceinline__ vec3 min(const vec3 &a, const vec3 &b) {
  return {min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)};
}
__host__ __device__ __forceinline__ vec3 min(const vec3 &a, float val) {
  return {min(a.x, val), min(a.y, val), min(a.z, val)};
}
__host__ __device__ __forceinline__ vec4 min(const vec4 &a, const vec4 &b) {
  return {min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w)};
}
__host__ __device__ __forceinline__ vec4 min(const vec4 &a, float val) {
  return {min(a.x, val), min(a.y, val), min(a.z, val), min(a.w, val)};
}

} // namespace gsm

namespace gsm {

__host__ __device__ __forceinline__ float length(const vec2 &v) { return v.length(); }
__host__ __device__ __forceinline__ float length(const vec3 &v) { return v.length(); }
__host__ __device__ __forceinline__ float length(const vec4 &v) { return v.length(); }

__host__ __device__ __forceinline__ float lengthSquared(const vec2 &v) { return v.lengthSquared(); }
__host__ __device__ __forceinline__ float lengthSquared(const vec3 &v) { return v.lengthSquared(); }
__host__ __device__ __forceinline__ float lengthSquared(const vec4 &v) { return v.lengthSquared(); }

__host__ __device__ __forceinline__ vec3 normalize(const vec3 &a) { return a.normalized(); }
__host__ __device__ __forceinline__ vec2 normalize(const vec2 &a) { return a.normalized(); }
__host__ __device__ __forceinline__ vec4 normalize(const vec4 &a) { return a.normalized(); }

__host__ __device__ __forceinline__ float dot(const vec2 &a, const vec2 &b) { return a.x * b.x + a.y * b.y; }
__host__ __device__ __forceinline__ float dot(const vec3 &a, const vec3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ __forceinline__ float dot(const vec4 &a, const vec4 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__host__ __device__ __forceinline__ const vec3 cross(const vec3 &a, const vec3 &b) { return a.cross(b); }

} // namespace gsm

namespace gsm {

__host__ __device__ __forceinline__ ivec2 floor(const vec2 &a) { return {floor(a.x), floor(a.y)}; }
__host__ __device__ __forceinline__ ivec3 floor(const vec3 &a) {
  return {floor(a.x), floor(a.y), floor(a.z)};
}
__host__ __device__ __forceinline__ ivec4 floor(const vec4 &a) {
  return {floor(a.x), floor(a.y), floor(a.z), floor(a.w)};
}

__host__ __device__ __forceinline__ ivec2 ceil(const vec2 &a) { return {ceil(a.x), ceil(a.y)}; }
__host__ __device__ __forceinline__ ivec3 ceil(const vec3 &a) { return {ceil(a.x), ceil(a.y), ceil(a.z)}; }
__host__ __device__ __forceinline__ ivec4 ceil(const vec4 &a) {
  return {ceil(a.x), ceil(a.y), ceil(a.z), ceil(a.w)};
}

__host__ __device__ __forceinline__ vec2 clamp(const vec2 &a, float min, float max) {
  return {clamp(a.x, min, max), clamp(a.y, min, max)};
}
__host__ __device__ __forceinline__ vec3 clamp(const vec3 &a, float min, float max) {
  return {clamp(a.x, min, max), clamp(a.y, min, max), clamp(a.z, min, max)};
}
__host__ __device__ __forceinline__ vec4 clamp(const vec4 &a, float min, float max) {
  return {clamp(a.x, min, max), clamp(a.y, min, max), clamp(a.z, min, max), clamp(a.w, min, max)};
}

__host__ __device__ __forceinline__ ivec2 clamp(const ivec2 &a, int min, int max) {
  return {clamp(a.x, min, max), clamp(a.y, min, max)};
}
__host__ __device__ __forceinline__ ivec3 clamp(const ivec3 &a, int min, int max) {
  return {clamp(a.x, min, max), clamp(a.y, min, max), clamp(a.z, min, max)};
}
__host__ __device__ __forceinline__ ivec4 clamp(const ivec4 &a, int min, int max) {
  return {clamp(a.x, min, max), clamp(a.y, min, max), clamp(a.z, min, max), clamp(a.w, min, max)};
}

} // namespace gsm

namespace gsm {

__host__ __device__ __forceinline__ float *value_ptr(vec2 &v) { return &v.x; }
__host__ __device__ __forceinline__ const float *value_ptr(const vec2 &v) { return &v.x; }
__host__ __device__ __forceinline__ float *value_ptr(vec3 &v) { return &v.x; }
__host__ __device__ __forceinline__ const float *value_ptr(const vec3 &v) { return &v.x; }
__host__ __device__ __forceinline__ float *value_ptr(vec4 &v) { return &v.x; }
__host__ __device__ __forceinline__ const float *value_ptr(const vec4 &v) { return &v.x; }

} // namespace gsm

#endif // !GSM_VECTOR_OPERATOR_CUH
