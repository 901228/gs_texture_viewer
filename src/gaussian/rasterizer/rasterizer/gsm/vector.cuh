#ifndef GSM_VECTOR_CUH
#define GSM_VECTOR_CUH
#pragma once

#include <cassert>

#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include "base.cuh"

namespace gsm {
struct vec2;
struct vec3;
struct vec4;

struct ivec2;
struct ivec3;
struct ivec4;
} // namespace gsm

namespace gsm {

struct vec2 {
  float x, y;

  // Constructors
  __host__ __device__ __forceinline__ vec2() : x(0), y(0) {}
  __host__ __device__ __forceinline__ explicit vec2(const float &val) : x(val), y(val) {}
  __host__ __device__ __forceinline__ vec2(const float &x, const float &y) : x(x), y(y) {}
  __host__ __device__ __forceinline__ explicit vec2(const float *val) : x(val[0]), y(val[1]) {}

  __host__ __device__ __forceinline__ explicit vec2(const float2 &val) : x(val.x), y(val.y) {}
  __host__ __device__ __forceinline__ explicit vec2(const glm::vec2 &val) : x(val.x), y(val.y) {}

  __host__ __device__ __forceinline__ explicit vec2(const float3 &val) : x(val.x), y(val.y) {}
  __host__ __device__ __forceinline__ explicit vec2(const glm::vec3 &val) : x(val.x), y(val.y) {}
  __host__ __device__ explicit vec2(const vec3 &val);

  __host__ __device__ __forceinline__ explicit vec2(const float4 &val) : x(val.x), y(val.y) {}
  __host__ __device__ __forceinline__ explicit vec2(const glm::vec4 &val) : x(val.x), y(val.y) {}
  __host__ __device__ explicit vec2(const vec4 &val);

  __host__ __device__ explicit vec2(const ivec2 &val);

  __host__ __device__ __forceinline__ explicit operator float2() const { return {x, y}; }
  __host__ __device__ explicit operator ivec2() const;

  // Addition
  __host__ __device__ __forceinline__ vec2 operator+(const vec2 &other) const {
    return {x + other.x, y + other.y};
  }
  __host__ __device__ __forceinline__ vec2 &operator+=(const vec2 &other) {
    x += other.x;
    y += other.y;
    return *this;
  }
  __host__ __device__ __forceinline__ vec2 &operator+=(const float &val) {
    x += val;
    y += val;
    return *this;
  }
  __host__ __device__ __forceinline__ vec2 operator+(float val) const { return {x + val, y + val}; }

  // Subtraction
  __host__ __device__ __forceinline__ vec2 operator-(const vec2 &other) const {
    return {x - other.x, y - other.y};
  }
  __host__ __device__ __forceinline__ vec2 &operator-=(const vec2 &other) {
    x -= other.x;
    y -= other.y;
    return *this;
  }
  __host__ __device__ __forceinline__ vec2 &operator-=(const float &val) {
    x -= val;
    y -= val;
    return *this;
  }
  __host__ __device__ __forceinline__ vec2 operator-(float val) const { return {x - val, y - val}; }
  __host__ __device__ __forceinline__ vec2 operator-() const { return {-x, -y}; }

  // Multiplication
  __host__ __device__ __forceinline__ vec2 operator*(const vec2 &other) const {
    return {x * other.x, y * other.y};
  }
  __host__ __device__ __forceinline__ vec2 operator*(float val) const { return {x * val, y * val}; }
  __host__ __device__ __forceinline__ vec2 &operator*=(float val) {
    x *= val;
    y *= val;
    return *this;
  }

  // Division
  __host__ __device__ __forceinline__ vec2 operator/(const vec2 &other) const {
    return {x / other.x, y / other.y};
  }
  __host__ __device__ __forceinline__ vec2 operator/(float val) const { return {x / val, y / val}; }
  __host__ __device__ __forceinline__ vec2 &operator/=(float val) {
    x /= val;
    y /= val;
    return *this;
  }

  // Utility functions
  __host__ __device__ [[nodiscard]] __forceinline__ float length() const { return gsm::sqrt(x * x + y * y); }

  __host__ __device__ [[nodiscard]] __forceinline__ float lengthSquared() const { return x * x + y * y; }

  __host__ __device__ [[nodiscard]] __forceinline__ vec2 normalized() const {
    float len = length();
    return {x / len, y / len};
  }

  __host__ __device__ __forceinline__ void normalize() {
    float len = length();
    x /= len;
    y /= len;
  }

  __host__ __device__ [[nodiscard]] __forceinline__ float dot(const vec2 &other) const {
    return x * other.x + y * other.y;
  }

  __host__ __device__ __forceinline__ float &operator[](int i) {
    assert(i >= 0 && i < 2);
    if (i == 0)
      return x;
    else // i == 1
      return y;
  }
  __host__ __device__ __forceinline__ const float &operator[](int i) const {
    assert(i >= 0 && i < 3);
    if (i == 0)
      return x;
    else // i == 1
      return y;
  }
};

// Scalar * vec2
__host__ __device__ __forceinline__ vec2 operator*(float val, const vec2 &v) {
  return {v.x * val, v.y * val};
}

} // namespace gsm

namespace gsm {

struct vec3 {
  float x, y, z;

  // Constructors
  __host__ __device__ __forceinline__ vec3() : x(0), y(0), z(0) {}
  __host__ __device__ __forceinline__ explicit vec3(const float &val) : x(val), y(val), z(val) {}
  __host__ __device__ __forceinline__ vec3(const float &x, const float &y, const float &z)
      : x(x), y(y), z(z) {}
  __host__ __device__ __forceinline__ explicit vec3(const float *val) : x(val[0]), y(val[1]), z(val[2]) {}

  __host__ __device__ __forceinline__ explicit vec3(const float3 &val) : x(val.x), y(val.y), z(val.z) {}
  __host__ __device__ __forceinline__ explicit vec3(const glm::vec3 &val) : x(val.x), y(val.y), z(val.z) {}

  __host__ __device__ __forceinline__ explicit vec3(const float4 &val) : x(val.x), y(val.y), z(val.z) {}
  __host__ __device__ __forceinline__ explicit vec3(const glm::vec4 &val) : x(val.x), y(val.y), z(val.z) {}
  __host__ __device__ explicit vec3(const vec4 &val);

  __host__ __device__ explicit vec3(const ivec3 &val);

  __host__ __device__ __forceinline__ explicit operator float3() const { return {x, y, z}; }
  __host__ __device__ explicit operator ivec3() const;

  // Addition
  __host__ __device__ __forceinline__ vec3 operator+(const vec3 &other) const {
    return {x + other.x, y + other.y, z + other.z};
  }
  __host__ __device__ __forceinline__ vec3 &operator+=(const vec3 &other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
  }
  __host__ __device__ __forceinline__ vec3 &operator+=(const float &val) {
    x += val;
    y += val;
    z += val;
    return *this;
  }
  __host__ __device__ __forceinline__ vec3 operator+(float val) const { return {x + val, y + val, z + val}; }

  // Subtraction
  __host__ __device__ __forceinline__ vec3 operator-(const vec3 &other) const {
    return {x - other.x, y - other.y, z - other.z};
  }
  __host__ __device__ __forceinline__ vec3 &operator-=(const vec3 &other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
  }
  __host__ __device__ __forceinline__ vec3 &operator-=(const float &val) {
    x -= val;
    y -= val;
    z -= val;
    return *this;
  }
  __host__ __device__ __forceinline__ vec3 operator-(float val) const { return {x - val, y - val, z - val}; }
  __host__ __device__ __forceinline__ vec3 operator-() const { return {-x, -y, -z}; }

  // Multiplication
  __host__ __device__ __forceinline__ vec3 operator*(const vec3 &other) const {
    return {x * other.x, y * other.y, z * other.z};
  }
  __host__ __device__ __forceinline__ vec3 operator*(float val) const { return {x * val, y * val, z * val}; }
  __host__ __device__ __forceinline__ vec3 &operator*=(float val) {
    x *= val;
    y *= val;
    z *= val;
    return *this;
  }

  // Division
  __host__ __device__ __forceinline__ vec3 operator/(const vec3 &other) const {
    return {x / other.x, y / other.y, z / other.z};
  }
  __host__ __device__ __forceinline__ vec3 operator/(float val) const { return {x / val, y / val, z / val}; }
  __host__ __device__ __forceinline__ vec3 &operator/=(float val) {
    x /= val;
    y /= val;
    z /= val;
    return *this;
  }

  // Utility functions
  __host__ __device__ [[nodiscard]] __forceinline__ float length() const {
    return gsm::sqrt(x * x + y * y + z * z);
  }

  __host__ __device__ [[nodiscard]] __forceinline__ float lengthSquared() const {
    return x * x + y * y + z * z;
  }

  __host__ __device__ [[nodiscard]] __forceinline__ vec3 normalized() const {
    float len = length();
    return {x / len, y / len, z / len};
  }

  __host__ __device__ __forceinline__ void normalize() {
    float len = length();
    x /= len;
    y /= len;
    z /= len;
  }

  __host__ __device__ [[nodiscard]] __forceinline__ float dot(const vec3 &other) const {
    return x * other.x + y * other.y + z * other.z;
  }

  __host__ __device__ [[nodiscard]] __forceinline__ vec3 cross(const vec3 &other) const {
    return {y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x};
  }

  __host__ __device__ __forceinline__ float &operator[](int i) {
    assert(i >= 0 && i < 3);
    if (i == 0)
      return x;
    else if (i == 1)
      return y;
    else // i == 2
      return z;
  }
  __host__ __device__ __forceinline__ const float &operator[](int i) const {
    assert(i >= 0 && i < 3);
    if (i == 0)
      return x;
    else if (i == 1)
      return y;
    else // i == 2
      return z;
  }
};

// Scalar * vec3
__host__ __device__ __forceinline__ vec3 operator*(float val, const vec3 &v) {
  return {v.x * val, v.y * val, v.z * val};
}

} // namespace gsm

namespace gsm {

struct vec4 {
  float x, y, z, w;

  // Constructors
  __host__ __device__ __forceinline__ vec4() : x(0), y(0), z(0), w(0) {}
  __host__ __device__ __forceinline__ explicit vec4(const float &val) : x(val), y(val), z(val), w(val) {}
  __host__ __device__ __forceinline__ vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
  __host__ __device__ __forceinline__ explicit vec4(const float *val)
      : x(val[0]), y(val[1]), z(val[2]), w(val[3]) {}

  __host__ __device__ __forceinline__ explicit vec4(const glm::vec4 &val)
      : x(val.x), y(val.y), z(val.z), w(val.w) {}

  __host__ __device__ __forceinline__ explicit vec4(const vec3 &val) : x(val.x), y(val.y), z(val.z), w(0.0) {}
  __host__ __device__ __forceinline__ explicit vec4(const vec3 &val, const float &w)
      : x(val.x), y(val.y), z(val.z), w(w) {}

  __host__ __device__ __forceinline__ explicit vec4(const float3 &val)
      : x(val.x), y(val.y), z(val.z), w(0.0) {}
  __host__ __device__ __forceinline__ explicit vec4(const float3 &val, const float &w)
      : x(val.x), y(val.y), z(val.z), w(w) {}
  __host__ __device__ __forceinline__ explicit vec4(const float4 &val)
      : x(val.x), y(val.y), z(val.z), w(val.w) {}

  __host__ __device__ explicit vec4(const ivec4 &val);

  __host__ __device__ __forceinline__ explicit operator vec3() const { return {x, y, z}; }
  __host__ __device__ __forceinline__ explicit operator float3() const { return {x, y, z}; }
  __host__ __device__ __forceinline__ explicit operator float4() const { return {x, y, z, w}; }
  __host__ __device__ explicit operator ivec4() const;

  // Addition
  __host__ __device__ __forceinline__ vec4 operator+(const vec4 &other) const {
    return {x + other.x, y + other.y, z + other.z, w + other.w};
  }
  __host__ __device__ __forceinline__ vec4 &operator+=(const vec4 &other) {
    x += other.x;
    y += other.y;
    z += other.z;
    w += other.w;
    return *this;
  }
  __host__ __device__ __forceinline__ vec4 &operator+=(const float &val) {
    x += val;
    y += val;
    z += val;
    w += val;
    return *this;
  }
  __host__ __device__ __forceinline__ vec4 operator+(float val) const {
    return {x + val, y + val, z + val, w + val};
  }

  // Subtraction
  __host__ __device__ __forceinline__ vec4 operator-(const vec4 &other) const {
    return {x - other.x, y - other.y, z - other.z, w - other.w};
  }
  __host__ __device__ __forceinline__ vec4 &operator-=(const vec4 &other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    w -= other.w;
    return *this;
  }
  __host__ __device__ __forceinline__ vec4 &operator-=(const float &val) {
    x -= val;
    y -= val;
    z -= val;
    w -= val;
    return *this;
  }
  __host__ __device__ __forceinline__ vec4 operator-(float val) const {
    return {x - val, y - val, z - val, w - val};
  }
  __host__ __device__ __forceinline__ vec4 operator-() const { return {-x, -y, -z, -w}; }

  // Multiplication
  __host__ __device__ __forceinline__ vec4 operator*(const vec4 &other) const {
    return {x * other.x, y * other.y, z * other.z, w * other.w};
  }
  __host__ __device__ __forceinline__ vec4 operator*(float val) const {
    return {x * val, y * val, z * val, w * val};
  }
  __host__ __device__ __forceinline__ vec4 &operator*=(float val) {
    x *= val;
    y *= val;
    z *= val;
    w *= val;
    return *this;
  }

  // Division
  __host__ __device__ __forceinline__ vec4 operator/(const vec4 &other) const {
    return {x / other.x, y / other.y, z / other.z, w / other.w};
  }
  __host__ __device__ __forceinline__ vec4 operator/(float val) const {
    return {x / val, y / val, z / val, w / val};
  }
  __host__ __device__ __forceinline__ vec4 &operator/=(float val) {
    x /= val;
    y /= val;
    z /= val;
    w /= val;
    return *this;
  }

  // Utility functions
  __host__ __device__ [[nodiscard]] __forceinline__ float length() const {
    return gsm::sqrt(x * x + y * y + z * z + w * w);
  }

  __host__ __device__ [[nodiscard]] __forceinline__ float lengthSquared() const {
    return x * x + y * y + z * z + w * w;
  }

  __host__ __device__ [[nodiscard]] __forceinline__ vec4 normalized() const {
    float len = length();
    return {x / len, y / len, z / len, w / len};
  }

  __host__ __device__ __forceinline__ void normalize() {
    float len = length();
    x /= len;
    y /= len;
    z /= len;
    w /= len;
  }

  __host__ __device__ [[nodiscard]] __forceinline__ float dot(const vec4 &other) const {
    return x * other.x + y * other.y + z * other.z + w * other.w;
  }

  __host__ __device__ __forceinline__ float &operator[](int i) {
    assert(i >= 0 && i < 4);
    if (i == 0)
      return x;
    else if (i == 1)
      return y;
    else if (i == 2)
      return z;
    else // i == 3
      return w;
  }
  __host__ __device__ __forceinline__ const float &operator[](int i) const {
    assert(i >= 0 && i < 4);
    if (i == 0)
      return x;
    else if (i == 1)
      return y;
    else if (i == 2)
      return z;
    else // i == 3
      return w;
  }
};

// Scalar * vec4
__host__ __device__ __forceinline__ vec4 operator*(float val, const vec4 &v) {
  return {v.x * val, v.y * val, v.z * val, v.w * val};
}

} // namespace gsm

namespace gsm {

__host__ __device__ __forceinline__ vec2::vec2(const vec3 &val) : x(val.x), y(val.y) {}
__host__ __device__ __forceinline__ vec2::vec2(const vec4 &val) : x(val.x), y(val.y) {}
__host__ __device__ __forceinline__ vec3::vec3(const vec4 &val) : x(val.x), y(val.y), z(val.z) {}

} // namespace gsm

#endif // !GSM_VECTOR_CUH
