
#ifndef RASTERIZER_VECTOR_HPP
#define RASTERIZER_VECTOR_HPP
#pragma once

#include <cmath>
#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace rasterizer {

__host__ __device__ inline float max(float a, float b) { return fmaxf(a, b); }
__host__ __device__ inline int max(int a, int b) { return a > b ? a : b; }

__host__ __device__ inline float min(float a, float b) { return fminf(a, b); }
__host__ __device__ inline int min(int a, int b) { return a < b ? a : b; }

__host__ __device__ inline int floor(float b) { return static_cast<int>(floorf(b)); }
__host__ __device__ inline int ceil(float b) { return static_cast<int>(ceilf(b)); }

struct vec2 {
  float x, y;

  // Constructors
  __host__ __device__ vec2() : x(0), y(0) {}
  __host__ __device__ explicit vec2(const float &val) : x(val), y(val) {}
  __host__ __device__ vec2(const float &x, const float &y) : x(x), y(y) {}

  __host__ __device__ explicit vec2(const float2 &val) : x(val.x), y(val.y) {}
  __host__ __device__ explicit vec2(const glm::vec2 &val) : x(val.x), y(val.y) {}

  __host__ __device__ explicit operator float2() const { return {x, y}; }

  // Addition
  __host__ __device__ vec2 operator+(const vec2 &other) const { return {x + other.x, y + other.y}; }
  __host__ __device__ vec2 &operator+=(const vec2 &other) {
    x += other.x;
    y += other.y;
    return *this;
  }
  __host__ __device__ vec2 &operator+=(const float &val) {
    x += val;
    y += val;
    return *this;
  }
  __host__ __device__ vec2 operator+(float val) const { return {x + val, y + val}; }

  // Subtraction
  __host__ __device__ vec2 operator-(const vec2 &other) const { return {x - other.x, y - other.y}; }
  __host__ __device__ vec2 &operator-=(const vec2 &other) {
    x -= other.x;
    y -= other.y;
    return *this;
  }
  __host__ __device__ vec2 &operator-=(const float &val) {
    x -= val;
    y -= val;
    return *this;
  }
  __host__ __device__ vec2 operator-(float val) const { return {x - val, y - val}; }
  __host__ __device__ vec2 operator-() const { return {-x, -y}; }

  // Multiplication
  __host__ __device__ vec2 operator*(const vec2 &other) const { return {x * other.x, y * other.y}; }
  __host__ __device__ vec2 operator*(float val) const { return {x * val, y * val}; }
  __host__ __device__ vec2 &operator*=(float val) {
    x *= val;
    y *= val;
    return *this;
  }

  // Division
  __host__ __device__ vec2 operator/(const vec2 &other) const { return {x / other.x, y / other.y}; }
  __host__ __device__ vec2 operator/(float val) const { return {x / val, y / val}; }
  __host__ __device__ vec2 &operator/=(float val) {
    x /= val;
    y /= val;
    return *this;
  }

  // Utility functions
  __host__ __device__ [[nodiscard]] float length() const { return sqrtf(x * x + y * y); }

  __host__ __device__ [[nodiscard]] float lengthSquared() const { return x * x + y * y; }

  __host__ __device__ [[nodiscard]] vec2 normalized() const {
    float len = length();
    return {x / len, y / len};
  }

  __host__ __device__ void normalize() {
    float len = length();
    x /= len;
    y /= len;
  }

  __host__ __device__ [[nodiscard]] float dot(const vec2 &other) const { return x * other.x + y * other.y; }

  __host__ __device__ float &operator[](int i) {
    assert(i >= 0 && i < 2);
    if (i == 0)
      return x;
    else
      return y;
  }
};

// Scalar * vec2
__host__ __device__ inline vec2 operator*(float val, const vec2 &v) { return {v.x * val, v.y * val}; }

// Component-wise max/min
__host__ __device__ inline vec2 max(const vec2 &a, const vec2 &b) { return {max(a.x, b.x), max(a.y, b.y)}; }
__host__ __device__ inline vec2 max(const vec2 &a, float val) { return {max(a.x, val), max(a.y, val)}; }

__host__ __device__ inline vec2 min(const vec2 &a, const vec2 &b) { return {min(a.x, b.x), min(a.y, b.y)}; }

__host__ __device__ inline vec2 min(const vec2 &a, float val) { return {min(a.x, val), min(a.y, val)}; }

struct vec3 {
  float x, y, z;

  // Constructors
  __host__ __device__ vec3() : x(0), y(0), z(0) {}
  __host__ __device__ explicit vec3(const float &val) : x(val), y(val), z(val) {}
  __host__ __device__ vec3(const float &x, const float &y, const float &z) : x(x), y(y), z(z) {}

  __host__ __device__ explicit vec3(const float3 &val) : x(val.x), y(val.y), z(val.z) {}
  __host__ __device__ explicit vec3(const glm::vec3 &val) : x(val.x), y(val.y), z(val.z) {}

  __host__ __device__ explicit operator float3() const { return {x, y, z}; }

  // Addition
  __host__ __device__ vec3 operator+(const vec3 &other) const {
    return {x + other.x, y + other.y, z + other.z};
  }
  __host__ __device__ vec3 &operator+=(const vec3 &other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
  }
  __host__ __device__ vec3 &operator+=(const float &val) {
    x += val;
    y += val;
    z += val;
    return *this;
  }
  __host__ __device__ vec3 operator+(float val) const { return {x + val, y + val, z + val}; }

  // Subtraction
  __host__ __device__ vec3 operator-(const vec3 &other) const {
    return {x - other.x, y - other.y, z - other.z};
  }
  __host__ __device__ vec3 &operator-=(const vec3 &other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
  }
  __host__ __device__ vec3 &operator-=(const float &val) {
    x -= val;
    y -= val;
    z -= val;
    return *this;
  }
  __host__ __device__ vec3 operator-(float val) const { return {x - val, y - val, z - val}; }
  __host__ __device__ vec3 operator-() const { return {-x, -y, -z}; }

  // Multiplication
  __host__ __device__ vec3 operator*(const vec3 &other) const {
    return {x * other.x, y * other.y, z * other.z};
  }
  __host__ __device__ vec3 operator*(float val) const { return {x * val, y * val, z * val}; }
  __host__ __device__ vec3 &operator*=(float val) {
    x *= val;
    y *= val;
    z *= val;
    return *this;
  }

  // Division
  __host__ __device__ vec3 operator/(const vec3 &other) const {
    return {x / other.x, y / other.y, z / other.z};
  }
  __host__ __device__ vec3 operator/(float val) const { return {x / val, y / val, z / val}; }
  __host__ __device__ vec3 &operator/=(float val) {
    x /= val;
    y /= val;
    z /= val;
    return *this;
  }

  // Utility functions
  __host__ __device__ [[nodiscard]] float length() const { return sqrtf(x * x + y * y + z * z); }

  __host__ __device__ [[nodiscard]] float lengthSquared() const { return x * x + y * y + z * z; }

  __host__ __device__ [[nodiscard]] vec3 normalized() const {
    float len = length();
    return {x / len, y / len, z / len};
  }

  __host__ __device__ void normalize() {
    float len = length();
    x /= len;
    y /= len;
    z /= len;
  }

  __host__ __device__ [[nodiscard]] float dot(const vec3 &other) const {
    return x * other.x + y * other.y + z * other.z;
  }

  __host__ __device__ [[nodiscard]] vec3 cross(const vec3 &other) const {
    return {y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x};
  }

  __host__ __device__ float &operator[](int i) {
    assert(i >= 0 && i < 3);
    if (i == 0)
      return x;
    else if (i == 1)
      return y;
    else
      return z;
  }
};

// Scalar * vec3
__host__ __device__ inline vec3 operator*(float val, const vec3 &v) {
  return {v.x * val, v.y * val, v.z * val};
}

// Component-wise max/min
__host__ __device__ inline vec3 max(const vec3 &a, const vec3 &b) {
  return {max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)};
}
__host__ __device__ inline vec3 max(const vec3 &a, float val) {
  return {max(a.x, val), max(a.y, val), max(a.z, val)};
}

__host__ __device__ inline vec3 min(const vec3 &a, const vec3 &b) {
  return {min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)};
}

__host__ __device__ inline vec3 min(const vec3 &a, float val) {
  return {min(a.x, val), min(a.y, val), min(a.z, val)};
}

struct vec4 {
  float x, y, z, w;

  // Constructors
  __host__ __device__ vec4() : x(0), y(0), z(0), w(0) {}
  __host__ __device__ explicit vec4(const float &val) : x(val), y(val), z(val), w(val) {}
  __host__ __device__ vec4(const float &x, const float &y, const float &z, const float &w)
      : x(x), y(y), z(z), w(w) {}

  __host__ __device__ explicit vec4(const glm::vec4 &val) : x(val.x), y(val.y), z(val.z), w(val.w) {}

  __host__ __device__ explicit vec4(const vec3 &val) : x(val.x), y(val.y), z(val.z), w(0.0) {}
  __host__ __device__ explicit vec4(const vec3 &val, const float &w) : x(val.x), y(val.y), z(val.z), w(w) {}

  __host__ __device__ explicit vec4(const float3 &val) : x(val.x), y(val.y), z(val.z), w(0.0) {}
  __host__ __device__ explicit vec4(const float3 &val, const float &w) : x(val.x), y(val.y), z(val.z), w(w) {}
  __host__ __device__ explicit vec4(const float4 &val) : x(val.x), y(val.y), z(val.z), w(val.w) {}

  __host__ __device__ explicit operator vec3() const { return {x, y, z}; }
  __host__ __device__ explicit operator float3() const { return {x, y, z}; }
  __host__ __device__ explicit operator float4() const { return {x, y, z, w}; }

  // Addition
  __host__ __device__ vec4 operator+(const vec4 &other) const {
    return {x + other.x, y + other.y, z + other.z, w + other.w};
  }
  __host__ __device__ vec4 &operator+=(const vec4 &other) {
    x += other.x;
    y += other.y;
    z += other.z;
    w += other.w;
    return *this;
  }
  __host__ __device__ vec4 &operator+=(const float &val) {
    x += val;
    y += val;
    z += val;
    w += val;
    return *this;
  }
  __host__ __device__ vec4 operator+(float val) const { return {x + val, y + val, z + val, w + val}; }

  // Subtraction
  __host__ __device__ vec4 operator-(const vec4 &other) const {
    return {x - other.x, y - other.y, z - other.z, w - other.w};
  }
  __host__ __device__ vec4 &operator-=(const vec4 &other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    w -= other.w;
    return *this;
  }
  __host__ __device__ vec4 &operator-=(const float &val) {
    x -= val;
    y -= val;
    z -= val;
    w -= val;
    return *this;
  }
  __host__ __device__ vec4 operator-(float val) const { return {x - val, y - val, z - val, w - val}; }
  __host__ __device__ vec4 operator-() const { return {-x, -y, -z, -w}; }

  // Multiplication
  __host__ __device__ vec4 operator*(const vec4 &other) const {
    return {x * other.x, y * other.y, z * other.z, w * other.w};
  }
  __host__ __device__ vec4 operator*(float val) const { return {x * val, y * val, z * val, w * val}; }
  __host__ __device__ vec4 &operator*=(float val) {
    x *= val;
    y *= val;
    z *= val;
    w *= val;
    return *this;
  }

  // Division
  __host__ __device__ vec4 operator/(const vec4 &other) const {
    return {x / other.x, y / other.y, z / other.z, w / other.w};
  }
  __host__ __device__ vec4 operator/(float val) const { return {x / val, y / val, z / val, w / val}; }
  __host__ __device__ vec4 &operator/=(float val) {
    x /= val;
    y /= val;
    z /= val;
    w /= val;
    return *this;
  }

  // Utility functions
  __host__ __device__ [[nodiscard]] float length() const { return sqrtf(x * x + y * y + z * z + w * w); }

  __host__ __device__ [[nodiscard]] float lengthSquared() const { return x * x + y * y + z * z + w * w; }

  __host__ __device__ [[nodiscard]] vec4 normalized() const {
    float len = length();
    return {x / len, y / len, z / len, w / len};
  }

  __host__ __device__ void normalize() {
    float len = length();
    x /= len;
    y /= len;
    z /= len;
    w /= len;
  }

  __host__ __device__ [[nodiscard]] float dot(const vec4 &other) const {
    return x * other.x + y * other.y + z * other.z + w * other.w;
  }

  __host__ __device__ float &operator[](int i) {
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
__host__ __device__ inline vec4 operator*(float val, const vec4 &v) {
  return {v.x * val, v.y * val, v.z * val, v.w * val};
}

// Component-wise max/min
__host__ __device__ inline vec4 max(const vec4 &a, const vec4 &b) {
  return {max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w)};
}
__host__ __device__ inline vec4 max(const vec4 &a, float val) {
  return {max(a.x, val), max(a.y, val), max(a.z, val), max(a.w, val)};
}
__host__ __device__ inline vec4 min(const vec4 &a, const vec4 &b) {
  return {min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w)};
}
__host__ __device__ inline vec4 min(const vec4 &a, float val) {
  return {min(a.x, val), min(a.y, val), min(a.z, val), min(a.w, val)};
}

__host__ __device__ inline float length(const vec3 &v) { return v.length(); }
__host__ __device__ inline float length(const vec4 &v) { return v.length(); }

} // namespace rasterizer

#endif // !RASTERIZER_VECTOR_HPP
