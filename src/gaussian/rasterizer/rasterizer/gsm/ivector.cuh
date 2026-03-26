#ifndef GSM_IVECTOR_CUH
#define GSM_IVECTOR_CUH
#pragma once

#include <cassert>

#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include "base.cuh"
#include "vector.cuh"

namespace gsm {
struct ivec2;
struct ivec3;
struct ivec4;
} // namespace gsm

namespace gsm {

struct ivec2 {
  int x, y;

  // Constructors
  __host__ __device__ __forceinline__ ivec2() : x(0), y(0) {}
  __host__ __device__ __forceinline__ explicit ivec2(const int &val) : x(val), y(val) {}
  __host__ __device__ __forceinline__ ivec2(const int &x, const int &y) : x(x), y(y) {}
  __host__ __device__ __forceinline__ explicit ivec2(const int *val) : x(val[0]), y(val[1]) {}

  __host__ __device__ __forceinline__ explicit ivec2(const int2 &val) : x(val.x), y(val.y) {}
  __host__ __device__ __forceinline__ explicit ivec2(const glm::ivec2 &val) : x(val.x), y(val.y) {}

  __host__ __device__ __forceinline__ explicit ivec2(const int3 &val) : x(val.x), y(val.y) {}
  __host__ __device__ __forceinline__ explicit ivec2(const glm::ivec3 &val) : x(val.x), y(val.y) {}
  __host__ __device__ explicit ivec2(const ivec3 &val);

  __host__ __device__ __forceinline__ explicit ivec2(const int4 &val) : x(val.x), y(val.y) {}
  __host__ __device__ __forceinline__ explicit ivec2(const glm::ivec4 &val) : x(val.x), y(val.y) {}
  __host__ __device__ explicit ivec2(const ivec4 &val);

  __host__ __device__ __forceinline__ explicit ivec2(const vec2 &val) : x(val.x), y(val.y) {}

  __host__ __device__ __forceinline__ explicit operator int2() const { return {x, y}; }
  __host__ __device__ __forceinline__ explicit operator vec2() const {
    return {static_cast<float>(x), static_cast<float>(y)};
  }

  // Addition
  __host__ __device__ __forceinline__ ivec2 operator+(const ivec2 &other) const {
    return {x + other.x, y + other.y};
  }
  __host__ __device__ __forceinline__ ivec2 &operator+=(const ivec2 &other) {
    x += other.x;
    y += other.y;
    return *this;
  }
  __host__ __device__ __forceinline__ ivec2 &operator+=(const int &val) {
    x += val;
    y += val;
    return *this;
  }
  __host__ __device__ __forceinline__ ivec2 operator+(int val) const { return {x + val, y + val}; }

  // Subtraction
  __host__ __device__ __forceinline__ ivec2 operator-(const ivec2 &other) const {
    return {x - other.x, y - other.y};
  }
  __host__ __device__ __forceinline__ ivec2 &operator-=(const ivec2 &other) {
    x -= other.x;
    y -= other.y;
    return *this;
  }
  __host__ __device__ __forceinline__ ivec2 &operator-=(const int &val) {
    x -= val;
    y -= val;
    return *this;
  }
  __host__ __device__ __forceinline__ ivec2 operator-(int val) const { return {x - val, y - val}; }
  __host__ __device__ __forceinline__ ivec2 operator-() const { return {-x, -y}; }

  // Multiplication
  __host__ __device__ __forceinline__ ivec2 operator*(const ivec2 &other) const {
    return {x * other.x, y * other.y};
  }
  __host__ __device__ __forceinline__ ivec2 operator*(int val) const { return {x * val, y * val}; }
  __host__ __device__ __forceinline__ ivec2 &operator*=(int val) {
    x *= val;
    y *= val;
    return *this;
  }

  // Division
  __host__ __device__ __forceinline__ ivec2 operator/(const ivec2 &other) const {
    return {x / other.x, y / other.y};
  }
  __host__ __device__ __forceinline__ ivec2 operator/(int val) const { return {x / val, y / val}; }
  __host__ __device__ __forceinline__ ivec2 &operator/=(int val) {
    x /= val;
    y /= val;
    return *this;
  }

  // Utility functions
  __host__ __device__ [[nodiscard]] __forceinline__ float length() const { return gsm::sqrt(x * x + y * y); }

  __host__ __device__ [[nodiscard]] __forceinline__ int lengthSquared() const { return x * x + y * y; }

  __host__ __device__ [[nodiscard]] __forceinline__ vec2 normalized() const {
    float len = length();
    return {x / len, y / len};
  }

  __host__ __device__ [[nodiscard]] __forceinline__ int dot(const ivec2 &other) const {
    return x * other.x + y * other.y;
  }

  __host__ __device__ __forceinline__ int &operator[](int i) {
    assert(i >= 0 && i < 2);
    if (i == 0)
      return x;
    else // i == 1
      return y;
  }
  __host__ __device__ __forceinline__ const int &operator[](int i) const {
    assert(i >= 0 && i < 3);
    if (i == 0)
      return x;
    else // i == 1
      return y;
  }
};

// Scalar * ivec2
__host__ __device__ __forceinline__ ivec2 operator*(int val, const ivec2 &v) {
  return {v.x * val, v.y * val};
}

} // namespace gsm

namespace gsm {

struct ivec3 {
  int x, y, z;

  // Constructors
  __host__ __device__ __forceinline__ ivec3() : x(0), y(0), z(0) {}
  __host__ __device__ __forceinline__ explicit ivec3(const int &val) : x(val), y(val), z(val) {}
  __host__ __device__ __forceinline__ ivec3(const int &x, const int &y, const int &z) : x(x), y(y), z(z) {}
  __host__ __device__ __forceinline__ explicit ivec3(const int *val) : x(val[0]), y(val[1]), z(val[2]) {}

  __host__ __device__ __forceinline__ explicit ivec3(const int3 &val) : x(val.x), y(val.y), z(val.z) {}
  __host__ __device__ __forceinline__ explicit ivec3(const glm::ivec3 &val) : x(val.x), y(val.y), z(val.z) {}

  __host__ __device__ __forceinline__ explicit ivec3(const int4 &val) : x(val.x), y(val.y), z(val.z) {}
  __host__ __device__ __forceinline__ explicit ivec3(const glm::ivec4 &val) : x(val.x), y(val.y), z(val.z) {}
  __host__ __device__ explicit ivec3(const ivec4 &val);

  __host__ __device__ __forceinline__ explicit ivec3(const vec3 &val) : x(val.x), y(val.y), z(val.z) {}

  __host__ __device__ __forceinline__ explicit operator int3() const { return {x, y, z}; }
  __host__ __device__ __forceinline__ explicit operator vec3() const {
    return {static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
  }

  // Addition
  __host__ __device__ __forceinline__ ivec3 operator+(const ivec3 &other) const {
    return {x + other.x, y + other.y, z + other.z};
  }
  __host__ __device__ __forceinline__ ivec3 &operator+=(const ivec3 &other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
  }
  __host__ __device__ __forceinline__ ivec3 &operator+=(const int &val) {
    x += val;
    y += val;
    z += val;
    return *this;
  }
  __host__ __device__ __forceinline__ ivec3 operator+(int val) const { return {x + val, y + val, z + val}; }

  // Subtraction
  __host__ __device__ __forceinline__ ivec3 operator-(const ivec3 &other) const {
    return {x - other.x, y - other.y, z - other.z};
  }
  __host__ __device__ __forceinline__ ivec3 &operator-=(const ivec3 &other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
  }
  __host__ __device__ __forceinline__ ivec3 &operator-=(const int &val) {
    x -= val;
    y -= val;
    z -= val;
    return *this;
  }
  __host__ __device__ __forceinline__ ivec3 operator-(int val) const { return {x - val, y - val, z - val}; }
  __host__ __device__ __forceinline__ ivec3 operator-() const { return {-x, -y, -z}; }

  // Multiplication
  __host__ __device__ __forceinline__ ivec3 operator*(const ivec3 &other) const {
    return {x * other.x, y * other.y, z * other.z};
  }
  __host__ __device__ __forceinline__ ivec3 operator*(int val) const { return {x * val, y * val, z * val}; }
  __host__ __device__ __forceinline__ ivec3 &operator*=(int val) {
    x *= val;
    y *= val;
    z *= val;
    return *this;
  }

  // Division
  __host__ __device__ __forceinline__ ivec3 operator/(const ivec3 &other) const {
    return {x / other.x, y / other.y, z / other.z};
  }
  __host__ __device__ __forceinline__ ivec3 operator/(int val) const { return {x / val, y / val, z / val}; }
  __host__ __device__ __forceinline__ ivec3 &operator/=(int val) {
    x /= val;
    y /= val;
    z /= val;
    return *this;
  }

  // Utility functions
  __host__ __device__ [[nodiscard]] __forceinline__ float length() const {
    return gsm::sqrt(x * x + y * y + z * z);
  }

  __host__ __device__ [[nodiscard]] __forceinline__ int lengthSquared() const {
    return x * x + y * y + z * z;
  }

  __host__ __device__ [[nodiscard]] __forceinline__ vec3 normalized() const {
    float len = length();
    return {x / len, y / len, z / len};
  }

  __host__ __device__ [[nodiscard]] __forceinline__ int dot(const ivec3 &other) const {
    return x * other.x + y * other.y + z * other.z;
  }

  __host__ __device__ [[nodiscard]] __forceinline__ ivec3 cross(const ivec3 &other) const {
    return {y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x};
  }

  __host__ __device__ __forceinline__ int &operator[](int i) {
    assert(i >= 0 && i < 3);
    if (i == 0)
      return x;
    else if (i == 1)
      return y;
    else // i == 2
      return z;
  }
  __host__ __device__ __forceinline__ const int &operator[](int i) const {
    assert(i >= 0 && i < 3);
    if (i == 0)
      return x;
    else if (i == 1)
      return y;
    else // i == 2
      return z;
  }
};

// Scalar * ivec3
__host__ __device__ __forceinline__ ivec3 operator*(int val, const ivec3 &v) {
  return {v.x * val, v.y * val, v.z * val};
}

} // namespace gsm

namespace gsm {

struct ivec4 {
  int x, y, z, w;

  // Constructors
  __host__ __device__ __forceinline__ ivec4() : x(0), y(0), z(0), w(0) {}
  __host__ __device__ __forceinline__ explicit ivec4(const int &val) : x(val), y(val), z(val), w(val) {}
  __host__ __device__ __forceinline__ ivec4(int x, int y, int z, int w) : x(x), y(y), z(z), w(w) {}
  __host__ __device__ __forceinline__ explicit ivec4(const int *val)
      : x(val[0]), y(val[1]), z(val[2]), w(val[3]) {}

  __host__ __device__ __forceinline__ explicit ivec4(const glm::ivec4 &val)
      : x(val.x), y(val.y), z(val.z), w(val.w) {}

  __host__ __device__ __forceinline__ explicit ivec4(const ivec3 &val)
      : x(val.x), y(val.y), z(val.z), w(0.0) {}
  __host__ __device__ __forceinline__ explicit ivec4(const ivec3 &val, const int &w)
      : x(val.x), y(val.y), z(val.z), w(w) {}

  __host__ __device__ __forceinline__ explicit ivec4(const int3 &val)
      : x(val.x), y(val.y), z(val.z), w(0.0) {}
  __host__ __device__ __forceinline__ explicit ivec4(const int3 &val, const int &w)
      : x(val.x), y(val.y), z(val.z), w(w) {}
  __host__ __device__ __forceinline__ explicit ivec4(const int4 &val)
      : x(val.x), y(val.y), z(val.z), w(val.w) {}

  __host__ __device__ __forceinline__ explicit ivec4(const vec4 &val)
      : x(val.x), y(val.y), z(val.z), w(val.w) {}

  __host__ __device__ __forceinline__ explicit operator ivec3() const { return {x, y, z}; }
  __host__ __device__ __forceinline__ explicit operator int3() const { return {x, y, z}; }
  __host__ __device__ __forceinline__ explicit operator int4() const { return {x, y, z, w}; }
  __host__ __device__ __forceinline__ explicit operator vec4() const {
    return {static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), static_cast<float>(w)};
  }

  // Addition
  __host__ __device__ __forceinline__ ivec4 operator+(const ivec4 &other) const {
    return {x + other.x, y + other.y, z + other.z, w + other.w};
  }
  __host__ __device__ __forceinline__ ivec4 &operator+=(const ivec4 &other) {
    x += other.x;
    y += other.y;
    z += other.z;
    w += other.w;
    return *this;
  }
  __host__ __device__ __forceinline__ ivec4 &operator+=(const int &val) {
    x += val;
    y += val;
    z += val;
    w += val;
    return *this;
  }
  __host__ __device__ __forceinline__ ivec4 operator+(int val) const {
    return {x + val, y + val, z + val, w + val};
  }

  // Subtraction
  __host__ __device__ __forceinline__ ivec4 operator-(const ivec4 &other) const {
    return {x - other.x, y - other.y, z - other.z, w - other.w};
  }
  __host__ __device__ __forceinline__ ivec4 &operator-=(const ivec4 &other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    w -= other.w;
    return *this;
  }
  __host__ __device__ __forceinline__ ivec4 &operator-=(const int &val) {
    x -= val;
    y -= val;
    z -= val;
    w -= val;
    return *this;
  }
  __host__ __device__ __forceinline__ ivec4 operator-(int val) const {
    return {x - val, y - val, z - val, w - val};
  }
  __host__ __device__ __forceinline__ ivec4 operator-() const { return {-x, -y, -z, -w}; }

  // Multiplication
  __host__ __device__ __forceinline__ ivec4 operator*(const ivec4 &other) const {
    return {x * other.x, y * other.y, z * other.z, w * other.w};
  }
  __host__ __device__ __forceinline__ ivec4 operator*(int val) const {
    return {x * val, y * val, z * val, w * val};
  }
  __host__ __device__ __forceinline__ ivec4 &operator*=(int val) {
    x *= val;
    y *= val;
    z *= val;
    w *= val;
    return *this;
  }

  // Division
  __host__ __device__ __forceinline__ ivec4 operator/(const ivec4 &other) const {
    return {x / other.x, y / other.y, z / other.z, w / other.w};
  }
  __host__ __device__ __forceinline__ ivec4 operator/(int val) const {
    return {x / val, y / val, z / val, w / val};
  }
  __host__ __device__ __forceinline__ ivec4 &operator/=(int val) {
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

  __host__ __device__ [[nodiscard]] __forceinline__ int lengthSquared() const {
    return x * x + y * y + z * z + w * w;
  }

  __host__ __device__ [[nodiscard]] __forceinline__ vec4 normalized() const {
    float len = length();
    return {x / len, y / len, z / len, w / len};
  }

  __host__ __device__ [[nodiscard]] __forceinline__ int dot(const ivec4 &other) const {
    return x * other.x + y * other.y + z * other.z + w * other.w;
  }

  __host__ __device__ __forceinline__ int &operator[](int i) {
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
  __host__ __device__ __forceinline__ const int &operator[](int i) const {
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

// Scalar * ivec4
__host__ __device__ __forceinline__ ivec4 operator*(int val, const ivec4 &v) {
  return {v.x * val, v.y * val, v.z * val, v.w * val};
}

} // namespace gsm

namespace gsm {

__host__ __device__ __forceinline__ ivec2::ivec2(const ivec3 &val) : x(val.x), y(val.y) {}
__host__ __device__ __forceinline__ ivec2::ivec2(const ivec4 &val) : x(val.x), y(val.y) {}
__host__ __device__ __forceinline__ ivec3::ivec3(const ivec4 &val) : x(val.x), y(val.y), z(val.z) {}

__host__ __device__ __forceinline__ vec2::vec2(const ivec2 &val) : x(val.x), y(val.y) {}
__host__ __device__ __forceinline__ vec3::vec3(const ivec3 &val) : x(val.x), y(val.y), z(val.z) {}
__host__ __device__ __forceinline__ vec4::vec4(const ivec4 &val) : x(val.x), y(val.y), z(val.z), w(val.w) {}

__host__ __device__ __forceinline__ vec2::operator ivec2() const {
  return {static_cast<int>(x), static_cast<int>(y)};
}
__host__ __device__ __forceinline__ vec3::operator ivec3() const {
  return {static_cast<int>(x), static_cast<int>(y), static_cast<int>(z)};
}
__host__ __device__ __forceinline__ vec4::operator ivec4() const {
  return {static_cast<int>(x), static_cast<int>(y), static_cast<int>(z), static_cast<int>(w)};
}

} // namespace gsm

#endif // !GSM_IVECTOR_CUH
