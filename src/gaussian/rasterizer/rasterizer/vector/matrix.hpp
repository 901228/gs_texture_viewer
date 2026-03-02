
#ifndef RASTERIZER_MATRIX_HPP
#define RASTERIZER_MATRIX_HPP
#pragma once

#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include "vector.hpp"

namespace rasterizer {
struct mat2 {
  vec2 cols[2]; // column-major

  // Constructors
  __host__ __device__ mat2() : cols{{1, 0}, {0, 1}} {}
  __host__ __device__ explicit mat2(float val) : cols{{val, 0}, {0, val}} {}
  __host__ __device__ mat2(const vec2 &c0, const vec2 &c1) : cols{c0, c1} {}
  __host__ __device__ mat2(const float &x0, const float &y0, const float &x1, const float &y1)
      : cols{{x0, y0}, {x1, y1}} {}
  __host__ __device__ explicit mat2(const float *vals) : cols{{vals[0], vals[1]}, {vals[2], vals[3]}} {}

  // Access
  __host__ __device__ vec2 &operator[](int i) {
    assert(i >= 0 && i < 2);
    return cols[i];
  }

  __host__ __device__ const vec2 &operator[](int i) const {
    assert(i >= 0 && i < 2);
    return cols[i];
  }

  // Addition
  __host__ __device__ mat2 operator+(const mat2 &m) const { return {cols[0] + m[0], cols[1] + m[1]}; }

  __host__ __device__ mat2 &operator+=(const mat2 &m) {
    cols[0] += m[0];
    cols[1] += m[1];
    return *this;
  }

  // Subtraction
  __host__ __device__ mat2 operator-(const mat2 &m) const { return {cols[0] - m[0], cols[1] - m[1]}; }

  __host__ __device__ mat2 &operator-=(const mat2 &m) {
    cols[0] -= m[0];
    cols[1] -= m[1];
    return *this;
  }

  // Scalar multiply
  __host__ __device__ mat2 operator*(float val) const { return {cols[0] * val, cols[1] * val}; }

  __host__ __device__ mat2 &operator*=(float val) {
    cols[0] *= val;
    cols[1] *= val;
    return *this;
  }

  // Matrix * vec
  __host__ __device__ vec2 operator*(const vec2 &v) const { return cols[0] * v.x + cols[1] * v.y; }

  // Matrix * matrix
  __host__ __device__ mat2 operator*(const mat2 &m) const {
    return {
        (*this) * m[0],
        (*this) * m[1],
    };
  }

  // Transpose
  __host__ __device__ mat2 transpose() const {
    return mat2(vec2(cols[0].x, cols[1].x), vec2(cols[0].y, cols[1].y));
  }

  // Identity helper
  __host__ __device__ static mat2 identity() { return mat2(); }
};

// scalar * mat
__host__ __device__ inline mat2 operator*(float val, const mat2 &m) { return m * val; }

__host__ __device__ inline mat2 transpose(const mat2 &m) { return m.transpose(); }

struct mat3 {
  vec3 cols[3]; // column-major

  // Constructors
  __host__ __device__ mat3() : cols{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}} {}
  __host__ __device__ explicit mat3(float val) : cols{{val, 0, 0}, {0, val, 0}, {0, 0, val}} {}
  __host__ __device__ mat3(const vec3 &c0, const vec3 &c1, const vec3 &c2) : cols{c0, c1, c2} {}
  __host__ __device__ mat3(const float &x0, const float &y0, const float &z0, const float &x1,
                           const float &y1, const float &z1, const float &x2, const float &y2,
                           const float &z2)
      : cols{{x0, y0, z0}, {x1, y1, z1}, {x2, y2, z2}} {}
  __host__ __device__ explicit mat3(const float *vals)
      : cols{{vals[0], vals[1], vals[2]}, {vals[3], vals[4], vals[5]}, {vals[6], vals[7], vals[8]}} {}

  // Access
  __host__ __device__ vec3 &operator[](int i) {
    assert(i >= 0 && i < 3);
    return cols[i];
  }

  __host__ __device__ const vec3 &operator[](int i) const {
    assert(i >= 0 && i < 3);
    return cols[i];
  }

  // Addition
  __host__ __device__ mat3 operator+(const mat3 &m) const {
    return {cols[0] + m[0], cols[1] + m[1], cols[2] + m[2]};
  }

  __host__ __device__ mat3 &operator+=(const mat3 &m) {
    cols[0] += m[0];
    cols[1] += m[1];
    cols[2] += m[2];
    return *this;
  }

  // Subtraction
  __host__ __device__ mat3 operator-(const mat3 &m) const {
    return {cols[0] - m[0], cols[1] - m[1], cols[2] - m[2]};
  }

  __host__ __device__ mat3 &operator-=(const mat3 &m) {
    cols[0] -= m[0];
    cols[1] -= m[1];
    cols[2] -= m[2];
    return *this;
  }

  // Scalar multiply
  __host__ __device__ mat3 operator*(float val) const {
    return {cols[0] * val, cols[1] * val, cols[2] * val};
  }

  __host__ __device__ mat3 &operator*=(float val) {
    cols[0] *= val;
    cols[1] *= val;
    cols[2] *= val;
    return *this;
  }

  // Matrix * vec
  __host__ __device__ vec3 operator*(const vec3 &v) const {
    return cols[0] * v.x + cols[1] * v.y + cols[2] * v.z;
  }

  // Matrix * matrix
  __host__ __device__ mat3 operator*(const mat3 &m) const {
    return {
        (*this) * m[0],
        (*this) * m[1],
        (*this) * m[2],
    };
  }

  // Transpose
  __host__ __device__ mat3 transpose() const {
    return mat3(vec3(cols[0].x, cols[1].x, cols[2].x), vec3(cols[0].y, cols[1].y, cols[2].y),
                vec3(cols[0].z, cols[1].z, cols[2].z));
  }

  // Identity helper
  __host__ __device__ static mat3 identity() { return mat3(); }
};

// scalar * mat
__host__ __device__ inline mat3 operator*(float val, const mat3 &m) { return m * val; }

__host__ __device__ inline mat3 transpose(const mat3 &m) { return m.transpose(); }

struct mat4 {
  vec4 cols[4]; // column-major

  // Constructors
  __host__ __device__ mat4() : cols{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}} {}
  __host__ __device__ explicit mat4(float val)
      : cols{{val, 0, 0, 0}, {0, val, 0, 0}, {0, 0, val, 0}, {0, 0, 0, val}} {}
  __host__ __device__ mat4(const vec4 &c0, const vec4 &c1, const vec4 &c2, const vec4 &c3)
      : cols{c0, c1, c2, c3} {}
  __host__ __device__ mat4(const float &x0, const float &y0, const float &z0, const float &w0,
                           const float &x1, const float &y1, const float &z1, const float &w1,
                           const float &x2, const float &y2, const float &z2, const float &w2,
                           const float &x3, const float &y3, const float &z3, const float &w3)
      : cols{{x0, y0, z0, w0}, {x1, y1, z1, w1}, {x2, y2, z2, w2}, {x3, y3, z3, w3}} {}
  __host__ __device__ explicit mat4(const float *vals)
      : cols{{vals[0], vals[1], vals[2], vals[3]},
             {vals[4], vals[5], vals[6], vals[7]},
             {vals[8], vals[9], vals[10], vals[11]},
             {vals[12], vals[13], vals[14], vals[15]}} {}

  // Access
  __host__ __device__ vec4 &operator[](int i) {
    assert(i >= 0 && i < 4);
    return cols[i];
  }

  __host__ __device__ const vec4 &operator[](int i) const {
    assert(i >= 0 && i < 4);
    return cols[i];
  }

  // Addition
  __host__ __device__ mat4 operator+(const mat4 &m) const {
    return {cols[0] + m[0], cols[1] + m[1], cols[2] + m[2], cols[3] + m[3]};
  }

  __host__ __device__ mat4 &operator+=(const mat4 &m) {
    cols[0] += m[0];
    cols[1] += m[1];
    cols[2] += m[2];
    cols[3] += m[3];
    return *this;
  }

  // Subtraction
  __host__ __device__ mat4 operator-(const mat4 &m) const {
    return {cols[0] - m[0], cols[1] - m[1], cols[2] - m[2], cols[3] - m[3]};
  }

  __host__ __device__ mat4 &operator-=(const mat4 &m) {
    cols[0] -= m[0];
    cols[1] -= m[1];
    cols[2] -= m[2];
    cols[3] -= m[3];
    return *this;
  }

  // Scalar multiply
  __host__ __device__ mat4 operator*(float val) const {
    return {cols[0] * val, cols[1] * val, cols[2] * val, cols[3] * val};
  }

  __host__ __device__ mat4 &operator*=(float val) {
    cols[0] *= val;
    cols[1] *= val;
    cols[2] *= val;
    cols[3] *= val;
    return *this;
  }

  // Matrix * vec
  __host__ __device__ vec4 operator*(const vec4 &v) const {
    return cols[0] * v.x + cols[1] * v.y + cols[2] * v.z + cols[3] * v.w;
  }

  // Matrix * matrix
  __host__ __device__ mat4 operator*(const mat4 &m) const {
    return {
        (*this) * m[0],
        (*this) * m[1],
        (*this) * m[2],
        (*this) * m[3],
    };
  }

  // Transpose
  __host__ __device__ [[nodiscard]] mat4 transpose() const {
    return {
        vec4(cols[0].x, cols[1].x, cols[2].x, cols[3].x), vec4(cols[0].y, cols[1].y, cols[2].y, cols[3].y),
        vec4(cols[0].z, cols[1].z, cols[2].z, cols[3].z), vec4(cols[0].w, cols[1].w, cols[2].w, cols[3].w)};
  }

  // Identity helper
  __host__ __device__ static mat4 identity() { return {}; }
};

// scalar * mat
__host__ __device__ inline mat4 operator*(float val, const mat4 &m) { return m * val; }

__host__ __device__ inline mat4 transpose(const mat4 &m) { return m.transpose(); }

} // namespace rasterizer

#endif // !RASTERIZER_MATRIX_HPP
