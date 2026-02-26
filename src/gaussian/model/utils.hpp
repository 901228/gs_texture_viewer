#ifndef GAUSSIAN_UTILS_HPP
#define GAUSSIAN_UTILS_HPP
#pragma once

#include <functional>

#include <glm/glm.hpp>

#include <cuda_runtime.h>

#include "../utils/logger.hpp"

#define CUDA_SAFE_CALL_ALWAYS(A)                                                                             \
  A;                                                                                                         \
  cudaDeviceSynchronize();                                                                                   \
  if (cudaPeekAtLastError() != cudaSuccess)                                                                  \
    ERROR(cudaGetErrorString(cudaGetLastError()));

#if DEBUG || _DEBUG
#define CUDA_SAFE_CALL(A) CUDA_SAFE_CALL_ALWAYS(A)
#else
#define CUDA_SAFE_CALL(A) A
#endif

namespace Utils {

inline std::function<char *(size_t N)> resizeFunctional(void **ptr, size_t &S) {
  auto lambda = [ptr, &S](size_t N) {
    if (N > S) {
      if (*ptr)
        CUDA_SAFE_CALL(cudaFree(*ptr));
      CUDA_SAFE_CALL(cudaMalloc(ptr, 2 * N));
      S = 2 * N;
    }
    return reinterpret_cast<char *>(*ptr);
  };
  return lambda;
}

inline void flipRow(glm::mat4 &mat, int row) {
  for (int c = 0; c < 4; ++c)
    mat[c][row] *= -1.0f;
}

inline glm::vec3 getModelCenter(glm::vec3 boxmin, glm::vec3 boxmax) {
  return {(boxmin.x + boxmax.x) / 2.0f, (boxmin.y + boxmax.y) / 2.0f, (boxmin.z + boxmax.z) / 2.0f};
}

} // namespace Utils

#endif // !GAUSSIAN_UTILS_HPP
