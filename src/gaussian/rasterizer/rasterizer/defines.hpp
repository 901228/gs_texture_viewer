#ifndef RASTERIZER_DEFINES_HPP
#define RASTERIZER_DEFINES_HPP
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_types.h>

namespace CudaRasterizer {

enum class RenderingMode : int { Color, TextureCoords, Depth, Normal };

enum class MaskCullingMode : int {
  None = 0,                                              // 0b000 = 0
  T_Comparison = 1 << 0,                                 // 0b001 = 1
  NormalCulling = 1 << 1,                                // 0b010 = 2
  Depth_Comparison = 1 << 2,                             // 0b100 = 4
  Mixed_TN = T_Comparison | NormalCulling,               // 0b011 = 3
  Mixed_TD = T_Comparison | Depth_Comparison,            // 0b101 = 5
  Mixed_ND = NormalCulling | Depth_Comparison,           // 0b110 = 6
  All = T_Comparison | NormalCulling | Depth_Comparison, // 0b111 = 7
};

__device__ __host__ inline MaskCullingMode operator|(MaskCullingMode a, MaskCullingMode b) {
  return static_cast<MaskCullingMode>(static_cast<int>(a) | static_cast<int>(b));
}
__device__ __host__ inline bool operator&(MaskCullingMode a, MaskCullingMode b) {
  return static_cast<int>(a) & static_cast<int>(b);
}

struct TextureOption {

  float scale = 1;
  float2 offset = {};
  float theta = 0;
  MaskCullingMode cullingMode = MaskCullingMode::Mixed_ND;
};

struct PixelMask {
  uint8_t mask;     // whether to replace the pixel with texture
  float3 color;     // sample color of each pixel
  float depth;      // depth of each pixel
  float2 texCoords; // texture coordinates of each pixel
  float3 normal;    // normal of each pixel
};

struct Light {
  float3 direction;
  float3 color;
  float intensity;
};

} // namespace CudaRasterizer

#endif // !RASTERIZER_DEFINES_HPP
