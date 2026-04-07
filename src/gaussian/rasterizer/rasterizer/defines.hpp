#ifndef RASTERIZER_DEFINES_HPP
#define RASTERIZER_DEFINES_HPP
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_types.h>

namespace CudaRasterizer {

enum class RenderingMode : int { Color, TextureCoords, Depth, Normal };

enum class MaskCullingMode : int {
  None = 0,                                                               // 0b0000 = 0
  T_Comparison = 1 << 0,                                                  // 0b0001 = 1
  NormalCulling = 1 << 1,                                                 // 0b0010 = 2
  Depth_Comparison = 1 << 2,                                              // 0b0100 = 4
  FaceIdx = 1 << 3,                                                       // 0b1000 = 8
  Mixed_TN = T_Comparison | NormalCulling,                                // 0b0011 = 3
  Mixed_TD = T_Comparison | Depth_Comparison,                             // 0b0101 = 5
  Mixed_ND = NormalCulling | Depth_Comparison,                            // 0b0110 = 6
  Mixed_TND = T_Comparison | NormalCulling | Depth_Comparison,            // 0b0111 = 7
  Mixed_TF = T_Comparison | FaceIdx,                                      // 0b1001 = 9
  Mixed_NF = NormalCulling | FaceIdx,                                     // 0b1010 = 10
  Mixed_TNF = T_Comparison | NormalCulling | FaceIdx,                     // 0b1011 = 11
  Mixed_DF = Depth_Comparison | FaceIdx,                                  // 0b1100 = 12
  Mixed_TDF = T_Comparison | Depth_Comparison | FaceIdx,                  // 0b1101 = 13
  Mixed_NDF = NormalCulling | Depth_Comparison | FaceIdx,                 // 0b1110 = 14
  Mixed_TNDF = T_Comparison | NormalCulling | Depth_Comparison | FaceIdx, // 0b1111 = 15
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
