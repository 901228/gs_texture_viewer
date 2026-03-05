#ifndef RASTERIZER_DEFINES_HPP
#define RASTERIZER_DEFINES_HPP
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace CudaRasterizer {

enum class RenderingMode : int { Color, Depth };

enum class MaskCullingMode : int { DepthComparison, NormalCulling };

struct TextureOption {

  enum class RenderingMode : int { None, TextureCoords, Texture };

  float scale = 1;
  float2 offset = {};
  float theta = 0;
  RenderingMode mode = RenderingMode::None;
  MaskCullingMode cullingMode = MaskCullingMode::DepthComparison;
};

struct PixelMask {
  uint8_t mask;              // whether to replace the pixel with texture
  cudaTextureObject_t texId; // texture ID of each pixel
  float2 texCoords;          // texture coordinates of each pixel
  float depth;               // depth of each pixel
};

} // namespace CudaRasterizer

#endif // !RASTERIZER_DEFINES_HPP
