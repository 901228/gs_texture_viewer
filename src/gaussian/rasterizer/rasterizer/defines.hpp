#ifndef RASTERIZER_DEFINES_HPP
#define RASTERIZER_DEFINES_HPP
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_types.h>

namespace CudaRasterizer {

enum class RenderingMode : int { Color, TextureCoords, Depth, Normal };

enum class MaskCullingMode : int { None, DepthComparison, NormalCulling };

struct TextureOption {

  float scale = 1;
  float2 offset = {};
  float theta = 0;
  MaskCullingMode cullingMode = MaskCullingMode::DepthComparison;
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
