#ifndef RASTERIZER_TEXTURE_RASTERIZER_HPP
#define RASTERIZER_TEXTURE_RASTERIZER_HPP
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace CudaRasterizer {

enum class RenderingMode : int { None, TextureCoords, Texture };

struct TextureOption {
  float scale = 1;
  float2 offset = {};
  float theta = 0;
  RenderingMode mode = RenderingMode::Texture;
};

struct PixelMask {
  uint8_t mask;              // whether to replace the pixel with texture
  cudaTextureObject_t texId; // texture ID of each pixel
  float2 texCoords;          // texture coordinates of each pixel
  float depth;               // depth of each pixel
};

void makeMask(const float *position, const float *texCoords, int num_vertices, const cudaTextureObject_t *sl,
              int num_triangles, const uint8_t *face_mask, int width, int height, const float *viewmatrix,
              const float *projmatrix, PixelMask *mask);

} // namespace CudaRasterizer

#endif // !RASTERIZER_TEXTURE_RASTERIZER_HPP
