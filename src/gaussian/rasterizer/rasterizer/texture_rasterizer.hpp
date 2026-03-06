#ifndef RASTERIZER_TEXTURE_RASTERIZER_HPP
#define RASTERIZER_TEXTURE_RASTERIZER_HPP
#pragma once

#include "defines.hpp"

namespace CudaRasterizer {

__device__ float4 sampleTexture(cudaTextureObject_t texId, float2 texCoord, TextureOption textureOption = {});

void makeMask(const float *position, const float *normal, const float *texCoords, const float *tangents,
              const float *bitangents, int num_vertices, const cudaTextureObject_t *basecolorTexId,
              const cudaTextureObject_t *normalTexId, const cudaTextureObject_t *heightTexId,
              TextureOption textureOption, float heightScale, Light lightDirection, int num_triangles,
              int width, int height, const float *projviewmatrix, const float *viewpos,
              MaskCullingMode maskCullingMode, PixelMask *mask);

} // namespace CudaRasterizer

#endif // !RASTERIZER_TEXTURE_RASTERIZER_HPP
