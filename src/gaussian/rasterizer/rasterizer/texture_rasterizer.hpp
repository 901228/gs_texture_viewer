#ifndef RASTERIZER_TEXTURE_RASTERIZER_HPP
#define RASTERIZER_TEXTURE_RASTERIZER_HPP
#pragma once

#include "defines.hpp"

namespace CudaRasterizer {

void makeMask(const float *position, const float *texCoords, int num_vertices, const cudaTextureObject_t *sl,
              int num_triangles, int width, int height, const float *projviewmatrix, const float *viewpos,
              MaskCullingMode maskCullingMode, PixelMask *mask);

} // namespace CudaRasterizer

#endif // !RASTERIZER_TEXTURE_RASTERIZER_HPP
