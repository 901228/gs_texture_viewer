#ifndef RASTERIZER_RASTERIZER_HPP
#define RASTERIZER_RASTERIZER_HPP
#pragma once

#include <functional>

#include <cuda.h>
#include <cuda_runtime.h>

#include "texture_rasterizer.hpp"

namespace CudaRasterizer {

int forward(const std::function<char *(size_t)> &geometryBuffer,
            const std::function<char *(size_t)> &binningBuffer,
            const std::function<char *(size_t)> &imageBuffer, int P, int D, int M, const float *background,
            int width, int height, const float *means3D, const float *shs, const float *colors_precomp,
            const float *opacities, const float *scales, float scale_modifier, const float *rotations,
            const float *cov3D_precomp, const float *viewmatrix, const float *projmatrix,
            const float *cam_pos, float tan_fovx, float tan_fovy, bool prefiltered, float *out_color,
            bool antialiasing, int *radii = nullptr, int *rects = nullptr, const float *boxmin = nullptr,
            const float *boxmax = nullptr, const CudaRasterizer::PixelMask *mask = nullptr,
            float threshold = 0.005f, TextureOption textureOption = {});

} // namespace CudaRasterizer

#endif // !RASTERIZER_RASTERIZER_HPP
