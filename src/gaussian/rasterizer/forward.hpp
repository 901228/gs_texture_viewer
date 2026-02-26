#ifndef RASTERIZER_FORWARD_HPP
#define RASTERIZER_FORWARD_HPP
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "vector/vector.hpp"
namespace rs = rasterizer;

#include "texture_rasterizer.hpp"

namespace FORWARD {

// Perform initial steps for each Gaussian prior to rasterization.
void preprocess(int P, int D, int M, const float *means3D, const rs::vec3 *scales, float scale_modifier,
                const rs::vec4 *rotations, const float *opacities, const float *shs, bool *clamped,
                const float *cov3D_precomp, const float *colors_precomp, const float *viewmatrix,
                const float *projmatrix, const rs::vec3 *cam_pos, int W, int H, float focal_x, float focal_y,
                float tan_fovx, float tan_fovy, int *radii, float2 *means2D, float *depths, float *cov3Ds,
                float *rgb, float4 *conic_opacity, dim3 grid, uint32_t *tiles_touched, bool prefiltered,
                int2 *rects, float3 boxmin, float3 boxmax, bool antialiasing);

// Main rasterization method.
void render(dim3 grid, dim3 block, const uint2 *ranges, const uint32_t *point_list, int W, int H,
            const float2 *means2D, const float *depths, const float *colors, const float4 *conic_opacity,
            float *final_T, uint32_t *n_contrib, const float *bg_color, float *out_color,
            const CudaRasterizer::PixelMask *mask = nullptr, float threshold = 0.005f,
            CudaRasterizer::TextureOption textureOption = {});
} // namespace FORWARD

#endif // !RASTERIZER_FORWARD_HPP
