#include "forward.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdint>
#include <texture_types.h>

#include "auxiliary.h"
#include "rasterizer/defines.hpp"

#include "gsm/gsm.cuh"

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ gsm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const gsm::vec3 *means,
                                        gsm::vec3 campos, const float *shs, bool *clamped) {
  // The implementation is loosely based on code for
  // "Differentiable Point-Based Radiance Fields for
  // Efficient View Synthesis" by Zhang et al. (2022)
  gsm::vec3 dir = means[idx] - campos;
  dir.normalize();

  gsm::vec3 *sh = ((gsm::vec3 *)shs) + idx * max_coeffs;
  gsm::vec3 result = SH_C0 * gsm::vec3(sh[0]);

  if (deg > 0) {
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

    if (deg > 1) {
      float xx = x * x, yy = y * y, zz = z * z;
      float xy = x * y, yz = y * z, xz = x * z;
      result = result + SH_C2[0] * xy * sh[4] + SH_C2[1] * yz * sh[5] +
               SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] + SH_C2[3] * xz * sh[7] +
               SH_C2[4] * (xx - yy) * sh[8];

      if (deg > 2) {
        result = result + SH_C3[0] * y * (3.0f * xx - yy) * sh[9] + SH_C3[1] * xy * z * sh[10] +
                 SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                 SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                 SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] + SH_C3[5] * z * (xx - yy) * sh[14] +
                 SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
      }
    }
  }
  result += 0.5f;

  // RGB colors are clamped to positive values. If values are
  // clamped, we need to keep track of this for the backward pass.
  clamped[3 * idx + 0] = (result.x < 0);
  clamped[3 * idx + 1] = (result.y < 0);
  clamped[3 * idx + 2] = (result.z < 0);
  return gsm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(gsm::vec3 &p_view, float focal_x, float focal_y, float tan_fovx,
                               float tan_fovy, const float *cov3D, const gsm::mat4 &viewmatrix) {
  // The following models the steps outlined by equations 29
  // and 31 in "EWA Splatting" (Zwicker et al., 2002).
  // Additionally considers aspect / scaling of viewport.
  // Transposes used to account for row-/column-major conventions.
  const float limx = 1.3f * tan_fovx;
  const float limy = 1.3f * tan_fovy;
  const float txtz = p_view.x / p_view.z;
  const float tytz = p_view.y / p_view.z;
  p_view.x = gsm::min(limx, gsm::max(-limx, txtz)) * p_view.z;
  p_view.y = gsm::min(limy, gsm::max(-limy, tytz)) * p_view.z;

  gsm::mat3 J = gsm::mat3(focal_x / p_view.z, 0.0f, -(focal_x * p_view.x) / (p_view.z * p_view.z), 0.0f,
                          focal_y / p_view.z, -(focal_y * p_view.y) / (p_view.z * p_view.z), 0, 0, 0);

  gsm::mat3 W =
      gsm::mat3(viewmatrix[0][0], viewmatrix[1][0], viewmatrix[2][0], viewmatrix[0][1], viewmatrix[1][1],
                viewmatrix[2][1], viewmatrix[0][2], viewmatrix[1][2], viewmatrix[2][2]);

  gsm::mat3 T = W * J;

  gsm::mat3 Vrk =
      gsm::mat3(cov3D[0], cov3D[1], cov3D[2], cov3D[1], cov3D[3], cov3D[4], cov3D[2], cov3D[4], cov3D[5]);

  gsm::mat3 cov = T.transpose() * Vrk.transpose() * T;

  return {float(cov[0][0]), float(cov[0][1]), float(cov[1][1])};
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const gsm::vec3 scale, float mod, const gsm::vec4 rot, float *cov3D) {
  // Create scaling matrix
  gsm::mat3 S{1.0f};
  S[0][0] = mod * scale.x;
  S[1][1] = mod * scale.y;
  S[2][2] = mod * scale.z;

  // Normalize quaternion to get valid rotation
  gsm::vec4 q = rot; // / gsm::length(rot);
  float r = q.x;
  float x = q.y;
  float y = q.z;
  float z = q.w;

  // Compute rotation matrix from quaternion
  gsm::mat3 R = gsm::mat3(1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
                          2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
                          2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

  gsm::mat3 M = S * R;

  // Compute 3D world covariance matrix Sigma
  gsm::mat3 Sigma = M.transpose() * M;

  // Covariance is symmetric, only store upper right
  cov3D[0] = Sigma[0][0];
  cov3D[1] = Sigma[0][1];
  cov3D[2] = Sigma[0][2];
  cov3D[3] = Sigma[1][1];
  cov3D[4] = Sigma[1][2];
  cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template <int C>
__global__ void
preprocessCUDA(int P, int D, int M, const float *orig_points, const gsm::vec3 *scales,
               const float scale_modifier, const gsm::vec4 *rotations, const float *opacities,
               const float *shs, bool *clamped, const float *cov3D_precomp, const float *colors_precomp,
               const float *viewmatrix, const float *projviewmatrix, const gsm::vec3 *cam_pos, const int W,
               int H, const float tan_fovx, float tan_fovy, const float focal_x, float focal_y, int *radii,
               float2 *points_xy_image, float *depths, float *cov3Ds, float *rgb, float4 *conic_opacity,
               const dim3 grid, uint32_t *tiles_touched, bool prefiltered, int2 *rects, float3 boxmin,
               float3 boxmax, bool antialiasing) {
  auto idx = cooperative_groups::__v1::grid_group::thread_rank();
  if (idx >= P)
    return;

  gsm::mat4 view_mat{viewmatrix};
  gsm::mat4 proj_view_mat{projviewmatrix};

  // Initialize radius and touched tiles to 0. If this isn't changed,
  // this Gaussian will not be processed further.
  radii[idx] = 0;
  tiles_touched[idx] = 0;

  // Transform point by projecting
  gsm::vec3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]};

  if (p_orig.x < boxmin.x || p_orig.y < boxmin.y || p_orig.z < boxmin.z || p_orig.x > boxmax.x ||
      p_orig.y > boxmax.y || p_orig.z > boxmax.z)
    return;

  gsm::vec4 p_hom = proj_view_mat * gsm::vec4(p_orig, 1.0f);
  float p_w = 1.0f / (p_hom.w + 0.0000001f);
  float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};
  gsm::vec3 p_view = gsm::vec3(view_mat * gsm::vec4(p_orig, 1.0f));

  // Perform near culling, quit if outside.
  if (!in_frustum(prefiltered, p_view))
    return;

  // If 3D covariance matrix is precomputed, use it, otherwise compute from scaling and rotation parameters.
  const float *cov3D;
  if (cov3D_precomp != nullptr) {
    cov3D = cov3D_precomp + idx * 6;
  } else {
    computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
    cov3D = cov3Ds + idx * 6;
  }

  // Compute 2D screen-space covariance matrix
  float3 cov = computeCov2D(p_view, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, view_mat);

  constexpr float h_var = 0.3f;
  const float det_cov = cov.x * cov.z - cov.y * cov.y;
  cov.x += h_var;
  cov.z += h_var;
  const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
  float h_convolution_scaling = 1.0f;

  if (antialiasing)
    h_convolution_scaling = sqrt(max(0.000025f,
                                     det_cov / det_cov_plus_h_cov)); // max for numerical stability

  // Invert covariance (EWA algorithm)
  const float det = det_cov_plus_h_cov;

  if (det == 0.0f)
    return;
  float det_inv = 1.f / det;
  float3 conic = {cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv};

  // Compute extent in screen space (by finding eigenvalues of
  // 2D covariance matrix). Use extent to compute a bounding rectangle
  // of screen-space tiles that this Gaussian overlaps with. Quit if
  // rectangle covers 0 tiles.

  float mid = 0.5f * (cov.x + cov.z);
  float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
  float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
  float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
  float2 point_image = {ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H)};
  uint2 rect_min, rect_max;

  if (rects == nullptr) // More conservative
  {
    getRect(point_image, static_cast<int>(my_radius), rect_min, rect_max, grid);
  } else // Slightly more aggressive, might need a math cleanup
  {
    const int2 my_rect = {(int)ceil(3.f * sqrt(cov.x)), (int)ceil(3.f * sqrt(cov.z))};
    rects[idx] = my_rect;
    getRect(point_image, my_rect, rect_min, rect_max, grid);
  }

  if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
    return;

  // If colors have been precomputed, use them, otherwise convert
  // spherical harmonics coefficients to RGB color.
  if (colors_precomp == nullptr) {
    gsm::vec3 result =
        computeColorFromSH(static_cast<int>(idx), D, M, (gsm::vec3 *)orig_points, *cam_pos, shs, clamped);
    rgb[idx * C + 0] = result.x;
    rgb[idx * C + 1] = result.y;
    rgb[idx * C + 2] = result.z;
  }

  // Store some useful helper data for the next steps.
  depths[idx] = p_view.z;
  radii[idx] = static_cast<int>(my_radius);
  points_xy_image[idx] = point_image;
  // Inverse 2D covariance and opacity neatly pack into one float4
  float opacity = opacities[idx];

  conic_opacity[idx] = {conic.x, conic.y, conic.z, opacity * h_convolution_scaling};

  tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

void FORWARD::preprocess(int P, int D, int M, const float *means3D, const gsm::vec3 *scales,
                         float scale_modifier, const gsm::vec4 *rotations, const float *opacities,
                         const float *shs, bool *clamped, const float *cov3D_precomp,
                         const float *colors_precomp, const float *viewmatrix, const float *projviewmatrix,
                         const gsm::vec3 *cam_pos, int W, int H, float focal_x, float focal_y, float tan_fovx,
                         float tan_fovy, int *radii, float2 *means2D, float *depths, float *cov3Ds,
                         float *rgb, float4 *conic_opacity, dim3 grid, uint32_t *tiles_touched,
                         bool prefiltered, int2 *rects, float3 boxmin, float3 boxmax, bool antialiasing) {

  preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
      P, D, M, means3D, scales, scale_modifier, rotations, opacities, shs, clamped, cov3D_precomp,
      colors_precomp, viewmatrix, projviewmatrix, cam_pos, W, H, tan_fovx, tan_fovy, focal_x, focal_y, radii,
      means2D, depths, cov3Ds, rgb, conic_opacity, grid, tiles_touched, prefiltered, rects, boxmin, boxmax,
      antialiasing);
}

// tested:
//   - armadillo: ~50
__device__ static uint32_t depthMax = DEPTH_MIN;

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
    renderCUDA(const uint2 *__restrict__ ranges, const uint32_t *__restrict__ point_list, int W, int H,
               const float2 *__restrict__ points_xy_image, const float *__restrict__ depths,
               const float *__restrict__ features, const float4 *__restrict__ conic_opacity,
               float *__restrict__ final_T, uint32_t *__restrict__ n_contrib,
               const float *__restrict__ bg_color, float *__restrict__ out_color,
               float *__restrict__ out_depth_raw = nullptr, float *__restrict__ out_t_final = nullptr,
               int P0 = -1, const unsigned int *__restrict__ appearance_face_idx = nullptr,
               const unsigned int *__restrict__ selectedID = nullptr, int selectedIDSize = 0,
               CudaRasterizer::RenderingMode renderingMode = CudaRasterizer::RenderingMode::Color,
               const CudaRasterizer::PixelMask *__restrict__ mask = nullptr, float threshold1 = 0.005f,
               float threshold2 = 0.005f, float threshold3 = 0.005f, float threshold4 = 0.005f,
               CudaRasterizer::TextureOption textureOption = {}) {
  // Identify current tile and associated min/max pixel range.
  uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
  uint2 pix_min = {cooperative_groups::__v1::thread_block::group_index().x * BLOCK_X,
                   cooperative_groups::__v1::thread_block::group_index().y * BLOCK_Y};
  uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
  uint2 pix = {pix_min.x + cooperative_groups::__v1::thread_block::thread_index().x,
               pix_min.y + cooperative_groups::__v1::thread_block::thread_index().y};
  uint32_t pix_id = W * pix.y + pix.x;
  float2 pixf = {(float)pix.x, (float)pix.y};

  // Check if this thread is associated with a valid pixel or outside.
  bool inside = pix.x < W && pix.y < H;
  // Done threads can help with fetching, but don't rasterize
  bool done = !inside;

  // Load start/end range of IDs to process in bit sorted list.
  uint2 range = ranges[cooperative_groups::__v1::thread_block::group_index().y * horizontal_blocks +
                       cooperative_groups::__v1::thread_block::group_index().x];
  const int rounds = ((static_cast<int>(range.y) - static_cast<int>(range.x) + BLOCK_SIZE - 1) / BLOCK_SIZE);
  int toDo = static_cast<int>(range.y) - static_cast<int>(range.x);

  // Allocate storage for batches of collectively fetched data.
  __shared__ int collected_id[BLOCK_SIZE];
  __shared__ float2 collected_xy[BLOCK_SIZE];
  __shared__ float4 collected_conic_opacity[BLOCK_SIZE];

  // Initialize helper variables
  float T = 1.0f;
  uint32_t contributor = 0;
  uint32_t last_contributor = 0;
  float C[CHANNELS] = {0};

  bool hasMask = mask != nullptr && mask[pix_id].mask != 0;
  float mesh_depth = hasMask ? mask[pix_id].depth : FLT_MAX;
  bool compareT = (textureOption.cullingMode & CudaRasterizer::MaskCullingMode::T_Comparison);
  bool compareF = (textureOption.cullingMode & CudaRasterizer::MaskCullingMode::FaceIdx);
  bool found = !hasMask || (!compareT && !compareF);
  bool visible = false;
  float pixelDepth = 0;

  // Iterate over batches until all done or range is complete
  for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
    // End if entire block votes that it is done rasterizing
    int num_done = __syncthreads_count(done);
    if (num_done == BLOCK_SIZE)
      break;

    // Collectively fetch per-Gaussian data from global to shared
    int progress = i * BLOCK_SIZE + static_cast<int>(cooperative_groups::__v1::thread_block::thread_rank());
    if (range.x + progress < range.y) {
      int coll_id = static_cast<int>(point_list[range.x + progress]);
      collected_id[cooperative_groups::__v1::thread_block::thread_rank()] = coll_id;
      collected_xy[cooperative_groups::__v1::thread_block::thread_rank()] = points_xy_image[coll_id];
      collected_conic_opacity[cooperative_groups::__v1::thread_block::thread_rank()] = conic_opacity[coll_id];
    }
    cooperative_groups::__v1::thread_block::sync();

    // Iterate over current batch
    for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
      // Keep track of current position in range
      contributor++;

      if (!found) {
        bool selected = false;
        int id = collected_id[j];
        bool isAppearance = P0 >= 0 && id >= P0;
        if (compareF && isAppearance) {
          unsigned int face_idx = appearance_face_idx[id - P0];
          for (int k = 0; k < selectedIDSize; k++) {
            if (selectedID[k] == face_idx) {
              selected = true;
              break;
            }
          }
        }

        if (compareF && compareT) {
          if (depths[id] > mesh_depth) {
            found = true;

            // T is the rest available contribution for this gs
            visible = T >= threshold1;
          }
          // else if (std::abs(depths[id] - mesh_depth) < threshold3 && selected) {
          else if (depths[id] < mesh_depth && selected) {
            found = true;
            visible = T >= threshold2;
          }
        }
        // else if (compareF && std::abs(depths[id] - mesh_depth) < threshold3 && selected) {
        else if (compareF && depths[id] < mesh_depth && selected) {
          found = true;
          visible = T >= threshold2;
        } else if (compareT && depths[id] > mesh_depth) {
          found = true;

          // T is the rest available contribution for this gs
          visible = T >= threshold1;
        }
      }

      // Resample using conic matrix (cf. "Surface Splatting" by Zwicker et al., 2001)
      float2 xy = collected_xy[j];
      float2 d = {xy.x - pixf.x, xy.y - pixf.y};
      float4 con_o = collected_conic_opacity[j];
      float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
      if (power > 0.0f)
        continue;

      // Eq. (2) from 3D Gaussian splatting paper.
      // Obtain alpha by multiplying with Gaussian opacity and its exponential falloff from mean.
      // Avoid numerical instabilities (see paper appendix).
      float alpha = min(0.99f, con_o.w * exp(power));
      if (alpha < 1.0f / 255.0f)
        continue;
      float test_T = T * (1 - alpha);
      if (test_T < 0.0001f) {
        done = true;
        continue;
      }

      // Eq. (3) from 3D Gaussian splatting paper.
      float _contrib = alpha * T;
      for (int ch = 0; ch < CHANNELS; ch++)
        C[ch] += features[collected_id[j] * CHANNELS + ch] * _contrib;
      pixelDepth += depths[collected_id[j]] * _contrib;

      T = test_T;

      // Keep track of last range entry to update this pixel.
      last_contributor = contributor;
    }
  }
  uint32_t depth_uint = __float_as_uint(pixelDepth);
  atomicMax(&depthMax, depth_uint);

  // All threads that treat valid pixel write out their final
  // rendering data to the frame and auxiliary buffers.
  if (inside) {
    final_T[pix_id] = T;
    n_contrib[pix_id] = last_contributor;

    if (out_depth_raw != nullptr)
      out_depth_raw[pix_id] = pixelDepth;
    if (out_t_final != nullptr)
      out_t_final[pix_id] = T;

    float dMax = __uint_as_float(depthMax);
    float dRange = dMax - DEPTH_MIN;

    pixelDepth += T * dMax; // background compensation

    if (renderingMode != CudaRasterizer::RenderingMode::Depth) {
      for (int ch = 0; ch < CHANNELS; ch++)
        out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
    } else {
      for (int ch = 0; ch < CHANNELS; ch++)
        out_color[ch * H * W + pix_id] = (dMax - pixelDepth) / dRange;
    }

    // whether to override the GS color with the texture color
    bool overrideColor = (compareT && visible) || (compareF && visible) ||
                         (textureOption.cullingMode == CudaRasterizer::MaskCullingMode::NormalCulling) ||
                         ((textureOption.cullingMode & CudaRasterizer::MaskCullingMode::Depth_Comparison) &&
                          std::abs(mesh_depth - pixelDepth) <= threshold3 * dRange) ||
                         (textureOption.cullingMode == CudaRasterizer::MaskCullingMode::None);

    if (hasMask && overrideColor) {
      const CudaRasterizer::PixelMask &m = mask[pix_id];

      if (renderingMode == CudaRasterizer::RenderingMode::Color) {
        out_color[0 * H * W + pix_id] = m.color.x;
        out_color[1 * H * W + pix_id] = m.color.y;
        out_color[2 * H * W + pix_id] = m.color.z;
      } else if (renderingMode == CudaRasterizer::RenderingMode::TextureCoords) {
        out_color[0 * H * W + pix_id] = m.texCoords.x;
        out_color[1 * H * W + pix_id] = m.texCoords.y;
        out_color[2 * H * W + pix_id] = 0.0f;
      } else if (renderingMode == CudaRasterizer::RenderingMode::Normal) {
        out_color[0 * H * W + pix_id] = m.normal.x;
        out_color[1 * H * W + pix_id] = m.normal.y;
        out_color[2 * H * W + pix_id] = m.normal.z;
      } else if (renderingMode == CudaRasterizer::RenderingMode::Depth) {
        float dd = (dMax - m.depth) / dRange;
        out_color[0 * H * W + pix_id] = dd;
        out_color[1 * H * W + pix_id] = dd;
        out_color[2 * H * W + pix_id] = dd;
      } else {
        out_color[0 * H * W + pix_id] = 1.0f;
        out_color[1 * H * W + pix_id] = 0.0f;
        out_color[2 * H * W + pix_id] = 0.0f;
      }
    }

    // if (hasMask) {
    //   const CudaRasterizer::PixelMask &m = mask[pix_id];

    //   out_color[0 * H * W + pix_id] = m.normal.x;
    //   out_color[1 * H * W + pix_id] = m.normal.y;
    //   out_color[2 * H * W + pix_id] = m.normal.z;
    // }
  }
}

void FORWARD::render(dim3 grid, dim3 block, const uint2 *ranges, const uint32_t *point_list, int W, int H,
                     const float2 *means2D, const float *depths, const float *colors,
                     const float4 *conic_opacity, float *final_T, uint32_t *n_contrib, const float *bg_color,
                     float *out_color, float *out_depth_raw, float *out_t_final, int P0,
                     const unsigned int *appearance_face_idx, const unsigned int *selectedID,
                     int selectedIDSize, CudaRasterizer::RenderingMode renderingMode,
                     const CudaRasterizer::PixelMask *mask, float threshold1, float threshold2,
                     float threshold3, float threshold4, CudaRasterizer::TextureOption textureOption) {

  renderCUDA<NUM_CHANNELS><<<grid, block>>>(
      ranges, point_list, W, H, means2D, depths, colors, conic_opacity, final_T, n_contrib, bg_color,
      out_color, out_depth_raw, out_t_final, P0, appearance_face_idx, selectedID, selectedIDSize,
      renderingMode, mask, threshold1, threshold2, threshold3, threshold4, textureOption);
}
