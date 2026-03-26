#include "geodesics.hpp"

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_types.h>

#include "auxiliary.h"
#include "gsm/gsm.cuh"
#include "rasterizer/defines.hpp"
#include "texture_rasterizer.hpp"

namespace IDW {

struct SamplePoint {
  gsm::vec3 pos;
  gsm::vec2 uv;
  gsm::ivec2 screenPos;
};

} // namespace IDW

namespace {

__device__ bool queryLogMap(gsm::vec3 p, const gsm::vec3 *__restrict__ pts3d,
                            const gsm::vec2 *__restrict__ uvs, const int *__restrict__ gridData,
                            const int *gridOffsets, int gridRes, gsm::vec3 gridMin, float cellSize,
                            gsm::vec2 *out_uv) {

  // find the cell
  gsm::ivec3 c = gsm::clamp(gsm::floor((p - gridMin) / cellSize), 0, gridRes - 1);

  float best = 1e18f;
  int bestIdx = -1;

  const int radius = 1;

  // query 3x3x3 neighborhood
  // (LogMapTable's points are closer than GaussianGrid, enough for 1x1x1, safe to go 3x3x3)
  for (int dz = -radius; dz <= radius; ++dz) {
    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
        int nx = c.x + dx, ny = c.y + dy, nz = c.z + dz;
        if (nx < 0 || ny < 0 || nz < 0 || nx >= gridRes || ny >= gridRes || nz >= gridRes)
          continue;

        int cellIdx = nx + gridRes * (ny + gridRes * nz);
        int start = gridOffsets[cellIdx];
        int end = gridOffsets[cellIdx + 1];

        for (int i = start; i < end; ++i) {
          float d2 = gsm::lengthSquared(p - pts3d[i]);
          if (d2 < best) {
            best = d2;
            bestIdx = i;
          }
        }
      }
    }
  }

  if (bestIdx < 0)
    return false;

  *out_uv = uvs[bestIdx];
  return true;
}

__device__ gsm::ivec2 projectPoint(const gsm::vec3 p, const float *__restrict__ projviewMatrix, int width,
                                   int height) {
  // World → Clip space
  gsm::vec4 clip = gsm::mat4(projviewMatrix) * gsm::vec4(p, 1.0f);

  // Perspective divide → NDC (-1 to 1)
  if (std::abs(clip.w) < 1e-5f)
    return {-1, -1};
  gsm::vec3 ndc = gsm::vec3(clip) / clip.w;
  if (ndc.z < -1.f || ndc.z > 1.f)
    return {-1, -1};

  int px = __float2int_rd((ndc.x * 0.5f + 0.5f) * width);
  int py = __float2int_rd((ndc.y * 0.5f + 0.5f) * height); // don't flip Y
  if (px < 0 || px >= width || py < 0 || py >= height)
    return {-1, -1};

  return {px, py};
}

__global__ void projectPointsKernel(const gsm::vec3 *__restrict__ lastPoints,
                                    gsm::vec2 *__restrict__ lastPoints2D, int nLastPoints,
                                    const float *__restrict__ projviewmatrix, int width, int height) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nLastPoints)
    return;

  lastPoints2D[i] = gsm::vec2(projectPoint(lastPoints[i], projviewmatrix, width, height));
}

__global__ void projectPts3dToMask(const gsm::vec3 *__restrict__ pts3d, int nPts,
                                   const gsm::vec2 *__restrict__ uvs, const int *__restrict__ gridData,
                                   const int *__restrict__ gridOffsets, int gridRes, float3 gridMin,
                                   float cellSize, float R,
                                   const float *__restrict__ projviewMatrix, // 4x4 colflip projview
                                   int width, int height, IDW::SamplePoint *__restrict__ mask) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nPts)
    return;

  gsm::ivec2 p = projectPoint(pts3d[i], projviewMatrix, width, height);
  if (p.x == -1 || p.y == -1)
    return;

  gsm::vec2 uv;
  if (!queryLogMap(pts3d[i], (gsm::vec3 *)pts3d, uvs, gridData, gridOffsets, gridRes, gsm::vec3(gridMin),
                   cellSize, &uv))
    return;

  gsm::vec3 scenemin{-6.34973, -5.422, -5.77145};
  gsm::vec3 scenemax{6.35174, 9.7087, 5.76965};
  gsm::vec3 scenerange = scenemax - scenemin;

  IDW::SamplePoint &m = mask[i];
  m.pos = pts3d[i];
  m.uv = uv / (2.f * R) + 0.5f;
  m.screenPos = p;
  // m.mask = 1;
  // normalize to [0,1]
  // m.texCoords = {uv.x / (2.f * R) + 0.5f, uv.y / (2.f * R) + 0.5f};
  // m.normal = {uv.x / (2.f * R) + 0.5f, uv.y / (2.f * R) + 0.5f, 0};
}

__global__ void dilateKernel(const CudaRasterizer::PixelMask *__restrict__ src,
                             CudaRasterizer::PixelMask *__restrict__ out_mask, int width, int height,
                             int radius) {

  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= width || py >= height)
    return;

  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      int nx = px + dx, ny = py + dy;
      if (nx < 0 || ny < 0 || nx >= width || ny >= height)
        continue;
      if (src[ny * width + nx].mask) {
        out_mask[py * width + px] = {src[ny * width + nx]};
        return;
      }
    }
  }

  out_mask[py * width + px].mask = 0;
}

} // namespace

/*
namespace JFA {

// JFA: record the nearest sparse pixel in each pixel

// if the pixel in sparseUV is not empty, the initial seed is the sparse pixel
// otherwise, the initial seed is (-1, -1)

__global__ void jfaInit(const CudaRasterizer::PixelMask *__restrict__ sparseMask,
                        gsm::ivec2 *__restrict__ seeds, int width, int height) {

  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= width || py >= height)
    return;

  int pid = py * width + px;
  const CudaRasterizer::PixelMask &m = sparseMask[pid];
  if (!m.mask) {
    seeds[pid] = {px, py};
  } else {
    seeds[pid] = {-1, -1};
  }
}

__global__ void jfaStep(const gsm::ivec2 *__restrict__ src, gsm::ivec2 *__restrict__ dst, int width,
                        int height, int step) {

  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= width || py >= height)
    return;

  int pid = py * width + px;
  gsm::ivec2 best = src[pid];
  float bestDist = (best.x < 0)
                       ? 1e18f
                       : static_cast<float>((px - best.x) * (px - best.x) + (py - best.y) * (py - best.y));

  // 8 directions, jump `step` pixels each time
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0)
        continue;
      int nx = px + dx * step;
      int ny = py + dy * step;
      if (nx < 0 || ny < 0 || nx >= width || ny >= height)
        continue;

      gsm::ivec2 nb = src[ny * width + nx];
      if (nb.x < 0)
        continue;

      float d = (float)((px - nb.x) * (px - nb.x) + (py - nb.y) * (py - nb.y));
      if (d < bestDist) {
        bestDist = d;
        best = nb;
      }
    }
  }

  dst[pid] = best;
}

__global__ void jfaApplyMask(const gsm::ivec2 *__restrict__ seeds, const uint8_t *__restrict__ origMask,
                             const float *t_final, float coverage_threshold,
                             CudaRasterizer::PixelMask *__restrict__ out_mask, int width, int height,
                             float maxScreenDist // 超過這個螢幕距離就不填
) {

  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= width || py >= height)
    return;

  int pid = py * width + px;
  out_mask[pid].mask = 0;

  // coverage gate
  if ((1.f - t_final[pid]) < coverage_threshold)
    return;

  gsm::ivec2 s = seeds[pid];
  if (s.x < 0)
    return;

  // 距離太遠，不填（控制 patch 邊界）
  float dist = sqrtf((float)((px - s.x) * (px - s.x) + (py - s.y) * (py - s.y)));
  if (dist > maxScreenDist)
    return;

  out_mask[pid].mask = 1;
  // out_mask[pid].texCoords = s.uv;
}

} // namespace JFA
*/

namespace IDW {

// __global__ void interpolate(const CudaRasterizer::PixelMask *__restrict__ sparseMask,
//                             const float *__restrict__ t_final, int width, int height, float maxScreenDist,
//                             cudaTextureObject_t model_basecolor_map_cuda,
//                             CudaRasterizer::PixelMask *__restrict__ out_mask) {

//   int px = blockIdx.x * blockDim.x + threadIdx.x;
//   int py = blockIdx.y * blockDim.y + threadIdx.y;
//   if (px >= width || py >= height)
//     return;

//   int pid = py * width + px;

//   if (sparseMask[pid].mask) {
//     out_mask[pid] = sparseMask[pid];
//     return;
//   } else {
//     out_mask[pid].mask = 0;
//   }

//   if ((1.f - t_final[pid]) < 0.5f)
//     return;

//   int radius = (int)maxScreenDist;
//   float2 uvSum = {0.f, 0.f};
//   float wSum = 0.f;
//   int count = 0;

//   for (int dy = -radius; dy <= radius; ++dy)
//     for (int dx = -radius; dx <= radius; ++dx) {
//       int nx = px + dx, ny = py + dy;
//       if (nx < 0 || ny < 0 || nx >= width || ny >= height)
//         continue;

//       const CudaRasterizer::PixelMask &m = sparseMask[ny * width + nx];
//       if (!m.mask)
//         continue;

//       float dist = sqrtf((float)(dx * dx + dy * dy));
//       if (dist > maxScreenDist)
//         continue;
//       if (dist < 1e-3f)
//         dist = 1e-3f;

//       // IDW (Inverse Distance Weighting)
//       float w = 1.f / (dist * dist);
//       float2 uv = m.texCoords;
//       uvSum.x += w * uv.x;
//       uvSum.y += w * uv.y;
//       wSum += w;
//       count++;
//     }

//   if (count == 0 || wSum < 1e-8f)
//     return;

//   out_mask[pid].mask = 1;
//   out_mask[pid].texCoords = {uvSum.x / wSum, uvSum.y / wSum};
//   // out_mask[pid].normal
//   float4 texColor = CudaRasterizer::sampleTexture(model_basecolor_map_cuda, out_mask[pid].texCoords);
//   out_mask[pid].color = {texColor.x, texColor.y, texColor.z};
//   // out_mask[pid].depth
// }

__global__ void idwKernel(const SamplePoint *__restrict__ samplePoints, int nPts, int width, int height,
                          float power, CudaRasterizer::PixelMask *__restrict__ out_mask) {
  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= width || py >= height)
    return;
  int pid = py * width + px;

  float weightSum = 0.0f;
  gsm::vec2 valueSum{0.0f};

  for (int i = 0; i < nPts; ++i) {

    float dx = px - samplePoints[i].screenPos.x;
    float dy = py - samplePoints[i].screenPos.y;
    float d2 = dx * dx + dy * dy;

    if (d2 < 1e-12f) {
      out_mask[pid].mask = 1;
      out_mask[pid].texCoords = (float2)(samplePoints[i].uv);
      out_mask[pid].normal = {samplePoints[i].uv.x, samplePoints[i].uv.y, 0};
      return;
    }

    float w = __powf(d2, -power * 0.5f); // d^(-p) = (d2)^(-p/2)
    weightSum += w;
    valueSum += w * samplePoints[i].uv.x;
  }

  out_mask[pid].mask = 1;
  out_mask[pid].texCoords = (float2)(valueSum / weightSum);
  out_mask[pid].normal = {out_mask[pid].texCoords.x, out_mask[pid].texCoords.y, 0};
}

} // namespace IDW

namespace {

__device__ bool hit(gsm::vec2 p, int width, int height, float depth_val, float T_val,
                    // camera
                    float tan_fovx, float tan_fovy, const float *__restrict__ inverse_modelview,
                    // output
                    gsm::vec3 *hitPoint) {
  if ((1.0f - T_val) < 0.5f)
    return false;

  gsm::vec2 ndcPos{
      (p.x / width) * 2.0f - 1.0f,
      1.0f - (p.y / height) * 2.0f // flip Y
  };

  gsm::vec3 viewspace_pos{ndcPos.x * depth_val * tan_fovx,
                          -ndcPos.y * depth_val * tan_fovy, // flip Y axis
                          depth_val};

  // pos_world = inv(colmap_view) * viewspace_pos
  gsm::vec4 pos_world = gsm::mat4(inverse_modelview) * gsm::vec4(viewspace_pos, 1.0f);

  *hitPoint = gsm::vec3(pos_world);
  return true;
}

__global__ void getFullMask(
    // LogMap
    const gsm::vec3 *pts3d, int nPts, const gsm::vec2 *uvs, const int *gridData, const int *gridOffsets,
    int gridRes, float3 gridMin, float cellSize, float R,
    // last points polygon
    const gsm::vec2 *__restrict__ lastPoints, int nLastPoints, int width, int height,
    // camera
    const float *__restrict__ inverse_colmap_viewmatrix, float tan_fovx, float tan_fovy,
    // depth, T
    const float *__restrict__ depth_raw, const float *__restrict__ t_final,
    CudaRasterizer::PixelMask *__restrict__ mask) {
  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= width || py >= height)
    return;
  int pid = py * width + px;
  gsm::vec2 p = gsm::vec2(px, py);

  // if (pointInPolygon(lastPoints, nLastPoints, p)) {
  //   mask[pid].mask = 1;
  // }

  gsm::vec3 scenemin{-6.34973, -5.422, -5.77145};
  gsm::vec3 scenemax{6.35174, 9.7087, 5.76965};
  gsm::vec3 scenerange = scenemax - scenemin;

  gsm::vec3 hitPoint;
  if (!hit(p, width, height, depth_raw[pid], t_final[pid], tan_fovx, tan_fovy, inverse_colmap_viewmatrix,
           &hitPoint))
    return;

  // gsm::vec2 uv;
  // if (!queryLogMap(hitPoint, pts3d, uvs, gridData, gridOffsets, gridRes, gsm::vec3(gridMin), cellSize,
  // &uv))
  //   return;

  // mask[pid].mask = 1;
  // // mask[pid].normal = (float3)((hitPoint - scenemin) / scenerange);
  // mask[pid].normal = {uv.x / (2.f * R) + 0.5f, uv.y / (2.f * R) + 0.5f, 0};
}

} // namespace

void CudaRasterizer::makeGeodesicsMask(const float *depth_raw, const float *t_final,
                                       // LogMap
                                       const float *pts3d, int nPts, const float *uvs, const int *gridData,
                                       const int *gridOffsets, int gridRes, float3 gridMin, float cellSize,
                                       float R, const float *lastPoints, int nLastPoints,
                                       // camera
                                       const float *colmap_projviewmatrix, int width, int height,
                                       const float *inverse_colmap_viewmatrix, float tan_fovx, float tan_fovy,
                                       // texture
                                       cudaTextureObject_t model_basecolor_map_cuda,
                                       cudaTextureObject_t model_normal_map_cuda,
                                       cudaTextureObject_t model_height_map_cuda,
                                       // output
                                       PixelMask *mask) {

  if (depth_raw == nullptr || t_final == nullptr || pts3d == nullptr || uvs == nullptr ||
      gridData == nullptr || gridOffsets == nullptr || mask == nullptr || R <= 0)
    return;

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / 16, (height + block.y - 1) / 16);

  IDW::SamplePoint *samplePoints;
  size_t pixels = (size_t)width * height;
  cudaMalloc(&samplePoints, pixels * sizeof(IDW::SamplePoint));
  cudaMemset(samplePoints, 0, pixels * sizeof(IDW::SamplePoint));
  cudaMemset(mask, 0, pixels * sizeof(CudaRasterizer::PixelMask));

  projectPts3dToMask<<<(nPts + 255) / 256, 256>>>((gsm::vec3 *)pts3d, nPts, (gsm::vec2 *)uvs, gridData,
                                                  gridOffsets, gridRes, gridMin, cellSize, R,
                                                  colmap_projviewmatrix, width, height, samplePoints);

  // IDW::interpolate<<<grid, block>>>(sparseMask, t_final, width, height,
  //                                   10.f, // maxScreenDist, controls the interpolation range
  //                                   model_basecolor_map_cuda, mask);

  // dilateKernel<<<grid, block>>>(src_mask, mask, width, height, 0);

  // IDW::idwKernel<<<grid, block>>>(samplePoints, nPts, width, height, 1, mask);

  gsm::vec2 *lastPoints2D;
  cudaMalloc(&lastPoints2D, nLastPoints * sizeof(gsm::vec2));
  projectPointsKernel<<<(nLastPoints + 255) / 256, 256>>>((gsm::vec3 *)lastPoints, lastPoints2D, nLastPoints,
                                                          colmap_projviewmatrix, width, height);

  getFullMask<<<grid, block>>>((gsm::vec3 *)pts3d, nPts, (gsm::vec2 *)uvs, gridData, gridOffsets, gridRes,
                               gridMin, cellSize, R, lastPoints2D, nLastPoints, width, height,
                               inverse_colmap_viewmatrix, tan_fovx, tan_fovy, depth_raw, t_final, mask);

  cudaFree(lastPoints2D);
  cudaFree(samplePoints);
}
