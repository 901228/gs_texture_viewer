#include "rasterizer.hpp"

#include <cfloat>
#include <cstdint>

#include <device_types.h>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "auxiliary.h"
#include "config.hpp"
#include "forward.hpp"

template <typename T> static void obtain(char *&chunk, T *&ptr, std::size_t count, std::size_t alignment) {
  std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
  ptr = reinterpret_cast<T *>(offset);
  chunk = reinterpret_cast<char *>(ptr + count);
}

template <typename T> size_t required(size_t P) {
  char *size = nullptr;
  T::fromChunk(size, P);
  return ((size_t)size) + 128;
}

struct GeometryState {
  size_t scan_size;
  float *depths;
  char *scanning_space;
  bool *clamped;
  int *internal_radii;
  float2 *means2D;
  float *cov3D;
  float4 *conic_opacity;
  float *rgb;
  uint32_t *point_offsets;
  uint32_t *tiles_touched;

  static GeometryState fromChunk(char *&chunk, size_t P) {
    GeometryState geom{};
    obtain(chunk, geom.depths, P, 128);
    obtain(chunk, geom.clamped, P * 3, 128);
    obtain(chunk, geom.internal_radii, P, 128);
    obtain(chunk, geom.means2D, P, 128);
    obtain(chunk, geom.cov3D, P * 6, 128);
    obtain(chunk, geom.conic_opacity, P, 128);
    obtain(chunk, geom.rgb, P * 3, 128);
    obtain(chunk, geom.tiles_touched, P, 128);
    cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
    obtain(chunk, geom.scanning_space, geom.scan_size, 128);
    obtain(chunk, geom.point_offsets, P, 128);
    return geom;
  }
};

struct ImageState {
  uint2 *ranges;
  uint32_t *n_contrib;
  float *accum_alpha;

  static ImageState fromChunk(char *&chunk, size_t N) {
    ImageState img{};
    obtain(chunk, img.accum_alpha, N, 128);
    obtain(chunk, img.n_contrib, N, 128);
    obtain(chunk, img.ranges, N, 128);
    return img;
  }
};

struct BinningState {
  size_t sorting_size;
  uint64_t *point_list_keys_unsorted;
  uint64_t *point_list_keys;
  uint32_t *point_list_unsorted;
  uint32_t *point_list;
  char *list_sorting_space;

  static BinningState fromChunk(char *&chunk, size_t P) {
    BinningState binning{};
    obtain(chunk, binning.point_list, P, 128);
    obtain(chunk, binning.point_list_unsorted, P, 128);
    obtain(chunk, binning.point_list_keys, P, 128);
    obtain(chunk, binning.point_list_keys_unsorted, P, 128);
    cub::DeviceRadixSort::SortPairs(nullptr, binning.sorting_size, binning.point_list_keys_unsorted,
                                    binning.point_list_keys, binning.point_list_unsorted, binning.point_list,
                                    P);
    obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
    return binning;
  }
};

// Helper function to find the next-highest bit of the MSB on the CPU.
uint32_t getHigherMsb(uint32_t n) {
  uint32_t msb = sizeof(n) * 4;
  uint32_t step = msb;
  while (step > 1) {
    step /= 2;
    if (n >> msb)
      msb += step;
    else
      msb -= step;
  }
  if (n >> msb)
    msb++;
  return msb;
}

// Generates one key/value pair for all Gaussian / tile overlaps.
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(int P, const float2 *points_xy, const float *depths,
                                  const uint32_t *offsets, uint64_t *gaussian_keys_unsorted,
                                  uint32_t *gaussian_values_unsorted, const int *radii, dim3 grid,
                                  int2 *rects) {
  auto idx = cooperative_groups::__v1::grid_group::thread_rank();
  if (idx >= P)
    return;

  // Generate no key/value pair for invisible Gaussians
  if (radii[idx] > 0) {
    // Find this Gaussian's offset in buffer for writing keys/values.
    uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
    uint2 rect_min, rect_max;

    if (rects == nullptr)
      getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);
    else
      getRect(points_xy[idx], rects[idx], rect_min, rect_max, grid);

    // For each tile that the bounding rect overlaps, emit a
    // key/value pair. The key is |  tile ID  |      depth      |,
    // and the value is the ID of the Gaussian. Sorting the values
    // with this key yields Gaussian IDs in a list, such that they
    // are first sorted by tile and then by depth.
    for (unsigned int y = rect_min.y; y < rect_max.y; y++) {
      for (unsigned int x = rect_min.x; x < rect_max.x; x++) {
        uint64_t key = y * grid.x + x;
        key <<= 32; // NOLINT(hicpp-signed-bitwise)
        key |= *((uint32_t *)&depths[idx]);
        gaussian_keys_unsorted[off] = key;
        gaussian_values_unsorted[off] = idx;
        off++;
      }
    }
  }
}

// Check keys to see if it is at the start/end of one tile's range in
// the full sorted list. If yes, write start/end of this tile.
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, const uint64_t *point_list_keys, uint2 *ranges) {
  auto idx = cooperative_groups::__v1::grid_group::thread_rank();
  if (idx >= L)
    return;

  // Read tile ID from key. Update start/end of tile range if at limit.
  uint64_t key = point_list_keys[idx];
  uint32_t currtile = key >> 32; // NOLINT(hicpp-signed-bitwise)
  if (idx == 0)
    ranges[currtile].x = 0;
  else {
    uint32_t prevtile = point_list_keys[idx - 1] >> 32; // NOLINT(hicpp-signed-bitwise)
    if (currtile != prevtile) {
      ranges[prevtile].y = idx;
      ranges[currtile].x = idx;
    }
    if (idx == L - 1)
      ranges[currtile].y = L;
  }
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::forward(
    const std::function<char *(size_t)> &geometryBuffer, const std::function<char *(size_t)> &binningBuffer,
    const std::function<char *(size_t)> &imageBuffer, int P, int D, int M, const float *background, int width,
    int height, const float *means3D, const float *shs, const float *colors_precomp, const float *opacities,
    const float *scales, float scale_modifier, const float *rotations, const float *cov3D_precomp,
    const float *viewmatrix, const float *projmatrix, const float *cam_pos, float tan_fovx, float tan_fovy,
    bool prefiltered, float *out_color, bool antialiasing, int *radii, int *rects, const float *boxmin,
    const float *boxmax, const PixelMask *mask, float threshold, TextureOption textureOption) {

  const float focal_y = static_cast<float>(height) / (2.0f * tan_fovy);
  const float focal_x = static_cast<float>(width) / (2.0f * tan_fovx);

  size_t chunk_size = required<GeometryState>(P);
  char *chunkptr = geometryBuffer(chunk_size);
  GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

  if (radii == nullptr) {
    radii = geomState.internal_radii;
  }

  dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
  dim3 block(BLOCK_X, BLOCK_Y, 1);

  // Dynamically resize image-based auxiliary buffers during training
  size_t img_chunk_size = required<ImageState>(width * height);
  char *img_chunkptr = imageBuffer(img_chunk_size);
  ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

  if (NUM_CHANNELS != 3 && colors_precomp == nullptr) {
    throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
  }

  float3 minn = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
  float3 maxx = {FLT_MAX, FLT_MAX, FLT_MAX};
  if (boxmin != nullptr) {
    minn = *((float3 *)boxmin);
    maxx = *((float3 *)boxmax);
  }

  // Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs
  // to RGB)
  FORWARD::preprocess(P, D, M, means3D, (rs::vec3 *)scales, scale_modifier, (rs::vec4 *)rotations, opacities,
                      shs, geomState.clamped, cov3D_precomp, colors_precomp, viewmatrix, projmatrix,
                      (rs::vec3 *)cam_pos, width, height, focal_x, focal_y, tan_fovx, tan_fovy, radii,
                      geomState.means2D, geomState.depths, geomState.cov3D, geomState.rgb,
                      geomState.conic_opacity, tile_grid, geomState.tiles_touched, prefiltered, (int2 *)rects,
                      minn, maxx, antialiasing);

  // Compute prefix sum over full list of touched tile counts by Gaussians
  // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
  cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched,
                                geomState.point_offsets, P);

  // Retrieve total number of Gaussian instances to launch and resize aux
  // buffers
  int num_rendered;
  cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost);

  if (num_rendered == 0)
    return 0;

  size_t binning_chunk_size = required<BinningState>(num_rendered);
  char *binning_chunkptr = binningBuffer(binning_chunk_size);
  BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

  // For each instance to be rendered, produce adequate [ tile | depth ] key
  // and corresponding duplicated Gaussian indices to be sorted
  duplicateWithKeys<<<(P + 255) / 256, 256>>>(
      P, geomState.means2D, geomState.depths, geomState.point_offsets, binningState.point_list_keys_unsorted,
      binningState.point_list_unsorted, radii, tile_grid, (int2 *)rects);

  int bit = static_cast<int>(getHigherMsb(tile_grid.x * tile_grid.y));

  // Sort complete list of (duplicated) Gaussian indices by keys
  cub::DeviceRadixSort::SortPairs(binningState.list_sorting_space, binningState.sorting_size,
                                  binningState.point_list_keys_unsorted, binningState.point_list_keys,
                                  binningState.point_list_unsorted, binningState.point_list, num_rendered, 0,
                                  32 + bit);

  cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2));

  // Identify start and end of per-tile workloads in sorted list
  identifyTileRanges<<<(num_rendered + 255) / 256, 256>>>(num_rendered, binningState.point_list_keys,
                                                          imgState.ranges);

  // Let each tile blend its range of Gaussians independently in parallel
  const float *feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
  FORWARD::render(tile_grid, block, imgState.ranges, binningState.point_list, width, height,
                  geomState.means2D, geomState.depths, feature_ptr, geomState.conic_opacity,
                  imgState.accum_alpha, imgState.n_contrib, background, out_color, mask, threshold,
                  textureOption);

  return num_rendered;
}
