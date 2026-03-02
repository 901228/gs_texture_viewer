#include "texture_rasterizer.hpp"

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <device_launch_parameters.h>

#include "vector/matrix.hpp"
#include "vector/vector.hpp"
namespace rs = rasterizer;

namespace {

__global__ void clipPosition(const float3 *position, int num_vertices, const float *viewmatrix,
                             const float *projmatrix, rs::vec4 *out_clip_pos) {
  unsigned int vid = blockIdx.x * blockDim.x + threadIdx.x;
  if (vid >= num_vertices)
    return;

  float3 vertex = position[vid];

  out_clip_pos[vid] = rs::mat4(projmatrix) * rs::mat4(viewmatrix) * rs::vec4(vertex, 1.0f);
}

/**
 * Computes the signed area of a triangle given by three vertices.
 *
 * If result > 0, vertex c is on the left of the directed edge ab.
 * If result < 0, vertex c is on the right of the directed edge ab.
 * If result == 0, vertex c is on the directed edge ab.
 */
__device__ float edge_function(float2 a, float2 b, float2 c) {
  return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

// convert float depth to a uint that can be used for atomicMin (keeps sorting correct)
__device__ uint32_t float_to_uint_depth(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof(u));
  // positive floats IEEE754 bit pattern can be compared directly
  return u;
}

// for each triangle
__global__ void render(const float3 *__restrict__ position, const rs::vec4 *__restrict__ clip_pos,
                       const float2 *__restrict__ texCoords, const cudaTextureObject_t *__restrict__ sl,
                       int num_triangles, const uint8_t *__restrict__ face_mask, int width, int height,
                       uint32_t *__restrict__ depth_buffer, // temp uint depth buffer for atomicMin
                       CudaRasterizer::PixelMask *__restrict__ mask) {
  unsigned int prim_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (prim_id >= num_triangles)
    return;

  if (face_mask != nullptr && face_mask[prim_id] == 0)
    return;

  // triangle vertex id
  int i0 = static_cast<int>(prim_id) * 3 + 0;
  int i1 = static_cast<int>(prim_id) * 3 + 1;
  int i2 = static_cast<int>(prim_id) * 3 + 2;

  // extract clip position of each triangle vertex
  rs::vec4 c0 = clip_pos[i0];
  rs::vec4 c1 = clip_pos[i1];
  rs::vec4 c2 = clip_pos[i2];

  // get texture coordinates
  float2 uv0 = texCoords[i0];
  float2 uv1 = texCoords[i1];
  float2 uv2 = texCoords[i2];

  // get texture ID
  cudaTextureObject_t texId = sl[prim_id];

  // Perspective divide → NDC
  // (screenX, screenY, depth)
  rs::vec3 ndc0 = rs::vec3(c0) / (c0.w + FLT_EPSILON);
  rs::vec3 ndc1 = rs::vec3(c1) / (c1.w + FLT_EPSILON);
  rs::vec3 ndc2 = rs::vec3(c2) / (c2.w + FLT_EPSILON);

  if (isnan(ndc0.x) || isnan(ndc1.x) || isnan(ndc2.x) || isnan(ndc0.y) || isnan(ndc1.y) || isnan(ndc2.y) ||
      isnan(ndc0.z) || isnan(ndc1.z) || isnan(ndc2.z)) {
    return;
  }

  // NDC → screen space (viewport transform)
  float hw = static_cast<float>(width) * 0.5f;
  float hh = static_cast<float>(height) * 0.5f;

  float2 s0 = {(ndc0.x + 1.0f) * hw, (1.0f - ndc0.y) * hh};
  float2 s1 = {(ndc1.x + 1.0f) * hw, (1.0f - ndc1.y) * hh};
  float2 s2 = {(ndc2.x + 1.0f) * hw, (1.0f - ndc2.y) * hh};

  // Bounding box (clamp to screen range)
  int minX = rs::max(0, static_cast<int>(rs::min(rs::min(s0.x, s1.x), s2.x)));
  int minY = rs::max(0, static_cast<int>(rs::min(rs::min(s0.y, s1.y), s2.y)));
  int maxX = rs::min(width - 1, rs::ceil(rs::max(rs::max(s0.x, s1.x), s2.x)));
  int maxY = rs::min(height - 1, rs::ceil(rs::max(rs::max(s0.y, s1.y), s2.y)));

  float area = edge_function(s0, s1, s2);
  if (area == 0.0f)
    return; // degenerate triangle

  // iterate over all pixels in the bounding box (triangle)
  for (int py = minY; py <= maxY; py++) {
    for (int px = minX; px <= maxX; px++) {

      int pixel = py * width + px;
      float2 p = {static_cast<float>(px) + 0.5f, static_cast<float>(py) + 0.5f};

      // Compute barycentric coordinates of the pixel
      float w0 = edge_function(s1, s2, p);
      float w1 = edge_function(s2, s0, p);
      float w2 = edge_function(s0, s1, p);

      // if signs are the same, the pixel is inside the triangle
      bool inside =
          (area > 0.0f) ? (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f) : (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f);

      if (!inside)
        continue;

      // barycentric coordinates are normalized to [0, 1]
      float bary0 = w0 / area;
      float bary1 = w1 / area;
      float bary2 = w2 / area;

      // interpolate depth from barycentric coordinates
      float depth = bary0 * ndc0.z + bary1 * ndc1.z + bary2 * ndc2.z;

      // depth test: atomicMin ensures the nearest primitive wins
      uint32_t depth_uint = float_to_uint_depth(depth);
      uint32_t old_depth = atomicMin(&depth_buffer[pixel], depth_uint);

      // if the depth is closer (closer to the camera), update primitive ID
      // (possible race condition, but acceptable for selection use case)
      if (depth_uint < old_depth) {

        CudaRasterizer::PixelMask &m = mask[pixel];

        m.mask = 1;
        m.texId = texId;

        float2 uv;
        uv.x = bary0 * uv0.x + bary1 * uv1.x + bary2 * uv2.x;
        uv.y = bary0 * uv0.y + bary1 * uv1.y + bary2 * uv2.y;
        m.texCoords = uv;

        m.depth = bary0 * c0.w + bary1 * c1.w + bary2 * c2.w;
      }
    }
  }
}

} // namespace

void CudaRasterizer::makeMask(const float *position, const float *texCoords, int num_vertices,
                              const cudaTextureObject_t *sl, int num_triangles, const uint8_t *face_mask,
                              int width, int height, const float *viewmatrix, const float *projmatrix,
                              PixelMask *mask) {

  rs::vec4 *clip_pos;
  uint32_t *d_depth;

  size_t pixels = (size_t)width * height;

  cudaMalloc(&clip_pos, num_vertices * sizeof(rs::vec4));
  cudaMalloc(&d_depth, pixels * sizeof(uint32_t));

  // initialize depth buffer to max value
  cudaMemset(d_depth, static_cast<int>(0xFFFFFFFF), pixels * sizeof(uint32_t));

  // clear output buffers
  cudaMemset(mask, 0, width * height * sizeof(PixelMask));

  // ── Vertex Shader ──
  clipPosition<<<(num_vertices + 255) / 256, 256>>>((float3 *)position, num_vertices, viewmatrix, projmatrix,
                                                    clip_pos);

  // ── Rasterize + Fragment Shader ──
  render<<<(num_triangles + 255) / 256, 256>>>((float3 *)position, clip_pos, (float2 *)texCoords, sl,
                                               num_triangles, face_mask, width, height, d_depth, mask);

  cudaFree(clip_pos);
  cudaFree(d_depth);
}
