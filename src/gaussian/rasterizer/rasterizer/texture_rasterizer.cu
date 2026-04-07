#include "texture_rasterizer.hpp"

#include <cfloat>
#include <cstddef>
#include <cstdint>

#include <device_launch_parameters.h>

#include "gsm/gsm.cuh"

__device__ float4 CudaRasterizer::sampleTexture(cudaTextureObject_t texId, float2 texCoord,
                                                TextureOption textureOption) {

  gsm::mat2 rotationMatrix = {std::cosf(textureOption.theta), -std::sinf(textureOption.theta),
                              std::sinf(textureOption.theta), std::cosf(textureOption.theta)};
  gsm::vec2 t = rotationMatrix * ((gsm::vec2(texCoord) - 0.5) * textureOption.scale) + 0.5 +
                gsm::vec2(textureOption.offset);
  return tex2D<float4>(texId, t.x, t.y);
}

namespace {

__host__ __device__ gsm::vec3 barycentric(gsm::vec3 bary, gsm::vec3 p0, gsm::vec3 p1, gsm::vec3 p2) {
  return bary.x * p0 + bary.y * p1 + bary.z * p2;
}
__host__ __device__ gsm::vec2 barycentric(gsm::vec3 bary, gsm::vec2 p0, gsm::vec2 p1, gsm::vec2 p2) {
  return bary.x * p0 + bary.y * p1 + bary.z * p2;
}
__host__ __device__ float barycentric(gsm::vec3 bary, float p0, float p1, float p2) {
  return bary.x * p0 + bary.y * p1 + bary.z * p2;
}

struct VertexOut {
  gsm::vec4 clipPos;
  gsm::vec3 position;
  gsm::vec3 normal;
  gsm::vec3 tangent;
  gsm::vec3 bitangent;
  gsm::vec2 uv;
  float depth;
};

__global__ void tessellate(const gsm::vec3 *__restrict__ position,              // per-vertex
                           const gsm::vec3 *__restrict__ normals,               // per-vertex
                           const gsm::vec2 *__restrict__ texCoords,             // per-vertex
                           const gsm::vec3 *__restrict__ tangent,               // per-vertex
                           const gsm::vec3 *__restrict__ bitangent,             // per-vertex
                           const cudaTextureObject_t *__restrict__ heightTexId, // per-face
                           CudaRasterizer::TextureOption textureOption, int num_vertices,
                           int num_coarse_triangles, float heightScale, int tessLevel,
                           const float *__restrict__ viewmatrix, const float *__restrict__ projmatrix,
                           VertexOut *__restrict__ vertex_out) {
  // every triangle generates tessLevel² fine triangles
  int tris_per_patch = tessLevel * tessLevel;
  int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_tid >= num_coarse_triangles * tris_per_patch)
    return;

  int patch_id = global_tid / tris_per_patch;
  int local_tid = global_tid % tris_per_patch;

  // vertex of the coarse triangle
  int i0 = patch_id * 3 + 0;
  int i1 = patch_id * 3 + 1;
  int i2 = patch_id * 3 + 2;

  gsm::vec3 pos0 = position[i0];
  gsm::vec3 pos1 = position[i1];
  gsm::vec3 pos2 = position[i2];

  gsm::vec3 norm0 = normals[i0];
  gsm::vec3 norm1 = normals[i1];
  gsm::vec3 norm2 = normals[i2];

  gsm::vec2 uv0 = texCoords[i0];
  gsm::vec2 uv1 = texCoords[i1];
  gsm::vec2 uv2 = texCoords[i2];

  gsm::vec3 tang0 = tangent[i0];
  gsm::vec3 tang1 = tangent[i1];
  gsm::vec3 tang2 = tangent[i2];

  gsm::vec3 bitang0 = bitangent[i0];
  gsm::vec3 bitang1 = bitangent[i1];
  gsm::vec3 bitang2 = bitangent[i2];

  // local_tid -> center coordinate
  // traverse tessLevel² sub-triangles
  // use row-major ordering: row i has 2*(tessLevel-i)-1 triangles
  // simple approach: find (row, col, upper/lower)
  float inv = 1.0f / static_cast<float>(tessLevel);

  // mapping local_tid to the coordinates of the three vertices
  // find the row first
  int row = 0, col = 0;
  int count = 0;
  bool found = false;
  for (int r = 0; r < tessLevel && !found; r++) {
    int tris_in_row = 2 * (tessLevel - r) - 1;
    if (local_tid < count + tris_in_row) {
      row = r;
      col = local_tid - count;
      found = true;
    }
    count += tris_in_row;
  }

  // index in col: even = upward triangle, odd = downward triangle
  bool upward = (col % 2 == 0);
  int k = col / 2;

  gsm::vec3 b0, b1, b2; // center coordinate of three vertices
  if (upward) {
    // upward triangle
    b0 = {(row + 0) * inv, (k + 0) * inv, 1.0f - (row + 0) * inv - (k + 0) * inv};
    b1 = {(row + 1) * inv, (k + 0) * inv, 1.0f - (row + 1) * inv - (k + 0) * inv};
    b2 = {(row + 0) * inv, (k + 1) * inv, 1.0f - (row + 0) * inv - (k + 1) * inv};
  } else {
    // downward triangle
    b0 = {(row + 1) * inv, (k + 0) * inv, 1.0f - (row + 1) * inv - (k + 0) * inv};
    b1 = {(row + 1) * inv, (k + 1) * inv, 1.0f - (row + 1) * inv - (k + 1) * inv};
    b2 = {(row + 0) * inv, (k + 1) * inv, 1.0f - (row + 0) * inv - (k + 1) * inv};
  }

  // output fine vertex index
  int out_base = global_tid * 3;

  // interpolate attributes
  gsm::vec3 bary_list[3] = {b0, b1, b2};
  for (int v = 0; v < 3; v++) {
    gsm::vec3 bary = bary_list[v];

    // barycentric interpolation
    gsm::vec3 pos = barycentric(bary, pos0, pos1, pos2);
    gsm::vec3 norm = gsm::normalize(barycentric(bary, norm0, norm1, norm2));
    gsm::vec3 tang = gsm::normalize(barycentric(bary, tang0, tang1, tang2));
    gsm::vec3 bitang = gsm::normalize(barycentric(bary, bitang0, bitang1, bitang2));
    gsm::vec2 uv = barycentric(bary, uv0, uv1, uv2);

    // Height map displacement
    if (heightTexId[patch_id / 3] > 0) {
      float h =
          CudaRasterizer::sampleTexture(heightTexId[patch_id / 3], static_cast<float2>(uv), textureOption).x;

      pos += norm * h * heightScale;
    }

    // clipPos
    gsm::vec3 p_view = gsm::vec3(gsm::mat4(viewmatrix) * gsm::vec4(pos, 1.0f));
    gsm::vec4 clipPos = gsm::mat4(projmatrix) * gsm::vec4(p_view, 1.0f);

    VertexOut &out = vertex_out[out_base + v];
    out.clipPos = clipPos;
    out.position = pos;
    out.normal = norm;
    out.tangent = tang;
    out.bitangent = bitang;
    out.uv = uv;
    out.depth = -p_view.z;
  }
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

__host__ __device__ inline gsm::vec3 reflect(const gsm::vec3 &I, const gsm::vec3 &N) {
  return I - N * 2.0f * gsm::dot(N, I);
}

__device__ gsm::vec3 calcDirLight(CudaRasterizer::Light light, const gsm::vec3 &normal,
                                  const gsm::vec3 &viewDir, float roughness) {

  gsm::vec3 lightDir = gsm::normalize(-static_cast<gsm::vec3>(light.direction));

  // ambient
  float amb = 0.1;

  // diffuse
  float diff = gsm::max(dot(normal, lightDir), 0.0);

  // roughness → shininess： roughness=0 is extremely shiny, roughness=1 is almost no specular
  float specularStrength = (1.0f - roughness) * (1.0f - roughness);
  float shininess = gsm::max(2.0f, (1.0f - roughness) * 128.0f);

  // specular
  gsm::vec3 reflectDir = reflect(-lightDir, normal);
  float spec = pow(gsm::max(dot(viewDir, reflectDir), 0.0), shininess) * specularStrength;

  // result
  return (amb + diff + spec) * static_cast<gsm::vec3>(light.color) * light.intensity;
}

// for each triangle
__global__ void render(const VertexOut *__restrict__ fs_in,                    // per-vertex
                       const cudaTextureObject_t *__restrict__ basecolorTexId, // per-face
                       const cudaTextureObject_t *__restrict__ normalTexId,    // per-face
                       const cudaTextureObject_t *__restrict__ roughnessTexId, // per-face
                       const cudaTextureObject_t *__restrict__ maskTexId,      // per-face
                       int num_triangles, int tessLevel, CudaRasterizer::TextureOption textureOption,
                       CudaRasterizer::Light lightDirection, const float *__restrict__ viewpos, int width,
                       int height,
                       uint32_t *__restrict__ depth_buffer, // temp uint depth buffer for atomicMin
                       CudaRasterizer::MaskCullingMode maskCullingMode,
                       CudaRasterizer::PixelMask *__restrict__ mask) {
  unsigned int prim_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (prim_id >= num_triangles)
    return;

  // triangle vertex id
  int i0 = static_cast<int>(prim_id) * 3 + 0;
  int i1 = static_cast<int>(prim_id) * 3 + 1;
  int i2 = static_cast<int>(prim_id) * 3 + 2;

  // extract VertexOut of each triangle vertex
  VertexOut v0 = fs_in[i0];
  VertexOut v1 = fs_in[i1];
  VertexOut v2 = fs_in[i2];

  // Perspective divide → NDC
  // (screenX, screenY, depth)
  gsm::vec3 ndc0 = gsm::vec3(v0.clipPos) / (v0.clipPos.w + FLT_EPSILON);
  gsm::vec3 ndc1 = gsm::vec3(v1.clipPos) / (v1.clipPos.w + FLT_EPSILON);
  gsm::vec3 ndc2 = gsm::vec3(v2.clipPos) / (v2.clipPos.w + FLT_EPSILON);

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
  int minX = gsm::max(0, static_cast<int>(gsm::min(gsm::min(s0.x, s1.x), s2.x)));
  int minY = gsm::max(0, static_cast<int>(gsm::min(gsm::min(s0.y, s1.y), s2.y)));
  int maxX = gsm::min(width - 1, gsm::ceil(gsm::max(gsm::max(s0.x, s1.x), s2.x)));
  int maxY = gsm::min(height - 1, gsm::ceil(gsm::max(gsm::max(s0.y, s1.y), s2.y)));

  float area = edge_function(s0, s1, s2);
  if (area == 0.0f) // 2 vertices are on the same line
    return;         // degenerate triangle

  if (maskCullingMode & CudaRasterizer::MaskCullingMode::NormalCulling) {
    // back face culling
    if (area <= 0.0f) {
      return;
    }
  }

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
      gsm::vec3 bary = {w0 / area, w1 / area, w2 / area};

      // interpolate depth from barycentric coordinates
      float depth = barycentric(bary, ndc0.z, ndc1.z, ndc2.z);

      // depth test: atomicMin ensures the nearest primitive wins
      uint32_t depth_uint = float_to_uint_depth(depth);
      uint32_t old_depth = atomicMin(&depth_buffer[pixel], depth_uint);

      // if the depth is closer (closer to the camera), update primitive ID
      // (possible race condition, but acceptable for selection use case)
      if (depth_uint < old_depth) {

        CudaRasterizer::PixelMask &m = mask[pixel];

        float2 uv = static_cast<float2>(barycentric(bary, v0.uv, v1.uv, v2.uv));
        float mask = sampleTexture(maskTexId[prim_id / (tessLevel * tessLevel)], uv, textureOption).x;
        if (mask < 0.5f)
          continue;

        // interpolate normal
        gsm::vec3 N = gsm::normalize(barycentric(bary, v0.normal, v1.normal, v2.normal));
        // interpolate world position
        gsm::vec3 worldPos = barycentric(bary, v0.position, v1.position, v2.position);
        gsm::vec3 viewDir = gsm::normalize(gsm::vec3(viewpos) - worldPos);

        if (normalTexId[prim_id / (tessLevel * tessLevel)] > 0) {

          // TBN matrix (all in world/view space)
          gsm::vec3 T = gsm::normalize(barycentric(bary, v0.tangent, v1.tangent, v2.tangent));
          // Gram-Schmidt, make sure T is orthogonal to new N
          T = gsm::normalize(T - gsm::dot(T, N) * N);
          gsm::vec3 B = gsm::cross(N, T);
          gsm::mat3 TBN = gsm::mat3(T, B, N);

          // normal map
          float4 mapN = CudaRasterizer::sampleTexture(normalTexId[prim_id / (tessLevel * tessLevel)], uv,
                                                      textureOption);
          gsm::vec3 mapN_ = gsm::vec3(mapN.x, mapN.y, mapN.z) * 2.0 - 1.0;
          N = normalize(TBN * mapN_);
        }

        // if (maskCullingMode & CudaRasterizer::MaskCullingMode::NormalCulling) {
        //   // back face culling
        //   if (gsm::dot(N, viewDir) <= 0.0f) {
        //     m.mask = 0;
        //     continue;
        //   }
        // }

        m.mask = 1;
        m.texCoords = uv;
        m.depth = barycentric(bary, v0.depth, v1.depth, v2.depth);
        m.normal = static_cast<float3>(N);

        if (basecolorTexId[prim_id / (tessLevel * tessLevel)] > 0) {
          float4 c = CudaRasterizer::sampleTexture(basecolorTexId[prim_id / (tessLevel * tessLevel)], uv,
                                                   textureOption);

          float roughness = 0.5f; // default
          if (roughnessTexId[prim_id / (tessLevel * tessLevel)] > 0) {
            roughness = CudaRasterizer::sampleTexture(roughnessTexId[prim_id / (tessLevel * tessLevel)], uv,
                                                      textureOption)
                            .x;
          }

          // lighting
          gsm::vec3 lightingResult =
              calcDirLight(lightDirection, N, viewDir, roughness) * gsm::vec3(c.x, c.y, c.z);

          m.color.x = lightingResult.x;
          m.color.y = lightingResult.y;
          m.color.z = lightingResult.z;
        }
      }
    }
  }
}

} // namespace

void CudaRasterizer::makeMask(const float *position, const float *normal, const float *texCoords,
                              const float *tangents, const float *bitangents, int num_vertices,
                              const cudaTextureObject_t *basecolorTexId,
                              const cudaTextureObject_t *normalTexId, const cudaTextureObject_t *heightTexId,
                              const cudaTextureObject_t *roughnessTexId, const cudaTextureObject_t *maskTexId,
                              TextureOption textureOption, float heightScale, Light lightDirection,
                              int num_triangles, int tessLevel, int width, int height,
                              const float *viewmatrix, const float *projmatrix, const float *viewpos,
                              MaskCullingMode maskCullingMode, PixelMask *mask) {

  if (position == nullptr || texCoords == nullptr || basecolorTexId == nullptr || normalTexId == nullptr ||
      heightTexId == nullptr)
    return;

  int num_fine_triangles = num_triangles * tessLevel * tessLevel;
  int num_fine_vertices = num_fine_triangles * 3;

  VertexOut *vertex_out;
  uint32_t *d_depth;

  size_t pixels = (size_t)width * height;

  cudaMalloc(&vertex_out, num_fine_vertices * sizeof(VertexOut));
  cudaMalloc(&d_depth, pixels * sizeof(uint32_t));

  // initialize depth buffer to max value
  cudaMemset(d_depth, static_cast<int>(0xFFFFFFFF), pixels * sizeof(uint32_t));
  // clear output buffers
  cudaMemset(mask, 0, width * height * sizeof(PixelMask));

  tessellate<<<(num_fine_triangles + 255) / 256, 256>>>(
      (gsm::vec3 *)position, (gsm::vec3 *)normal, (gsm::vec2 *)texCoords, (gsm::vec3 *)tangents,
      (gsm::vec3 *)bitangents, heightTexId, textureOption, num_vertices, num_triangles, heightScale,
      tessLevel, viewmatrix, projmatrix, vertex_out);

  // ── Rasterize + Fragment Shader ──
  render<<<(num_fine_triangles + 255) / 256, 256>>>(
      vertex_out, basecolorTexId, normalTexId, roughnessTexId, maskTexId, num_fine_triangles, tessLevel,
      textureOption, lightDirection, viewpos, width, height, d_depth, maskCullingMode, mask);

  cudaFree(vertex_out);
  cudaFree(d_depth);
}
