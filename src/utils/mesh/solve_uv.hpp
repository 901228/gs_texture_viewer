#ifndef SOLVE_UV_HPP
#define SOLVE_UV_HPP
#pragma once

#include <unordered_set>

#include "../utils.hpp"
#include "hit_test.hpp"
#include "mesh.hpp"

#include "expmap.hpp"
#include "geodesic_splines.hpp"
#include "harmonics.hpp"

namespace SolveUV {

enum class SolvingMode : int { Harmonics, ExpMap, GeodesicSplines };

inline void Solve(const SolvingMode &mode, const std::unordered_set<unsigned int> &selectedID,
                  MyMesh &originMesh, const BVH::BVH &bvh, HitResult hitResult = {}) {
  switch (mode) {
  case SolvingMode::Harmonics:
    Harmonics::Solve(selectedID, originMesh);
    break;
  case SolvingMode::ExpMap:
    ExpMap::Solve(selectedID, originMesh);
    break;
  case SolvingMode::GeodesicSplines:
    GeodesicSplines::Solve(selectedID, originMesh, bvh, hitResult);
    break;
  default:
    throw std::runtime_error("Unknown solving mode!");
  }
}

inline void calculateTB(MyMesh &mesh) {

  std::vector<glm::vec3> vtangent(mesh.n_vertices(), glm::vec3(0));
  std::vector<glm::vec3> vbitangent(mesh.n_vertices(), glm::vec3(0));

  for (auto fh : mesh.faces()) {

    auto fv = mesh.cfv_iter(fh);
    auto vh0 = *fv;
    ++fv;
    auto vh1 = *fv;
    ++fv;
    auto vh2 = *fv;

    glm::vec3 p0 = Utils::toGlm(mesh.point(vh0));
    glm::vec3 p1 = Utils::toGlm(mesh.point(vh1));
    glm::vec3 p2 = Utils::toGlm(mesh.point(vh2));

    auto tc0 = mesh.texcoord2D(vh0);
    auto tc1 = mesh.texcoord2D(vh1);
    auto tc2 = mesh.texcoord2D(vh2);

    // skip face without UV
    if (tc0[0] < 0 || tc1[0] < 0 || tc2[0] < 0)
      continue;

    glm::vec2 uv0(tc0[0], tc0[1]);
    glm::vec2 uv1(tc1[0], tc1[1]);
    glm::vec2 uv2(tc2[0], tc2[1]);

    glm::vec3 dp1 = p1 - p0, dp2 = p2 - p0;
    glm::vec2 duv1 = uv1 - uv0, duv2 = uv2 - uv0;

    float denom = duv1.x * duv2.y - duv2.x * duv1.y;
    if (std::abs(denom) < 1e-8f)
      continue;

    float inv = 1.f / denom;
    glm::vec3 T = inv * (duv2.y * dp1 - duv1.y * dp2);
    glm::vec3 B = inv * (-duv2.x * dp1 + duv1.x * dp2);

    // Accumulate (area-weighted 可以更精確，但這樣已夠)
    vtangent[vh0.idx()] += T;
    vtangent[vh1.idx()] += T;
    vtangent[vh2.idx()] += T;
    vbitangent[vh0.idx()] += B;
    vbitangent[vh1.idx()] += B;
    vbitangent[vh2.idx()] += B;
  }

  // notmalzie and write to mesh
  for (auto vh : mesh.vertices()) {
    glm::vec3 T = vtangent[vh.idx()];
    glm::vec3 B = vbitangent[vh.idx()];
    if (glm::length(T) < 1e-8f || glm::length(B) < 1e-8f)
      continue;

    glm::vec3 n = Utils::toGlm(mesh.normal(vh));

    // Gram-Schmidt orthogonalize against normal
    T = glm::normalize(T - glm::dot(T, n) * n);
    B = glm::normalize(B - glm::dot(B, n) * n - glm::dot(B, T) * T);

    mesh.data(vh).tangent = {T.x, T.y, T.z};
    mesh.data(vh).bitangent = {B.x, B.y, B.z};
  }
}

} // namespace SolveUV

#endif // !SOLVE_UV_HPP
