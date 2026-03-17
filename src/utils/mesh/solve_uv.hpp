#ifndef SOLVE_UV_HPP
#define SOLVE_UV_HPP
#pragma once

#include <unordered_set>

#include "hit_test.hpp"
#include "mesh.hpp"

#include "expmap.hpp"
#include "geodesic_splines.hpp"
#include "harmonics.hpp"

namespace SolveUV {

enum class SolvingMode : int { Harmonics, ExpMap, GeodesicSplines };

inline void Solve(const SolvingMode &mode, const std::unordered_set<unsigned int> &selectedID,
                  MyMesh &originMesh, const BVH::BVH &bvh) {
  switch (mode) {
  case SolvingMode::Harmonics:
    Harmonics::Solve(selectedID, originMesh);
    break;
  case SolvingMode::ExpMap:
    ExpMap::Solve(selectedID, originMesh);
    break;
  case SolvingMode::GeodesicSplines:
    GeodesicSplines::Solve(selectedID, originMesh, bvh);
    break;
  default:
    throw std::runtime_error("Unknown solving mode!");
  }
}

} // namespace SolveUV

#endif // !SOLVE_UV_HPP
