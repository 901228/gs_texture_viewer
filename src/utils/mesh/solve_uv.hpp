#ifndef SOLVE_UV_HPP
#define SOLVE_UV_HPP
#pragma once

#include <unordered_set>

#include "hit_test.hpp"
#include "mesh.hpp"

namespace SolveUV {

enum class SolvingMode : int { Harmonics, ExpMap, GeodesicSplines };

void SolveHarmonics(const std::unordered_set<unsigned int> &selectedID, MyMesh &originMesh);

void SolveExpMap(const std::unordered_set<unsigned int> &selectedID, MyMesh &originMesh);

void SolveGeodesicSplines(const std::unordered_set<unsigned int> &selectedID, MyMesh &originMesh,
                          const BVH::BVH &bvh);

inline void Solve(const SolvingMode &mode, const std::unordered_set<unsigned int> &selectedID,
                  MyMesh &originMesh, const BVH::BVH &bvh) {
  switch (mode) {
  case SolvingMode::Harmonics:
    SolveHarmonics(selectedID, originMesh);
    break;
  case SolvingMode::ExpMap:
    SolveExpMap(selectedID, originMesh);
    break;
  case SolvingMode::GeodesicSplines:
    SolveGeodesicSplines(selectedID, originMesh, bvh);
    break;
  default:
    throw std::runtime_error("Unknown solving mode!");
  }
}

} // namespace SolveUV

#endif // !SOLVE_UV_HPP
