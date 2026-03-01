#ifndef SOLVE_UV_HPP
#define SOLVE_UV_HPP
#pragma once

#include <unordered_set>

#include "mesh.hpp"

namespace SolveUV {

enum class SolvingMode : int { Harmonics, ExpMap };

void SolveHarmonics(const std::unordered_set<unsigned int> &selectedID, float uvRotateAngle,
                    MyMesh &originMesh);

void SolveExpMap(const std::unordered_set<unsigned int> &selectedID, float uvRotateAngle, MyMesh &originMesh);

inline void Solve(const SolvingMode &mode, const std::unordered_set<unsigned int> &selectedID,
                  float uvRotateAngle, MyMesh &originMesh) {
  switch (mode) {
  case SolvingMode::Harmonics:
    SolveHarmonics(selectedID, uvRotateAngle, originMesh);
    break;
  case SolvingMode::ExpMap:
    SolveExpMap(selectedID, uvRotateAngle, originMesh);
    break;
  default:
    throw std::runtime_error("Unknown solving mode!");
  }
}

} // namespace SolveUV

#endif // !SOLVE_UV_HPP
