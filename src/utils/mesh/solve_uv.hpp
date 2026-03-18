#ifndef SOLVE_UV_HPP
#define SOLVE_UV_HPP
#pragma once

#include <unordered_set>

#include "hit_test.hpp"
#include "mesh.hpp"

class Model;

namespace SolveUV {

enum class SolvingMode : int { Harmonics, ExpMap, GeodesicSplines };

void Solve(const SolvingMode &mode, const std::unordered_set<unsigned int> &selectedID, Model &model,
           HitResult hitResult = {});

void calculateTB(MyMesh &mesh);

} // namespace SolveUV

#endif // !SOLVE_UV_HPP
