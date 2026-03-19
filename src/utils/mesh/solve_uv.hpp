#ifndef SOLVE_UV_HPP
#define SOLVE_UV_HPP
#pragma once

#include <optional>
#include <unordered_set>

#include <glm/glm.hpp>

#include "geodesic_splines.hpp"
#include "mesh.hpp"

class Model;

namespace SolveUV {

enum class SolvingMode : int { Harmonics, ExpMap, GeodesicSplines };

std::pair<LogarithmicMap::LogMapTable, float> SolveGeodesic(glm::vec3 hitPoint,
                                                            GeodesicSplines::Implicit &model);

void Solve(const SolvingMode &mode, const std::unordered_set<unsigned int> &selectedID, Model &model,
           std::optional<glm::vec3> hitPoint);

void calculateTB(MyMesh &mesh);

} // namespace SolveUV

#endif // !SOLVE_UV_HPP
