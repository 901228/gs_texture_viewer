#ifndef HARMONICS_HPP
#define HARMONICS_HPP
#pragma once

#include <unordered_set>

#include "mesh.hpp"

namespace Harmonics {

void Solve(const std::unordered_set<unsigned int> &selectedID, MyMesh &originMesh, bool isQuad = true);

} // namespace Harmonics

#endif // !HARMONICS_HPP
