#ifndef EXPMAP_HPP
#define EXPMAP_HPP
#pragma once

#include <unordered_set>

#include "mesh.hpp"

namespace ExpMap {

void Solve(const std::unordered_set<unsigned int> &selectedID, MyMesh &originMesh);

} // namespace ExpMap

#endif // !EXPMAP_HPP
