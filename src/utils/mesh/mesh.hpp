
#ifndef MESH_HPP
#define MESH_HPP
#pragma once

#define _USE_MATH_DEFINES
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>

struct MyTraits : OpenMesh::DefaultTraits {

  // let Point and Normal be a vector made from doubles
  typedef OpenMesh::Vec3f Point;
  typedef OpenMesh::Vec3f Normal;

  // add normal property to vertices and faces
  VertexAttributes(OpenMesh::Attributes::Normal | OpenMesh::Attributes::TexCoord2D);
};
typedef OpenMesh::PolyMesh_ArrayKernelT<MyTraits> MyMesh;

#endif // !MESH_HPP
