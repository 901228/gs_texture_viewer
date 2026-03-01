#include "solve_uv.hpp"

#include <numbers>
#include <queue>

#include <Eigen/Sparse>
#include <glad/gl.h>

#include "expmap_helper.hpp"

#define Quad

void CopySelectFace(const std::unordered_set<unsigned int> &selectedID, const MyMesh &originMesh,
                    MyMesh &mesh) {

  mesh.request_vertex_normals();
  mesh.request_face_normals();

  std::vector<MyMesh::VertexHandle> vhs;
  vhs.reserve(3);

  std::map<int, int> usedVertices;

  for (unsigned int id : selectedID) {

    MyMesh::FaceHandle fh = originMesh.face_handle(id);
    for (MyMesh::VertexHandle vh_it : originMesh.fv_range(fh)) {

      MyMesh::VertexHandle vh;
      MyMesh::Point p = originMesh.point(vh_it);

      if (usedVertices.find(vh_it.idx()) == usedVertices.end()) {

        vh = mesh.add_vertex(p); // NOLINT(cppcoreguidelines-slicing)
        usedVertices[vh_it.idx()] = vh.idx();
      } else
        vh = mesh.vertex_handle(usedVertices[vh_it.idx()]);

      vhs.push_back(vh);
    }

    mesh.add_face(vhs);
    vhs.clear();
  }

  mesh.update_normals();
}

namespace SolveUV {

void SolveHarmonics(const std::unordered_set<unsigned int> &selectedID, float uvRotateAngle,
                    MyMesh &originMesh) {
  if (selectedID.empty())
    return;

  OpenMesh::HPropHandleT<double> heWeight;
  OpenMesh::VPropHandleT<int> row;
  MyMesh _mesh;
  _mesh.add_property(heWeight, "heWeight");
  _mesh.add_property(row, "row");
  _mesh.request_vertex_texcoords2D();

  CopySelectFace(selectedID, originMesh, _mesh);

  // calculate weight
  MyMesh::HalfedgeHandle heh;
  for (MyMesh::EdgeHandle eh : _mesh.edges()) {

    if (_mesh.is_boundary(eh)) {

      if (!heh.is_valid())
        heh = _mesh.halfedge_handle(eh, 1);
      continue;
    }

    MyMesh::HalfedgeHandle _heh = _mesh.halfedge_handle(eh, 0);
    MyMesh::Point pFrom = _mesh.point(_mesh.from_vertex_handle(_heh));
    MyMesh::Point pTo = _mesh.point(_mesh.to_vertex_handle(_heh));
    MyMesh::Point p1 = _mesh.point(_mesh.to_vertex_handle(_mesh.next_halfedge_handle(_heh)));
    MyMesh::Point p2 =
        _mesh.point(_mesh.to_vertex_handle(_mesh.next_halfedge_handle(_mesh.opposite_halfedge_handle(_heh))));

    double edgeLen = (pFrom - pTo).length();

    OpenMesh::Vec3d v1 = static_cast<OpenMesh::Vec3d>(pTo - pFrom);
    v1.normalize();

    // weight from to
    {
      OpenMesh::Vec3d v2 = static_cast<OpenMesh::Vec3d>(p1 - pFrom);
      v2.normalize();

      double angle1 = std::acos(OpenMesh::dot(v1, v2));

      v2 = (OpenMesh::Vec3d)(p2 - pFrom);
      v2.normalize();

      double angle2 = std::acos(OpenMesh::dot(v1, v2));

      _mesh.property(heWeight, _heh) = ((std::tan(angle1 / 2.0) + std::tan(angle2 / 2.0)) / edgeLen);
    }

    // weight to from
    {
      v1 = -v1;

      OpenMesh::Vec3d v2 = static_cast<OpenMesh::Vec3d>(p1 - pTo);
      v2.normalize();

      double angle1 = std::acos(OpenMesh::dot(v1, v2));

      v2 = (OpenMesh::Vec3d)(p2 - pTo);
      v2.normalize();

      double angle2 = std::acos(OpenMesh::dot(v1, v2));

      _mesh.property(heWeight, _heh) = ((std::tan(angle1 / 2.0) + std::tan(angle2 / 2.0)) / edgeLen);
    }
  }

  // calculate matrix size
  int count = 0;
  for (MyMesh::VertexHandle vh : _mesh.vertices()) {

    if (_mesh.is_boundary(vh))
      _mesh.property(row, vh) = -1;
    else
      _mesh.property(row, vh) = count++;
  }

  // calculate perimeter
  double perimeter = 0;
  std::vector<double> segLength;
  std::vector<MyMesh::VertexHandle> vhs;
  MyMesh::HalfedgeHandle hehNext = heh;
  do {

    MyMesh::Point from = _mesh.point(_mesh.from_vertex_handle(hehNext));
    MyMesh::Point to = _mesh.point(_mesh.to_vertex_handle(hehNext));
    perimeter += (from - to).length();

    segLength.push_back(perimeter);
    vhs.push_back(_mesh.from_vertex_handle(hehNext));

    hehNext = _mesh.next_halfedge_handle(hehNext);
  } while (heh != hehNext);

  // put boundry texture coords into mesh
  {
#ifdef Quad
    float rd = (225.0f + uvRotateAngle) * std::numbers::pi_v<float> / 180.0f;
    float initDist = 0;
    MyMesh::TexCoord2D st(0, 0);
    float R = std::sqrtf(2) / 2.0f;
    st[0] = R * cosf(rd) + 0.5f;
    st[1] = R * sinf(rd) + 0.5f;

    if (st[0] > 1) {
      st[0] = 1;
      st[1] = tanf(rd) / 2 + 0.5f;
    }

    if (st[0] < 0) {
      st[0] = 0;
      st[1] = 0.5f - tanf(rd) / 2;
    }

    if (st[1] > 1) {
      st[0] = tanf(std::numbers::pi_v<float> / 2.0f - rd) / 2 + 0.5f;
      st[1] = 1;
    }

    if (st[1] < 0) {
      st[0] = 0.5f - tanf(std::numbers::pi_v<float> / 2.0f - rd) / 2;
      st[1] = 0;
    }

    if (uvRotateAngle <= 90) {
      initDist = st.length();
    }

    else if (uvRotateAngle <= 180) {
      initDist = 1 + st[1];
    }

    else if (uvRotateAngle <= 270) {
      initDist = 3 - st[0];
    }

    else {
      initDist = 4 - st[1];
    }

    _mesh.set_texcoord2D(vhs[0], st);
    perimeter /= 4.0;
    for (int i = 1; i < vhs.size(); ++i) {
      double curLen = segLength[i - 1] / perimeter + initDist;
      if (curLen > 4) {
        curLen -= 4;
      }

      if (curLen <= 1) {
        st[0] = static_cast<float>(curLen);
        st[1] = 0;
      } else if (curLen <= 2) {
        st[0] = 1;
        st[1] = static_cast<float>(curLen) - 1;
      } else if (curLen <= 3) {
        st[0] = 3 - static_cast<float>(curLen);
        st[1] = 1;
      } else {
        st[0] = 0;
        st[1] = 4 - static_cast<float>(curLen);
      }

      _mesh.set_texcoord2D(vhs[i], st);
    }
#else
    MyMesh::TexCoord2D st(1.0f, 0.5f);
    _mesh.set_texcoord2D(vhs[0], st);

    for (int i = 1; i < vhs.size(); ++i) {
      float angle = 2.0f * std::numbers::pi_v<float> * static_cast<float>(segLength[i - 1]) /
                    static_cast<float>(perimeter);

      st[0] = (std::cosf(angle) + 1) / 2;
      st[1] = (std::sinf(angle) + 1) / 2;

      _mesh.set_texcoord2D(vhs[i], st);
    }
#endif
  }

  // cauculate inner points
  {
    Eigen::SparseMatrix<double> A(count, count);
    Eigen::VectorXd BX(count);
    BX.setZero();
    Eigen::VectorXd BY(count);
    BY.setZero();
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> linearSolver;

    // fill matrix
    for (MyMesh::VertexHandle vh_i : _mesh.vertices()) {

      if (_mesh.is_boundary(vh_i))
        continue;

      int i = _mesh.property(row, vh_i);
      double totalWeight = 0;

      for (MyMesh::VertexHandle vh_j : _mesh.vv_range(vh_i)) {

        // NOLINTNEXTLINE(cppcoreguidelines-slicing)
        double w = _mesh.property(heWeight, _mesh.find_halfedge(vh_i, vh_j));

        if (_mesh.is_boundary(vh_j)) {

          MyMesh::TexCoord2D texCoord = _mesh.texcoord2D(vh_j);
          BX[i] += w * texCoord[0];
          BY[i] += w * texCoord[1];
        } else {

          int j = _mesh.property(row, vh_j);
          A.insert(i, j) = -w;
        }

        totalWeight += w;
      }

      A.insert(i, i) = totalWeight;
    }

    // solve the linear system
    Eigen::VectorXd TX;
    Eigen::VectorXd TY;
    {
      A.makeCompressed();

      Eigen::SparseMatrix<double> At = A.transpose();
      linearSolver.compute(At * A);

      TX = linearSolver.solve(At * BX);
      TY = linearSolver.solve(At * BY);
    }

    // put the texture coords in to _mesh
    for (MyMesh::VertexHandle vh : _mesh.vertices()) {

      if (_mesh.is_boundary(vh))
        continue;

      int i = _mesh.property(row, vh);
      _mesh.set_texcoord2D(vh, {TX[i], TY[i]});
    }
  }

  // put the texture coords back to Mesh
  {
    // check whether Mesh has vertex texcoord2D
    if (!originMesh.has_vertex_texcoords2D()) {

      originMesh.request_vertex_texcoords2D();
      for (MyMesh::VertexHandle vh : originMesh.vertices())
        originMesh.set_texcoord2D(vh, {-1, -1});
    }

    // check whether Mesh has face texture index
    if (!originMesh.has_face_texture_index()) {

      originMesh.request_face_texture_index();
      for (MyMesh::FaceHandle fh : originMesh.faces())
        originMesh.set_texture_index(fh, -1);
    }

    // map texcoord back to Mesh
    auto s_it = selectedID.begin();
    for (MyMesh::FaceHandle fh : _mesh.faces()) {

      MyMesh::FaceHandle sfh = originMesh.face_handle(*s_it);
      s_it++;

      MyMesh::FaceVertexIter fv_it = _mesh.fv_iter(fh);
      MyMesh::FaceVertexIter sfv_it = originMesh.fv_iter(sfh);
      for (; (fv_it.is_valid() && sfv_it.is_valid()); ++fv_it, ++sfv_it) {

        MyMesh::TexCoord2D texCoord = _mesh.texcoord2D(*fv_it);
        originMesh.set_texcoord2D(*sfv_it, texCoord);
      }
    }
  }
}

void SolveExpMap(const std::unordered_set<unsigned int> &selectedID, float uvRotateAngle,
                 MyMesh &originMesh) {

  if (selectedID.empty())
    return;

  // Auto-compute center from selection
  int centerVertex = ExpHelper::computeCenterFromFaces(selectedID, originMesh);
  if (centerVertex < 0)
    return;

  auto seedVh = originMesh.vertex_handle(centerVertex);
  if (!seedVh.is_valid()) {
    return;
  }

  // Build set of vertices from selected faces
  std::unordered_set<int> selectedVertices;
  for (unsigned int f : selectedID) {
    MyMesh::FaceHandle fh = originMesh.face_handle(f);
    for (const MyMesh::VertexHandle &fv : originMesh.fv_range(fh)) {
      selectedVertices.insert(fv.idx());
    }
  }

  // Radius will be auto-computed from selection extent after Dijkstra
  float radius = 0.0f;

  // Initialize per-vertex data
  size_t numVerts = originMesh.n_vertices();
  std::vector<ExpHelper::ExpMapVertex> vertices(numVerts);

  for (MyMesh::VertexHandle vh : originMesh.vertices()) {
    vertices[vh.idx()].vh = vh;
    vertices[vh.idx()].state = ExpHelper::ExpMapVertex::State::INACTIVE;
    vertices[vh.idx()].distance = std::numeric_limits<float>::max();
  }

  // Initialize seed vertex
  int seedIdx = seedVh.idx();
  ExpHelper::ExpMapVertex &seed = vertices[seedIdx];
  seed.state = ExpHelper::ExpMapVertex::State::FROZEN;
  seed.distance = 0.0f;
  seed.surfaceVector = glm::vec2(0, 0);

  glm::vec3 seedPos = Utils::toGlm(originMesh.point(seedVh));
  glm::vec3 seedNormal = Utils::toGlm(originMesh.normal(seedVh));
  seed.frame = ExpHelper::TangentFrame(seedPos, seedNormal);

  // Apply rotation to seed frame around Z-axis (normal)
  if (uvRotateAngle != 0.0f) {
    float angleRadians = glm::radians(uvRotateAngle);
    float cosTheta = std::cos(angleRadians);
    float sinTheta = std::sin(angleRadians);

    glm::vec3 oldX = seed.frame.axes[0];
    glm::vec3 oldY = seed.frame.axes[1];
    glm::vec3 oldZ = seed.frame.axes[2]; // Normal, unchanged

    // Rotate X and Y around Z
    glm::vec3 newX = cosTheta * oldX + sinTheta * oldY;
    glm::vec3 newY = -sinTheta * oldX + cosTheta * oldY;

    seed.frame.axes = glm::mat3(newX, newY, oldZ);
  }

  // Priority queue (min-heap by distance)
  auto cmp = [&vertices](int a, int b) { return vertices[a].distance > vertices[b].distance; };
  std::priority_queue<int, std::vector<int>, decltype(cmp)> pq(cmp);

  // Add seed neighbors to queue (only if in selected vertices)
  for (const MyMesh::HalfedgeHandle &voh : originMesh.voh_range(seedVh)) {
    auto nvh = originMesh.to_vertex_handle(voh);
    if (!selectedVertices.count(nvh.idx()))
      continue; // Skip non-selected
    auto &nv = vertices[nvh.idx()];
    nv.distance = glm::length(Utils::toGlm(originMesh.point(nvh)) - seedPos);
    nv.nearest = seedVh;
    nv.state = ExpHelper::ExpMapVertex::State::ACTIVE;
    pq.push(nvh.idx());
  }

  // Dijkstra propagation
  while (!pq.empty()) {
    int curIdx = pq.top();
    pq.pop();

    auto &cur = vertices[curIdx];
    if (cur.state == ExpHelper::ExpMapVertex::State::FROZEN) {
      continue; // Already processed
    }
    cur.state = ExpHelper::ExpMapVertex::State::FROZEN;

    // Propagate frame and compute surface vector
    propagateFrame(vertices, cur, seedIdx, originMesh);

    // Track max distance for auto-radius computation
    radius = std::max(radius, cur.distance);

    // Update neighbors (only within selected vertices)
    auto curVh = cur.vh;
    glm::vec3 curPos = Utils::toGlm(originMesh.point(curVh));

    for (const MyMesh::HalfedgeHandle &voh : originMesh.voh_range(curVh)) {
      auto nvh = originMesh.to_vertex_handle(voh);
      if (!selectedVertices.count(nvh.idx()))
        continue; // Skip non-selected

      auto &nv = vertices[nvh.idx()];

      if (nv.state == ExpHelper::ExpMapVertex::State::FROZEN) {
        continue;
      }

      float edgeDist = glm::length(Utils::toGlm(originMesh.point(nvh)) - curPos);
      float newDist = cur.distance + edgeDist;

      if (newDist < nv.distance) {
        nv.distance = newDist;
        nv.nearest = curVh;
        nv.state = ExpHelper::ExpMapVertex::State::ACTIVE;
        pq.push(nvh.idx());
      }
    }
  }

  // Copy UVs to mesh with normalization
  // Ensure radius is valid (avoid division by zero)
  if (radius < 1e-6f) {
    radius = 1.0f;
  }

  // Map radius to [0,1] using inscribed square: scale = 1/(radius * sqrt(2))
  float uvScale = 1.0f / (radius * std::sqrt(2.0f));

  // put the texture coords back to Mesh
  {
    // check whether Mesh has vertex texcoord2D
    if (!originMesh.has_vertex_texcoords2D()) {

      originMesh.request_vertex_texcoords2D();
      for (MyMesh::VertexHandle vh : originMesh.vertices())
        originMesh.set_texcoord2D(vh, {-1, -1});
    }

    // check whether Mesh has face texture index
    if (!originMesh.has_face_texture_index()) {

      originMesh.request_face_texture_index();
      for (MyMesh::FaceHandle fh : originMesh.faces())
        originMesh.set_texture_index(fh, -1);
    }

    for (auto vh : originMesh.vertices()) {
      auto &v = vertices[vh.idx()];
      if (v.state == ExpHelper::ExpMapVertex::State::FROZEN) {
        // Transform geodesic coords to [0,1] UV space
        glm::vec2 uv = v.surfaceVector * uvScale + 0.5f;
        originMesh.set_texcoord2D(vh, {uv.x, uv.y});
      }
    }
  }
}

} // namespace SolveUV
