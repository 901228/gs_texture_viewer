#include "isosurface.hpp"

#include <glad/gl.h>

#include <MarchingCubeCpp/MC.h>

#include "../camera/camera.hpp"
#include "../gl/program.hpp"
#include "glm/gtc/type_ptr.hpp"

namespace Isosurface {

MarchingCubesResult extractIsosurface(GeodesicSplines::Implicit &model, glm::vec3 bmin, glm::vec3 bmax,
                                      int resolution) {
  MarchingCubesResult result;

  int N = resolution + 1;
  glm::vec3 step = (bmax - bmin) / float(resolution);

  // =========================================
  // Step 1: Fill scalar field (grid space 0..N)
  // =========================================
  MC::MC_FLOAT *field = new MC::MC_FLOAT[N * N * N];

  for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        glm::vec3 worldPos = bmin + glm::vec3(x, y, z) * step;
        field[x + N * (y + N * z)] = model.eval(worldPos); // zero-isosurface 剛好對應 σ = threshold
      }

  // =========================================
  // Step 2: Marching Cubes
  // =========================================
  MC::mcMesh mesh;
  MC::marching_cube(field, N, N, N, mesh);
  delete[] field;

  // Step 3: grid space → world space，並用 grad() 替換法線
  result.vertices.reserve(mesh.vertices.size());
  result.normals.reserve(mesh.vertices.size());
  result.indices = mesh.indices; // index buffer 直接沿用

  for (const auto &v : mesh.vertices) {
    // grid space (0..N) → world space
    glm::vec3 gridPos(v.x, v.y, v.z);
    glm::vec3 worldPos = bmin + gridPos * step;
    result.vertices.push_back(worldPos);

    result.normals.push_back(model.normal(worldPos));
  }

  return result;
}

static const char *isosurfaceVertexShader = R"(
  #version 450 core
  layout(location = 0) in vec3 aPos;
  layout(location = 1) in vec3 aNormal;

  uniform mat4 uMVP;
  uniform mat4 uModel;

  out vec3 vNormal;
  out vec3 vWorldPos;

  void main() {
    vWorldPos = vec3(uModel * vec4(aPos, 1.0));
    vNormal   = mat3(transpose(inverse(uModel))) * aNormal;
    gl_Position = uMVP * vec4(aPos, 1.0);
  }
)";

static const char *isosurfaceFragmentShader = R"(
  #version 450 core
  in vec3 vNormal;
  in vec3 vWorldPos;

  uniform vec3 uLightDir;
  uniform vec3 uColor;     // e.g. vec3(0.4, 0.8, 0.6)
  uniform float uAlpha;    // 半透明用

  out vec4 fragColor;

  void main() {
    vec3 N = normalize(vNormal);
    float diff = max(dot(N, normalize(-uLightDir)), 0.0);
    vec3 col = uColor * (0.2 + 0.8 * diff);
    fragColor = vec4(col, uAlpha);
  }
)";

IsosurfaceRenderer::IsosurfaceRenderer()
    : program(std::make_unique<Program>(isosurfaceVertexShader, isosurfaceFragmentShader)) {}
IsosurfaceRenderer::~IsosurfaceRenderer() {}

void IsosurfaceRenderer::upload(const MarchingCubesResult &mc) {

  // 交錯排列 pos + normal
  std::vector<float> vdata;
  vdata.reserve(mc.vertices.size() * 6);
  for (int i = 0; i < mc.vertices.size(); ++i) {
    vdata.push_back(mc.vertices[i].x);
    vdata.push_back(mc.vertices[i].y);
    vdata.push_back(mc.vertices[i].z);
    vdata.push_back(mc.normals[i].x);
    vdata.push_back(mc.normals[i].y);
    vdata.push_back(mc.normals[i].z);
  }

  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo);
  glGenBuffers(1, &ebo);

  glBindVertexArray(vao);

  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, vdata.size() * sizeof(float), vdata.data(), GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, mc.indices.size() * sizeof(uint32_t), mc.indices.data(),
               GL_STATIC_DRAW);

  // position
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
  // normal
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));

  glBindVertexArray(0);
  indexCount = mc.indices.size();
}

void IsosurfaceRenderer::render(const Camera &camera, const glm::vec3 &lightDir) {

  program->use();

  program->setVec3("uLightDir", glm::value_ptr(lightDir));
  glm::vec3 uColor{1.0f, 0.5f, 0.2f};
  program->setVec3("uColor", glm::value_ptr(uColor));
  program->setFloat("uAlpha", 1.0);

  glm::mat4 model = glm::identity<glm::mat4>();
  glm::mat4 mvp = camera.projectionMatrix() * camera.viewMatrix() * model;
  program->setMat4("uMVP", glm::value_ptr(mvp));
  program->setMat4("uModel", glm::value_ptr(model));

  glBindVertexArray(vao);
  // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

  glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);

  // glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glBindVertexArray(0);

  program->unUse();
}

} // namespace Isosurface
