#version 430 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv0;       // material UV (from glTF)
layout(location = 3) in vec2 uvDecal;   // decal UV (parameterization result / model UV)
layout(location = 4) in int  sl_in;     // selection marker: >= 0 where the decal applies
layout(location = 5) in vec3 matTangent;
layout(location = 6) in vec3 matBitangent;
layout(location = 7) in vec3 decalTangent;
layout(location = 8) in vec3 decalBitangent;

uniform mat4 projection_matrix;
uniform mat4 view_matrix;
uniform mat4 model_matrix;

out VS_OUT {
  vec3 worldPos;
  vec3 normal;
  vec2 uv0;
  vec2 uvDecal;
  vec3 matT;   // world-space, not normalized (may be zero if no tangents)
  vec3 matB;
  vec3 decalT;
  vec3 decalB;
} vs_out;

flat out int sl;

void main() {
  vec4 wp = model_matrix * vec4(position, 1.0);
  mat3 nm = transpose(inverse(mat3(model_matrix)));

  vs_out.worldPos = wp.xyz;
  vs_out.normal   = normalize(nm * normal);
  vs_out.uv0      = uv0;
  vs_out.uvDecal  = uvDecal;
  vs_out.matT     = nm * matTangent;
  vs_out.matB     = nm * matBitangent;
  vs_out.decalT   = nm * decalTangent;
  vs_out.decalB   = nm * decalBitangent;
  sl = sl_in;

  gl_Position = projection_matrix * view_matrix * wp;
}
