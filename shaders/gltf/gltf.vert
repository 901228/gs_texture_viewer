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

uniform mat4 model_matrix;

// World-space attributes are passed straight through; the tessellation evaluation
// shader does the view/projection transform (and optional height displacement).
out VtxData {
  vec3 worldPos;
  vec3 normal;
  vec2 uv0;
  vec2 uvDecal;
  vec3 matT;
  vec3 matB;
  vec3 decalT;
  vec3 decalB;
  int  sl;
} v;

void main() {
  vec4 wp = model_matrix * vec4(position, 1.0);
  mat3 nm = transpose(inverse(mat3(model_matrix)));

  v.worldPos = wp.xyz;
  v.normal   = normalize(nm * normal);
  v.uv0      = uv0;
  v.uvDecal  = uvDecal;
  v.matT     = nm * matTangent;
  v.matB     = nm * matBitangent;
  v.decalT   = nm * decalTangent;
  v.decalB   = nm * decalBitangent;
  v.sl       = sl_in;

  gl_Position = wp; // overridden by the tessellation evaluation shader
}
