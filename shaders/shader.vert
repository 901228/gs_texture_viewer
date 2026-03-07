#version 430 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 textureCoord;
layout (location = 3) in int sl_in;
layout (location = 4) in vec3 tangent;
layout (location = 5) in vec3 bitangent;

out Vs_out {
  vec3 position;
  vec3 normal;
  vec2 textureCoord;
  int sl;
  vec3 tangent;
  vec3 bitangent;
} vs_out;

void main() {

  // NOT to do any matrix transformations! TES does it
  vs_out.position    = position;
  vs_out.normal      = normal;
  vs_out.textureCoord = textureCoord;
  vs_out.sl          = sl_in;
  vs_out.tangent     = tangent;
  vs_out.bitangent   = bitangent;
  gl_Position = vec4(position, 1.0);
}
