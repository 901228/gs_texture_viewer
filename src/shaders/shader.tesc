#version 430 core

layout(vertices = 3) out;

in Vs_out {
  vec3 position;
  vec3 normal;
  vec2 textureCoord;
  int sl;
  vec3 tangent;
  vec3 bitangent;
} tcs_in[];

out Tcs_out {
  vec3 position;
  vec3 normal;
  vec2 textureCoord;
  int sl;
  vec3 tangent;
  vec3 bitangent;
} tcs_out[];

uniform float tessLevel;

void main() {
  // pass vertex data directly
  tcs_out[gl_InvocationID].position     = tcs_in[gl_InvocationID].position;
  tcs_out[gl_InvocationID].normal       = tcs_in[gl_InvocationID].normal;
  tcs_out[gl_InvocationID].textureCoord = tcs_in[gl_InvocationID].textureCoord;
  tcs_out[gl_InvocationID].sl           = tcs_in[gl_InvocationID].sl;
  tcs_out[gl_InvocationID].tangent      = tcs_in[gl_InvocationID].tangent;
  tcs_out[gl_InvocationID].bitangent    = tcs_in[gl_InvocationID].bitangent;

  // only set once in invocation 0
  if (gl_InvocationID == 0) {
    gl_TessLevelInner[0] = tessLevel;
    gl_TessLevelOuter[0] = tessLevel;
    gl_TessLevelOuter[1] = tessLevel;
    gl_TessLevelOuter[2] = tessLevel;
  }
}
