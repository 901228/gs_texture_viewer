#version 430 core

layout(vertices = 3) out;

in VtxData {
  vec3 worldPos;
  vec3 normal;
  vec2 uv0;
  vec2 uvDecal;
  vec3 matT;
  vec3 matB;
  vec3 decalT;
  vec3 decalB;
  int  sl;
} tc_in[];

out TescData {
  vec3 worldPos;
  vec3 normal;
  vec2 uv0;
  vec2 uvDecal;
  vec3 matT;
  vec3 matB;
  vec3 decalT;
  vec3 decalB;
  int  sl;
} tc_out[];

uniform float tessLevel; // 1 = no subdivision (only > 1 in tessellation-displacement mode)

void main() {
  tc_out[gl_InvocationID].worldPos = tc_in[gl_InvocationID].worldPos;
  tc_out[gl_InvocationID].normal   = tc_in[gl_InvocationID].normal;
  tc_out[gl_InvocationID].uv0      = tc_in[gl_InvocationID].uv0;
  tc_out[gl_InvocationID].uvDecal  = tc_in[gl_InvocationID].uvDecal;
  tc_out[gl_InvocationID].matT     = tc_in[gl_InvocationID].matT;
  tc_out[gl_InvocationID].matB     = tc_in[gl_InvocationID].matB;
  tc_out[gl_InvocationID].decalT   = tc_in[gl_InvocationID].decalT;
  tc_out[gl_InvocationID].decalB   = tc_in[gl_InvocationID].decalB;
  tc_out[gl_InvocationID].sl       = tc_in[gl_InvocationID].sl;

  if (gl_InvocationID == 0) {
    gl_TessLevelInner[0] = tessLevel;
    gl_TessLevelOuter[0] = tessLevel;
    gl_TessLevelOuter[1] = tessLevel;
    gl_TessLevelOuter[2] = tessLevel;
  }
}
