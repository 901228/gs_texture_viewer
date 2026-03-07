#version 430 core

layout(triangles, equal_spacing, ccw) in;

in Tcs_out {
  vec3 position;
  vec3 normal;
  vec2 textureCoord;
  int sl;
  vec3 tangent;
  vec3 bitangent;
} tes_in[];

out Tes_out {
  vec3 position;
  vec3 normal;
  vec2 textureCoord;
  int sl;
  vec3 tangent;
  vec3 bitangent;
} tes_out;

uniform mat4 projection_matrix;
uniform mat4 view_matrix;
uniform mat4 model_matrix;

uniform bool isEditTexture;
uniform bool isRenderTexture;
uniform bool isRenderSelect;

uniform sampler2D heightMap;
uniform float heightScale;

uniform float textureRadius;
uniform vec2 textureOffset;
uniform float textureTheta;

// barycentric interpolation helper
#define INTERP(attr) \
  (gl_TessCoord.x * tes_in[0].attr + \
   gl_TessCoord.y * tes_in[1].attr + \
   gl_TessCoord.z * tes_in[2].attr)

vec4 getEditingTextureColor(sampler2D tex, vec2 uv) {
  mat2 rot = mat2(
    vec2(cos(textureTheta), -sin(textureTheta)),
    vec2(sin(textureTheta), cos(textureTheta))
  );
  return texture(tex, rot * ((uv - 0.5) * textureRadius) + 0.5 + textureOffset);
}

void main() {
  // interpolate all attributes
  vec3 pos     = INTERP(position);
  vec3 norm    = normalize(INTERP(normal));
  vec2 uv      = INTERP(textureCoord);
  vec3 tang    = INTERP(tangent);
  vec3 bitang  = INTERP(bitangent);
  int  sl_val  = tes_in[0].sl; // flat (pick first)

  // Height map displacement
  int sl_min = min(tes_in[0].sl, min(tes_in[1].sl, tes_in[2].sl));
  if ((isRenderTexture && !isEditTexture && sl_min >= 0) ||
      (isRenderSelect && isEditTexture)) {
    float h = getEditingTextureColor(heightMap, uv).r;
    pos += norm * h * heightScale;

    // re-calculate normal from displaced surface
    const float epsilon = 0.01;
    float hR = getEditingTextureColor(heightMap, uv + vec2(epsilon, 0.0f)).r;
    float hU = getEditingTextureColor(heightMap, uv + vec2(0.0f, epsilon)).r;

    // use local-space tang/bitang/norm to build displaced tangent vectors
    vec3 dPdU = tang + norm * ((hR - h) / epsilon) * heightScale;
    vec3 dPdV = bitang + norm * ((hU - h) / epsilon) * heightScale;
    norm = normalize(cross(dPdU, dPdV));
  }

  // transform to world/view space after displacement
  mat3 normalMatrix = transpose(inverse(mat3(model_matrix)));
  vec3 N = normalize(normalMatrix * norm);
  vec3 T = normalize(normalMatrix * tang);
  // Gram-Schmidt re-orthogonalization
  T = normalize(T - dot(T, N) * N);
  vec3 B = cross(N, T);

  // matrix transformation
  vec4 p = view_matrix * model_matrix * vec4(pos, 1.0);
  gl_Position = projection_matrix * p;

  tes_out.position    = p.xyz;
  tes_out.textureCoord = uv;
  tes_out.normal      = N;
  tes_out.tangent     = T;
  tes_out.bitangent   = B;
  tes_out.sl          = sl_val;
}
