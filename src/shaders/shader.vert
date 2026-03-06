#version 430 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 textureCoord;
layout (location = 3) in int sl_in;
layout (location = 4) in vec3 tangent;
layout (location = 5) in vec3 bitangent;

uniform mat4 projection_matrix;
uniform mat4 view_matrix;
uniform mat4 model_matrix;

uniform bool isEditTexture;
uniform bool isRenderTexture;
uniform bool isRenderSelect;

uniform sampler2D heightMap;
uniform float heightScale;

out Vs_out {

  vec3 position;
  vec3 normal;
  vec2 textureCoord;
  int sl;
  vec3 tangent;
  vec3 bitangent;
} vs_out;

uniform float textureRadius;
uniform vec2 textureOffset;
uniform float textureTheta;
vec4 getEditingTextureColor(sampler2D tex) {
  mat2 rotationMatrix = mat2(
    vec2(cos(textureTheta), -sin(textureTheta)),
    vec2(sin(textureTheta), cos(textureTheta))
  );
  return texture(tex, rotationMatrix * ((textureCoord - 0.5) * textureRadius) + 0.5 + textureOffset);
}

void main() {

  vec3 displacedPos = position;
  if ((isRenderTexture && !isEditTexture && sl_in >= 0) || (isRenderSelect && isEditTexture)) {
    float h = getEditingTextureColor(heightMap).r;
    displacedPos += normal * h * heightScale;
  }

  vec4 p = view_matrix * model_matrix * vec4(displacedPos, 1.0f);

  gl_Position = projection_matrix * p;
  vs_out.position = p.xyz;
  vs_out.textureCoord = textureCoord.xy;

  mat3 normalMatrix = transpose(inverse(mat3(model_matrix)));
  vec3 N = normalize(normalMatrix * normal);
  vec3 T = normalize(normalMatrix * tangent);
  // Gram-Schmidt re-orthogonalization
  T = normalize(T - dot(T, N) * N);
  vec3 B = cross(N, T);

  vs_out.normal    = N;
  vs_out.tangent   = T;
  vs_out.bitangent = B;
  vs_out.sl        = sl_in;
}
