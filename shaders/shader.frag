#version 430 core
out vec4 FragColor;

in Gs_out {

	vec3 position;
	vec3 normal;
	vec2 textureCoord;
	vec3 barycentric;
	vec3 tangent;
	vec3 bitangent;
} fs_in;

flat in int sl;

uniform vec3 viewPos;
vec3 objectColor = vec3(1.0f, 0.5f, 0.2f);

uniform bool isRenderSelect;
uniform bool isRenderWire;
uniform bool isRenderTextureCoords;
vec4 wireColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);
float wireWidth = 1.5f;
float edgeFactor(float width) {

	vec3 d = fwidth(fs_in.barycentric);
	vec3 a3 = smoothstep(vec3(0.0), d * width, fs_in.barycentric);
	return min(min(a3.x, a3.y), a3.z);
}

struct DirLight {
    vec3 direction;
    vec3 color;
    float intensity;
};
#define N_DIRECTIONAL_LIGHTS 2
uniform DirLight dirLight[N_DIRECTIONAL_LIGHTS];

struct Material {
    sampler2D basecolor;
    sampler2D normal;
};
uniform Material material;

uniform sampler2D heightMap;
uniform float heightScale;

// height mapping mode: must match PBRTexture::HeightMode
//   0 = none, 1 = parallax occlusion mapping, 2 = tessellation displacement
uniform int heightMode;
const int HEIGHT_NONE = 0;
const int HEIGHT_POM  = 1;
const int HEIGHT_TESS = 2;

uniform bool isEditTexture;
uniform int currentSL;

uniform bool isRenderTexture;
uniform float textureRadius;
uniform vec2 textureOffset;
uniform float textureTheta;

mat2 texRotation() {
  return mat2(
    vec2(cos(textureTheta), -sin(textureTheta)),
    vec2(sin(textureTheta), cos(textureTheta))
  );
}

// map the mesh uv into texture-uv space (scale / rotate / offset, matching the editor)
vec2 toTextureUV(vec2 uv) {
  return texRotation() * ((uv - 0.5) * textureRadius) + 0.5 + textureOffset;
}

vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir) {

  vec3 lightDir = normalize(-light.direction);

  // ambient
  float amb = 0.2;

  // diffuse
  float diff = max(dot(normal, lightDir), 0.0);

  // specular
  float specularStrength = 0.5;
  vec3 reflectDir = reflect(-lightDir, normal);
  float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32) * specularStrength;

  // result
  return (amb + diff + spec) * light.color * light.intensity;
}

// Parallax Occlusion Mapping in texture-uv space.
// viewDirTS: view direction in tangent space (pointing from surface to eye).
vec2 parallaxOcclusion(vec2 uv, vec3 viewDirTS) {

  // avoid division by ~0 at grazing angles
  if (abs(viewDirTS.z) < 1e-3)
    return uv;

  const float minLayers = 8.0;
  const float maxLayers = 32.0;
  float numLayers = mix(maxLayers, minLayers, clamp(abs(viewDirTS.z), 0.0, 1.0));

  float layerDepth = 1.0 / numLayers;
  float currentDepth = 0.0;

  // total uv shift from top to bottom of the height field
  vec2 P = (viewDirTS.xy / viewDirTS.z) * heightScale;
  vec2 deltaUV = P / numLayers;

  vec2 currentUV = uv;
  float currentH = 1.0 - texture(heightMap, currentUV).r;

  for (int i = 0; i < int(maxLayers); ++i) {
    if (currentDepth >= currentH)
      break;
    currentUV -= deltaUV;
    currentH = 1.0 - texture(heightMap, currentUV).r;
    currentDepth += layerDepth;
  }

  // interpolate between the last two layers (occlusion)
  vec2 prevUV = currentUV + deltaUV;
  float afterDepth = currentH - currentDepth;
  float beforeDepth = (1.0 - texture(heightMap, prevUV).r) - currentDepth + layerDepth;
  float denom = afterDepth - beforeDepth;
  float weight = abs(denom) < 1e-5 ? 0.0 : afterDepth / denom;
  return mix(currentUV, prevUV, weight);
}

void main() {

  vec4 result = vec4(objectColor, 1.0f);

  vec3 norm = normalize(fs_in.normal);
  vec3 viewDir = normalize(viewPos - fs_in.position);

  // tangent-space basis (TBN). columns are world-space T, B, N.
  vec3 T = normalize(fs_in.tangent);
  vec3 B = normalize(fs_in.bitangent);
  mat3 TBN = mat3(T, B, norm);

  // editing  : a texture is selected and painted onto selected faces (live preview transform)
  // baked    : texture render mode using the baked per-vertex texture coords
  bool editing = isRenderSelect && isEditTexture;
  bool baked = isRenderTexture && !isEditTexture && sl >= 0;

  if (editing || baked) {

    vec2 uv = editing ? toTextureUV(fs_in.textureCoord) : fs_in.textureCoord;

    // Parallax occlusion mapping shifts the uv along the view direction before sampling.
    if (heightMode == HEIGHT_POM) {
      vec3 viewDirTS = normalize(transpose(TBN) * viewDir);
      uv = parallaxOcclusion(uv, viewDirTS);
    }

    vec4 texColor = texture(material.basecolor, uv);

    // blend the decal over the base mesh color by its alpha so transparent
    // regions keep showing the underlying mesh instead of white.
    result.rgb = mix(objectColor, texColor.rgb, texColor.a);

    // normal mapping (only where the texture is actually present)
    vec3 mapN = texture(material.normal, uv).rgb * 2.0 - 1.0;
    norm = normalize(mix(norm, normalize(TBN * mapN), texColor.a));
  }
  else if (isRenderSelect) {
    if (isRenderTextureCoords) {
      result = vec4(fs_in.textureCoord, 0.0f, 1.0f);
    }
    else {
      result = vec4(1.0f, 0.0f, 0.0f, 1.0f);
    }
  }

  // directional light
  vec4 lightingResult = vec4(0.0f);
  lightingResult += vec4(CalcDirLight(dirLight[0], norm, viewDir), 1.0f) * result;

  if (isRenderWire) {

      float ef = edgeFactor(wireWidth);
      lightingResult = mix(wireColor, result, ef);
  }

  FragColor = lightingResult;
}
