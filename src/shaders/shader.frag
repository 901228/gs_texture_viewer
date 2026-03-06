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

// #define N_TEXTURE_MAX 16
// uniform sampler2D tex[N_TEXTURE_MAX];

struct Material {
    sampler2D basecolor;
    sampler2D normal;
    // sampler2D height;
};
uniform Material material;

uniform bool isEditTexture;
uniform int currentSL;

uniform bool isRenderTexture;
uniform float textureRadius;
uniform vec2 textureOffset;
uniform float textureTheta;

// uniform float heightScale;

// // Parallax Occlusion Mapping
// vec2 parallaxMapping(vec2 uv, vec3 viewDirTS) {
//   // return if the view direction is parallel to the surface
//   if (abs(viewDirTS.z) < 0.001)
//     return uv;

//   const int minLayers = 8;
//   const int maxLayers = 32;
//   float numLayers = mix(float(maxLayers), float(minLayers),
//                         abs(dot(vec3(0.0, 0.0, 1.0), viewDirTS)));

//   float layerDepth  = 1.0 / numLayers;
//   float currentDepth = 0.0;
//   vec2 deltaUV = (viewDirTS.xy * heightScale) / (abs(viewDirTS.z) * numLayers);

//   vec2  currentUV     = uv;
//   float currentHeight = 1.0 - texture(material.height, currentUV).r;

//   for (int i = 0; i < maxLayers; i++) {
//     if (currentDepth >= currentHeight)
//       break;

//     currentUV     -= deltaUV;
//     currentHeight  = 1.0 - texture(material.height, currentUV).r;
//     currentDepth  += layerDepth;
//   }

//   // Interpolate between last two layers
//   vec2  prevUV      = currentUV + deltaUV;
//   float afterDepth  = currentHeight - currentDepth;
//   float beforeDepth = (1.0 - texture(material.height, prevUV).r) - (currentDepth - layerDepth);

//   float denom = afterDepth - beforeDepth;
//   if (abs(denom) < 1e-5)
//     return currentUV;

//   float weight      = afterDepth / (afterDepth - beforeDepth);
//   return mix(currentUV, prevUV, weight);
// }

vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir) {

  vec3 lightDir = normalize(-light.direction);

  // ambient
  float amb = 0.1;

  // diffuse
  float diff = max(dot(normal, lightDir), 0.0);

  // specular
  float specularStrength = 0.5;
  vec3 reflectDir = reflect(-lightDir, normal);
  float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32) * specularStrength;

  // result
  return (amb + diff + spec) * light.color * light.intensity;
}

vec4 getTextureColor(sampler2D tex) {
  return texture(tex, fs_in.textureCoord);
}

vec4 getEditingTextureColor(sampler2D tex) {
  mat2 rotationMatrix = mat2(
    vec2(cos(textureTheta), -sin(textureTheta)),
    vec2(sin(textureTheta), cos(textureTheta))
  );
  return texture(tex, rotationMatrix * ((fs_in.textureCoord - 0.5) * textureRadius) + 0.5 + textureOffset);
}

void main() {

  vec4 result = vec4(objectColor, 1.0f);

  bool useNormalMap = false;
  // bool useHeightMap = false;

  if (isRenderTexture && !isEditTexture && sl >= 0) {
    result = getTextureColor(material.basecolor);
    useNormalMap = true;
    // useHeightMap = true;
  }
  else if (isRenderSelect) {
    if (isEditTexture) {
      result = getEditingTextureColor(material.basecolor);
      useNormalMap = true;
      // useHeightMap = true;
    }
    else if (isRenderTextureCoords) {
      result = vec4(fs_in.textureCoord, 0.0f, 1.0f);
    }
    else {
      result = vec4(1.0f, 0.0f, 0.0f, 1.0f);
    }
  }

  vec3 viewDir = normalize(viewPos - fs_in.position);
  vec3 norm = normalize(fs_in.normal);

  // TBN matrix (all in world/view space)
  vec3 T   = normalize(fs_in.tangent);
  vec3 B   = normalize(fs_in.bitangent);
  mat3 TBN = mat3(T, B, norm);

  // Tangent-space view direction (for parallax)
  // vec3 viewDirTS = normalize(transpose(TBN) * viewDir);

  // Normal map
  if (useNormalMap) {
    vec3 mapN  = getEditingTextureColor(material.normal).rgb * 2.0 - 1.0;
    norm = normalize(TBN * mapN);
  }

  // directional light
  vec4 lightingResult = vec4(0.0f);
  // for(int i = 0; i < N_DIRECTIONAL_LIGHTS; i++)
  //     result += CalcDirLight(dirLight[i], norm, viewDir);
  lightingResult += vec4(CalcDirLight(dirLight[0], norm, viewDir), 1.0f) * result;

  if (isRenderWire) {

      float ef = edgeFactor(wireWidth);
      lightingResult = mix(wireColor, result, ef);
  }

  FragColor = lightingResult;
}
