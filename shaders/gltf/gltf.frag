#version 430 core
out vec4 FragColor;

in VS_OUT {
  vec3 worldPos;
  vec3 normal;
  vec2 uv0;
  vec2 uvDecal;
  vec3 matT;
  vec3 matB;
  vec3 decalT;
  vec3 decalB;
} fs_in;

flat in int sl;

const float PI = 3.14159265359;

uniform vec3 viewPos;

// ---- directional lights ----
#define MAX_LIGHTS 8
struct Light {
  vec3 direction; // direction the light travels (from light toward the scene)
  vec3 color;
  float intensity;
};
uniform Light lights[MAX_LIGHTS];
uniform int numLights;

uniform bool isRenderTextureCoords;

// ---- glTF metallic-roughness material ----
uniform sampler2D baseColorTex;          // unit 0
uniform sampler2D metallicRoughnessTex;  // unit 1 (G = roughness, B = metallic)
uniform sampler2D normalTex;             // unit 2
uniform sampler2D emissiveTex;           // unit 3
uniform sampler2D occlusionTex;          // unit 4
uniform vec3  baseColorFactor;
uniform float metallicFactor;
uniform float roughnessFactor;
uniform vec3  emissiveFactor;
uniform bool  hasNormalTex;
uniform bool  flipNormals;      // negate the geometric normal (e.g. when glb normals point inward)
uniform bool  decalNormalOnly;  // build the decal normal on the geometric normal, ignoring the glb normal map

// ---- decal (the user's PBR texture), only on the active sub-mesh ----
uniform bool isActiveSubMesh;
uniform bool hasDecal;
struct DecalMaterial { sampler2D basecolor; sampler2D normal; };
uniform DecalMaterial decal;             // units 5, 6
uniform sampler2D decalHeightMap;        // unit 7
uniform sampler2D decalMask;             // unit 8
uniform sampler2D decalRoughness;        // unit 9
uniform float     decalRoughnessScale;   // multiplies the decal roughness
uniform float decalHeightScale;
uniform int  heightMode;                 // 0 none, 1 parallax-occlusion (2 = tess, unsupported here)
uniform float textureRadius;
uniform vec2  textureOffset;
uniform float textureTheta;

mat2 texRotation() {
  return mat2(vec2(cos(textureTheta), -sin(textureTheta)),
              vec2(sin(textureTheta),  cos(textureTheta)));
}
vec2 toTextureUV(vec2 uv) {
  return texRotation() * ((uv - 0.5) * textureRadius) + 0.5 + textureOffset;
}

vec3 toLinear(vec3 c) { return pow(c, vec3(2.2)); }

// Parallax Occlusion Mapping in decal-uv space (ported from shader.frag).
vec2 parallaxOcclusion(vec2 uv, vec3 viewDirTS) {
  if (abs(viewDirTS.z) < 1e-3)
    return uv;
  const float minLayers = 8.0;
  const float maxLayers = 32.0;
  float numLayers = mix(maxLayers, minLayers, clamp(abs(viewDirTS.z), 0.0, 1.0));
  float layerDepth = 1.0 / numLayers;
  float currentDepth = 0.0;
  vec2 P = (viewDirTS.xy / viewDirTS.z) * decalHeightScale;
  vec2 deltaUV = P / numLayers;
  vec2 currentUV = uv;
  float currentH = 1.0 - texture(decalHeightMap, currentUV).r;
  for (int i = 0; i < int(maxLayers); ++i) {
    if (currentDepth >= currentH) break;
    currentUV -= deltaUV;
    currentH = 1.0 - texture(decalHeightMap, currentUV).r;
    currentDepth += layerDepth;
  }
  vec2 prevUV = currentUV + deltaUV;
  float afterDepth = currentH - currentDepth;
  float beforeDepth = (1.0 - texture(decalHeightMap, prevUV).r) - currentDepth + layerDepth;
  float denom = afterDepth - beforeDepth;
  float weight = abs(denom) < 1e-5 ? 0.0 : afterDepth / denom;
  return mix(currentUV, prevUV, weight);
}

// ---- Cook-Torrance terms ----
float D_GGX(float NdotH, float rough) {
  float a = rough * rough;
  float a2 = a * a;
  float d = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
  return a2 / max(PI * d * d, 1e-7);
}
float G_SchlickGGX(float NdotV, float rough) {
  float r = rough + 1.0;
  float k = (r * r) / 8.0;
  return NdotV / (NdotV * (1.0 - k) + k);
}
float G_Smith(float NdotV, float NdotL, float rough) {
  return G_SchlickGGX(NdotV, rough) * G_SchlickGGX(NdotL, rough);
}
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
  return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main() {
  if (isRenderTextureCoords) {
    FragColor = vec4(fs_in.uvDecal, 0.0, 1.0);
    return;
  }

  // ---- material sampling ----
  vec3 albedo = baseColorFactor * toLinear(texture(baseColorTex, fs_in.uv0).rgb);
  vec2 mr = texture(metallicRoughnessTex, fs_in.uv0).gb; // G=roughness, B=metallic
  float roughness = clamp(roughnessFactor * mr.x, 0.04, 1.0);
  float metallic = metallicFactor * mr.y;
  float occlusion = texture(occlusionTex, fs_in.uv0).r;
  vec3 emissive = emissiveFactor * toLinear(texture(emissiveTex, fs_in.uv0).rgb);

  // ---- normal ----
  vec3 N = normalize(fs_in.normal);
  if (flipNormals)
    N = -N;
  vec3 Ng = N; // geometric normal, kept free of the glb normal map for the decal frame
  vec3 V = normalize(viewPos - fs_in.worldPos);

  if (hasNormalTex && length(fs_in.matT) > 1e-5) {
    vec3 T = normalize(fs_in.matT - dot(fs_in.matT, N) * N);
    vec3 B = normalize(cross(N, T));
    vec3 mapN = texture(normalTex, fs_in.uv0).rgb * 2.0 - 1.0;
    N = normalize(mat3(T, B, N) * mapN);
  }

  // ---- decal overlay (active sub-mesh, marked faces only) ----
  if (hasDecal && isActiveSubMesh && sl >= 0) {
    bool hasDecalTBN = length(fs_in.decalT) > 1e-5 && length(fs_in.decalB) > 1e-5;
    vec3 dT = hasDecalTBN ? normalize(fs_in.decalT) : vec3(1, 0, 0);
    vec3 dB = hasDecalTBN ? normalize(fs_in.decalB) : vec3(0, 1, 0);
    // build the decal tangent frame on the geometric normal (decalNormalOnly) so the glb
    // normal map's detail (brushing / weave) does not bleed into the decal's own normal;
    // otherwise fall back to the glb-perturbed normal N (blended look)
    mat3 decalTBN = mat3(dT, dB, decalNormalOnly ? Ng : N);

    vec2 duv = toTextureUV(fs_in.uvDecal);
    if (heightMode == 1) {
      vec3 viewDirTS = normalize(transpose(decalTBN) * V);
      duv = parallaxOcclusion(duv, viewDirTS);
    }

    vec4 dcol = texture(decal.basecolor, duv);
    float coverage = dcol.a * texture(decalMask, duv).r;

    // Where the decal is opaque, drop the glb material entirely and use only the logo's
    // own material (basecolor / roughness / normal). A painted decal is a dielectric, so
    // force metallic to 0 and neutralize the glb occlusion/emissive under the logo.
    albedo    = mix(albedo, toLinear(dcol.rgb), coverage);
    roughness = mix(roughness, clamp(texture(decalRoughness, duv).r * decalRoughnessScale, 0.04, 1.0), coverage);
    metallic  = mix(metallic, 0.0, coverage);
    occlusion = mix(occlusion, 1.0, coverage);
    emissive  = mix(emissive, vec3(0.0), coverage);

    vec3 dmapN = texture(decal.normal, duv).rgb * 2.0 - 1.0;
    N = normalize(mix(N, normalize(decalTBN * dmapN), coverage));
  }

  // ---- Cook-Torrance direct lighting (accumulate every enabled directional light) ----
  float NdotV = max(dot(N, V), 1e-4);
  vec3 F0 = mix(vec3(0.04), albedo, metallic);

  vec3 Lo = vec3(0.0);
  for (int i = 0; i < numLights; ++i) {
    vec3 L = normalize(-lights[i].direction);
    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float HdotV = max(dot(H, V), 0.0);

    float D = D_GGX(NdotH, roughness);
    float G = G_Smith(NdotV, NdotL, roughness);
    vec3 F = fresnelSchlick(HdotV, F0);

    vec3 spec = (D * G * F) / max(4.0 * NdotV * NdotL, 1e-4);
    vec3 kd = (vec3(1.0) - F) * (1.0 - metallic);

    vec3 radiance = lights[i].color * lights[i].intensity;
    Lo += (kd * albedo / PI + spec) * radiance * NdotL;
  }

  vec3 ambient = 0.12 * albedo * occlusion;
  vec3 color = ambient + Lo + emissive;

  // Reinhard tonemap + gamma
  color = color / (color + vec3(1.0));
  color = pow(color, vec3(1.0 / 2.2));

  FragColor = vec4(color, 1.0);
}
