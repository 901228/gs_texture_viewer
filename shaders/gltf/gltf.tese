#version 430 core

layout(triangles, equal_spacing, ccw) in;

in TescData {
  vec3 worldPos;
  vec3 normal;
  vec2 uv0;
  vec2 uvDecal;
  vec3 matT;
  vec3 matB;
  vec3 decalT;
  vec3 decalB;
  int  sl;
} te_in[];

// Matches the fragment shader's input block exactly (sl is a separate flat varying).
out VS_OUT {
  vec3 worldPos;
  vec3 normal;
  vec2 uv0;
  vec2 uvDecal;
  vec3 matT;
  vec3 matB;
  vec3 decalT;
  vec3 decalB;
} te_out;

flat out int sl;

uniform mat4 projection_matrix;
uniform mat4 view_matrix;

// decal placement + height displacement (only used in tessellation-displacement mode)
uniform bool isActiveSubMesh;
uniform bool hasDecal;
uniform sampler2D decalHeightMap;
uniform sampler2D decalMask;
uniform float decalHeightScale;
uniform int  heightMode;   // 2 == tessellation displacement
uniform int  heightInvert; // 1 == displace inward (flip direction)
uniform float textureRadius;
uniform vec2  textureOffset;
uniform float textureTheta;

#define INTERP(attr)                                                                                          \
  (gl_TessCoord.x * te_in[0].attr + gl_TessCoord.y * te_in[1].attr + gl_TessCoord.z * te_in[2].attr)

vec2 toTextureUV(vec2 uv) {
  mat2 rot = mat2(vec2(cos(textureTheta), -sin(textureTheta)), vec2(sin(textureTheta), cos(textureTheta)));
  return rot * ((uv - 0.5) * textureRadius) + 0.5 + textureOffset;
}

void main() {
  vec3 worldPos = INTERP(worldPos);
  vec3 norm     = normalize(INTERP(normal));
  vec2 uv0      = INTERP(uv0);
  vec2 uvDecal  = INTERP(uvDecal);
  vec3 matT     = INTERP(matT);
  vec3 matB     = INTERP(matB);
  vec3 decalT   = INTERP(decalT);
  vec3 decalB   = INTERP(decalB);

  // selection marker is per-face; take the min like the mesh pipeline does
  int slMin = min(te_in[0].sl, min(te_in[1].sl, te_in[2].sl));

  // height displacement only on the active sub-mesh's decal region
  if (heightMode == 2 && hasDecal && isActiveSubMesh && slMin >= 0) {
    vec2 duv = toTextureUV(uvDecal);
    float h = texture(decalHeightMap, duv).r;
    float m = texture(decalMask, duv).r; // gate by mask so non-decal areas stay flat
    float dir = (heightInvert != 0) ? -1.0 : 1.0;
    worldPos += norm * h * decalHeightScale * m * dir;
  }

  te_out.worldPos = worldPos;
  te_out.normal   = norm;
  te_out.uv0      = uv0;
  te_out.uvDecal  = uvDecal;
  te_out.matT     = matT;
  te_out.matB     = matB;
  te_out.decalT   = decalT;
  te_out.decalB   = decalB;
  sl = slMin;

  gl_Position = projection_matrix * view_matrix * vec4(worldPos, 1.0);
}
