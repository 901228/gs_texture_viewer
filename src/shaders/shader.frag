#version 430 core
out vec4 FragColor;

in Gs_out {

	vec3 position;
	vec3 normal;
	vec2 textureCoord;
	vec3 barycentric;
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
};
#define N_DIRECTIONAL_LIGHTS 2
uniform DirLight dirLight[N_DIRECTIONAL_LIGHTS];

#define N_TEXTURE_MAX 16
uniform sampler2D tex[N_TEXTURE_MAX];

uniform bool isEditTexture;
uniform int currentSL;

uniform bool isRenderTexture;
uniform float textureRadius;
uniform vec2 textureOffset;
uniform float textureTheta;

vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir) {

  vec3 lightDir = normalize(-light.direction);

  // ambient
  float amb = 1.0;

  // diffuse
  float diff = max(dot(normal, lightDir), 0.0);

  // specular
  float specularStrength = 0.5;
  vec3 reflectDir = reflect(-lightDir, normal);
  float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32) * specularStrength;

  // result
  return (amb + diff + spec) * light.color;
}

vec4 getTextureColor() {
  return texture(tex[sl], fs_in.textureCoord);
}

vec4 getEditingTextureColor() {
  mat2 rotationMatrix = mat2(
    vec2(cos(textureTheta), -sin(textureTheta)),
    vec2(sin(textureTheta), cos(textureTheta))
  );
  return texture(tex[currentSL], rotationMatrix * ((fs_in.textureCoord - 0.5) * textureRadius) + 0.5 + textureOffset);
}

void main() {

  vec4 result = vec4(objectColor, 1.0f);

  if (isRenderTexture && !isEditTexture && sl >= 0) {
    result = getTextureColor();
  }
  else if (isRenderSelect) {
    if (isEditTexture) {
      result = getEditingTextureColor();
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
