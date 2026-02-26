#version 430 core
out vec4 FragColor;

in Gs_out {

	vec3 position;
	vec3 normal;
	vec2 textureCoord;
	vec3 barycentric;
} fs_in;

uniform vec3 viewPos;
vec3 objectColor = vec3(1.0f, 0.5f, 0.2f);

uniform bool isRenderWire;
vec3 wireColor = vec3(0.0f, 0.0f, 0.0f);
float wireWidth = 2.5f;
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
  return (amb + diff + spec) * light.color * objectColor;
}

void main() {

  vec3 viewDir = normalize(viewPos - fs_in.position);
  vec3 norm = normalize(fs_in.normal);

  // directional light
  vec3 result  = vec3(0.0);
  // for(int i = 0; i < N_DIRECTIONAL_LIGHTS; i++)
  //     result += CalcDirLight(dirLight[i], norm, viewDir);
  result += CalcDirLight(dirLight[0], norm, viewDir);

  if (isRenderWire) {

      float ef = edgeFactor(wireWidth);
      result = mix(wireColor, result, ef);
  }

  FragColor = vec4(result, 1.0);
}
