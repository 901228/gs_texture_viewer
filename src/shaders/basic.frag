#version 430 core
out vec4 FragColor;

in Vs_out {

    vec3 position;
    vec3 normal;
    vec2 textureCoord;
} fs_in;

vec3 lightPos = vec3(12.0f, 10.0f, 20.0f);
uniform vec3 viewPos;
vec3 lightColor = vec3(1.0f, 1.0f, 1.0f);
vec3 objectColor = vec3(1.0f, 0.5f, 0.2f);

void main()
{
    FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);

    // ambient
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    // diffuse
    vec3 norm = normalize(fs_in.normal);
    vec3 lightDir = normalize(lightPos - fs_in.position);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - fs_in.position);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, 1.0);
}
