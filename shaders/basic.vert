#version 430 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 textureCoord;
layout (location = 3) in int sl_in;

uniform mat4 projection_matrix;
uniform mat4 view_matrix;
uniform mat4 model_matrix;

out Vs_out {

    vec3 position;
    vec3 normal;
    vec2 textureCoord;
} vs_out;

void main()
{
    vec4 p = view_matrix * model_matrix * vec4(position, 1.0f);

    gl_Position = projection_matrix * p;
    vs_out.position = p.xyz;
    vs_out.normal = transpose(inverse(mat3(model_matrix))) * normal;
    vs_out.textureCoord = textureCoord.xy;
}

