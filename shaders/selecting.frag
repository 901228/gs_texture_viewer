#version 430 core

in Gs_out {

	vec3 position;
	vec3 normal;
	vec2 textureCoord;
	vec3 barycentric;
} fs_in;

out uint color;

void main() {

    color = gl_PrimitiveID;
}
