#version 430 core

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in Vs_out {

  vec3 position;
  vec3 normal;
  vec2 textureCoord;
  int sl;
  vec3 tangent;
  vec3 bitangent;
} gs_in[];

out Gs_out {

	vec3 position;
	vec3 normal;
	vec2 textureCoord;
	vec3 barycentric;
  vec3 tangent;
  vec3 bitangent;
} gs_out;

flat out int sl;

void main() {

  sl = gs_in[0].sl;
  if (gs_in[1].sl < sl) sl = gs_in[1].sl;
  if (gs_in[2].sl < sl) sl = gs_in[2].sl;

	gs_out.normal = gs_in[0].normal;
	gs_out.position = gs_in[0].position;
	gs_out.textureCoord = gs_in[0].textureCoord;
	gs_out.barycentric = vec3(1, 0, 0);
	gs_out.tangent = gs_in[0].tangent;
	gs_out.bitangent = gs_in[0].bitangent;
	gl_Position = gl_in[0].gl_Position;
	gl_PrimitiveID = gl_PrimitiveIDIn;
	EmitVertex();

	gs_out.normal = gs_in[1].normal;
	gs_out.position = gs_in[1].position;
	gs_out.textureCoord = gs_in[1].textureCoord;
	gs_out.barycentric = vec3(0, 1, 0);
	gs_out.tangent = gs_in[1].tangent;
	gs_out.bitangent = gs_in[1].bitangent;
	gl_Position = gl_in[1].gl_Position;
	gl_PrimitiveID = gl_PrimitiveIDIn;
	EmitVertex();

	gs_out.normal = gs_in[2].normal;
	gs_out.position = gs_in[2].position;
	gs_out.textureCoord = gs_in[2].textureCoord;
	gs_out.barycentric = vec3(0, 0, 1);
	gs_out.tangent = gs_in[2].tangent;
	gs_out.bitangent = gs_in[2].bitangent;
	gl_Position = gl_in[2].gl_Position;
	gl_PrimitiveID = gl_PrimitiveIDIn;
	EmitVertex();

	EndPrimitive();
}
