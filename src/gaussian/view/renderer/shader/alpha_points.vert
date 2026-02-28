/*
 * Copyright (C) 2020, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */


#version 430

layout (std430, binding = 0) buffer BoxCenters {
    float centers[];
};
layout (std430, binding = 4) buffer Colors {
    float colors[];
};

uniform mat4 projection_matrix;
uniform mat4 view_matrix;
uniform mat4 model_matrix;
uniform int radius;

out vec3 colorVert;

void main() {
  int instance_id = gl_VertexID;

  vec3 ellipsoidCenter = vec3(centers[3 * instance_id + 0], centers[3 * instance_id + 1], centers[3 * instance_id + 2]);

	float r = colors[instance_id * 48 + 0] * 0.2 + 0.5;
	float g = colors[instance_id * 48 + 1] * 0.2 + 0.5;
	float b = colors[instance_id * 48 + 2] * 0.2 + 0.5;
	colorVert = vec3(r, g, b);

  gl_Position = projection_matrix * view_matrix * model_matrix * vec4(ellipsoidCenter, 1.0);
  gl_PointSize = radius;
}
