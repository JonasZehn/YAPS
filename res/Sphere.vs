#version 410 core

layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 normal_modelspace;

uniform mat4 MVP;
uniform mat4 Model;
uniform mat3 ModelInverseTranspose;

out vec3 positionWorld;
out vec3 normalWorld;

void main(){	

	gl_Position = MVP * vec4(vertexPosition_modelspace, 1.0);
	vec4 pw = Model * vec4(vertexPosition_modelspace, 1.0);
	positionWorld = pw.xyz / pw.w;
	//normalWorld = normalize(ModelInverseTranspose * normal_modelspace);
	normalWorld = normalize(normal_modelspace);
}

