#version 410 core

layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec2 textureCoord;

out vec2 UV;

uniform mat4 model;

void main(){	

	gl_Position.xyz = (model*vec4(vertexPosition, 1.0)).xyz;
	gl_Position.w = 1.0;
	UV = textureCoord;
}

