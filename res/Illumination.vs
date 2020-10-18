#version 430 core

layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 vertexColor;
layout(location = 2) in int vertexType;
layout(location = 3) in vec3 x1;
layout(location = 4) in vec3 x2;

uniform mat4 ModelViewMatrix;

out vec3 vertexColor2;

out Vertex
{
  vec3 color;
  int type;
  vec3 x1;
  vec3 x2;
  int idx;
} vertex;

void main(){	

	gl_Position = vec4(vertexPosition_modelspace, 1.0);
	vertex.color = vertexColor;
	vertex.type = vertexType;
	vertex.x1 = x1;
	vertex.x2 = x2;
	vertex.idx = gl_VertexID;
}

