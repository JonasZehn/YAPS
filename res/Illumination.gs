#version 430 core

layout (points) in;
layout (triangle_strip) out;
layout (max_vertices = 4) out;
	
uniform mat4 MVP;

uniform vec3 right;
uniform vec3 up;

uniform float size; // Particle size

//http://www.geeks3d.com/20140815/particle-billboarding-with-the-geometry-shader-glsl/

in Vertex
{
  vec3 color;
  int type;
  vec3 x1;
  vec3 x2;
  int idx;
} vertex[];

out vec2 Vertex_UV;
out vec3 Vertex_Color;
flat out int Vertex_Type;
flat out int Vertex_ID;
	
void main (void)
{	
	if(vertex[0].type == 0)
	{
		vec3 P = gl_in[0].gl_Position.xyz;

		vec3 va = P - (right + up) * size;
		gl_Position = MVP * vec4(va, 1.0);
		Vertex_UV = vec2(0.0, 0.0);
		Vertex_Color = vertex[0].color;
		Vertex_Type = vertex[0].type;
		Vertex_ID = vertex[0].idx;
		EmitVertex();	
		
		vec3 vb = P - (right - up) * size;
		gl_Position = MVP * vec4(vb, 1.0);
		Vertex_UV = vec2(0.0, 1.0);
		Vertex_Color = vertex[0].color;
		Vertex_Type = vertex[0].type;
		Vertex_ID = vertex[0].idx;
		EmitVertex();	

		vec3 vd = P + (right - up) * size;
		gl_Position = MVP * vec4(vd, 1.0);
		Vertex_UV = vec2(1.0, 0.0);
		Vertex_Color = vertex[0].color;
		Vertex_Type = vertex[0].type;
		Vertex_ID = vertex[0].idx;
		EmitVertex();	

		vec3 vc = P + (right + up) * size;
		gl_Position = MVP * vec4(vc, 1.0);
		Vertex_UV = vec2(1.0, 1.0);
		Vertex_Color = vertex[0].color;
		Vertex_Type = vertex[0].type;
		Vertex_ID = vertex[0].idx;
		EmitVertex();	

		EndPrimitive();	
	} else {
		vec3 PAvg = gl_in[0].gl_Position.xyz;

		vec3 va = vertex[0].x1;
		gl_Position = MVP * vec4(va, 1.0);
		Vertex_UV = vec2(0.0, 0.0);
		Vertex_Color = vertex[0].color;
		Vertex_Type = vertex[0].type;
		Vertex_ID = vertex[0].idx;
		EmitVertex();	
		
		vec3 vb = vertex[0].x2;
		gl_Position = MVP * vec4(vb, 1.0);
		Vertex_UV = vec2(0.0, 1.0);
		Vertex_Color = vertex[0].color;
		Vertex_Type = vertex[0].type;
		Vertex_ID = vertex[0].idx;
		EmitVertex();

		vec3 vd = 3.0f * PAvg - vertex[0].x1 - vertex[0].x2; //PAvg = (x1 + x2 + x3)/3.0 => x3 = 3.0 * PAvg - x1 - x2
		gl_Position = MVP * vec4(vd, 1.0);
		Vertex_UV = vec2(1.0, 0.0);
		Vertex_Color = vertex[0].color;
		Vertex_Type = vertex[0].type;
		Vertex_ID = vertex[0].idx;
		EmitVertex();

		EndPrimitive();
	}
}

