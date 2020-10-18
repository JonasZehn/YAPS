#version 410 core

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
	vec3 illuminatedColor;
} vertex[];

out vec2 Vertex_UV;
out vec3 Vertex_Color;
flat out int Vertex_Type;
out vec3 Vertex_Normal;
out vec3 Vertex_World;
out vec3 Vertex_IlluminatedColor;
	
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
		Vertex_IlluminatedColor = vertex[0].illuminatedColor;
		EmitVertex();	
		
		vec3 vb = P - (right - up) * size;
		gl_Position = MVP * vec4(vb, 1.0);
		Vertex_UV = vec2(0.0, 1.0);
		Vertex_Color = vertex[0].color;
		Vertex_Type = vertex[0].type;
		Vertex_IlluminatedColor = vertex[0].illuminatedColor;
		EmitVertex();	

		vec3 vd = P + (right - up) * size;
		gl_Position = MVP * vec4(vd, 1.0);
		Vertex_UV = vec2(1.0, 0.0);
		Vertex_Color = vertex[0].color;
		Vertex_Type = vertex[0].type;
		Vertex_IlluminatedColor = vertex[0].illuminatedColor;
		EmitVertex();	

		vec3 vc = P + (right + up) * size;
		gl_Position = MVP * vec4(vc, 1.0);
		Vertex_UV = vec2(1.0, 1.0);
		Vertex_Color = vertex[0].color;
		Vertex_Type = vertex[0].type;
		Vertex_IlluminatedColor = vertex[0].illuminatedColor;
		EmitVertex();	

		EndPrimitive();
	} else {
		vec3 PAvg = gl_in[0].gl_Position.xyz;
		vec3 x1 = vertex[0].x1;
		vec3 x2 = vertex[0].x2;
		vec3 x3 = 3.0f * PAvg - vertex[0].x1 - vertex[0].x2; //PAvg = (x1 + x2 + x3)/3.0 => x3 = 3.0 * PAvg - x1 - x2
		vec3 n = normalize(cross(x2 - x1, x3 - x1));

		vec3 va = x1;
		gl_Position = MVP * vec4(va, 1.0);
		Vertex_UV = vec2(0.0, 0.0);
		Vertex_Color = vertex[0].color;
		Vertex_Type = vertex[0].type;
		Vertex_Normal = n;
		Vertex_World = x1;
		Vertex_IlluminatedColor = vertex[0].illuminatedColor;
		EmitVertex();	
		
		vec3 vb = x2;
		gl_Position = MVP * vec4(vb, 1.0);
		Vertex_UV = vec2(0.0, 1.0);
		Vertex_Color = vertex[0].color;
		Vertex_Type = vertex[0].type;
		Vertex_Normal = n;
		Vertex_World = x2;
		Vertex_IlluminatedColor = vertex[0].illuminatedColor;
		EmitVertex();

		vec3 vd = x3;
		gl_Position = MVP * vec4(vd, 1.0);
		Vertex_UV = vec2(1.0, 0.0);
		Vertex_Color = vertex[0].color;
		Vertex_Type = vertex[0].type;
		Vertex_Normal = n;
		Vertex_World = x3;
		Vertex_IlluminatedColor = vertex[0].illuminatedColor;
		EmitVertex();

		EndPrimitive();
	}
}	 