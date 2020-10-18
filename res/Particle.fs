#version 410 core

in vec2 Vertex_UV;
in vec3 Vertex_Color;
flat in int Vertex_Type;
in vec3 Vertex_Normal;
in vec3 Vertex_World;
in vec3 Vertex_IlluminatedColor;

out vec4 FragColor;

uniform float opacity;
uniform vec3 lightPosition;

void main(){
	if(Vertex_Type == 0){
		vec2 diff = Vertex_UV * 2.0 - vec2(1.0, 1.0);
		float distSq = dot(diff, diff);
		vec3 color = Vertex_IlluminatedColor;
		if(distSq < 1.0){
			float mkx = -opacity * (1.0 - distSq);
			FragColor.a = 1.0 - exp(mkx);
			//FragColor.a = 1.0f;
		} else {
			FragColor.a = 0.0;
		}
		FragColor.xyz = FragColor.a * color;
	} else {
		float c = dot(normalize(lightPosition - Vertex_World), Vertex_Normal);
		vec3 baseColor = vec3(0.1, 0.1, 0.1);
		if(c > 0.0){
			FragColor = vec4(baseColor + 0.8 * c * vec3(Vertex_IlluminatedColor.z, Vertex_IlluminatedColor.z, Vertex_IlluminatedColor.z), 1.0);
		} else {
			FragColor = vec4(baseColor, 1.0);
		}
	}
}

