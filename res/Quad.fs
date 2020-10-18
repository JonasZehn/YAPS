#version 410 core

in vec2 UV;

out vec4 FragColor;

uniform sampler2D myTextureSampler;

void main(){
	
	FragColor.xyz = texture( myTextureSampler, UV ).rgb;
	FragColor.a = 1.0f;
}