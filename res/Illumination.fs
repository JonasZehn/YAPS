#version 430 core

in vec2 Vertex_UV;
in vec3 Vertex_Color;
flat in int Vertex_Type;
flat in int Vertex_ID;

uniform float opacity;

layout(binding=0, r32i) uniform iimage2D img;
layout(binding=1, r32i) uniform iimageBuffer pimg;

//first order lighting model
// energy'(x) = -k(x) * energy(x)
// ... assuming k constant => energy(x) = exp(-k x)
// but that doesn't really make sense if k = 1, so how is this physical.... maybe the idea is that k is not the density but if density = 100%  k=infinity (s.t. like log(1.0 - density) ? )

// we can only add things to img, without loss of generality
// we can say that energy at the beginning is 1, and it goes down to 0
// we define energy[i] = exp(img[i])


void main(){
	vec2 diff = Vertex_UV * 2.0 - vec2(1.0, 1.0);
	float distSq = dot(diff, diff);
	distSq = min(1.0, distSq);
	float mkx = - opacity * (1.0 - distSq);
	
	if(Vertex_Type != 0){
		mkx = - 100;
	}
	
	const float discretizationMultiplier = 256.f;
	
	int imkx = int(mkx*discretizationMultiplier);
	
	int oldVal = imageAtomicAdd(img, ivec2(gl_FragCoord.xy), imkx);
	
	float fOldVal = float(oldVal)/discretizationMultiplier;
	
	//float diffEnergy = exp(fOldVal) - exp(fNewVal);
	//    exp(fOldVal) - exp(fOldVal + mkx) = exp(fOldVal) * (1.0 - exp(mkx));
	
	if(Vertex_Type != 0){
		fOldVal = fOldVal * 3; // increase energy loss for this vertex type
	}
	
	float energy = exp(fOldVal) * (1.0f - exp(mkx));
	
	if(Vertex_Type != 0){
		energy = energy * 0.3; // reduce energy increase
	}
	
	int energyInt = int(energy * 1000.f);
	
	imageAtomicAdd(pimg, Vertex_ID, energyInt);
}

//the following is optimized for a uniform opacity value

/*
void main(){
	vec2 diff = Vertex_UV * 2.0 - vec2(1.0, 1.0);
	float distSq = dot(diff, diff);
	
	float mkxWithoutOpacity = -(1.0 - distSq);
	float mkx = opacity * mkxWithoutOpacity;
	
	const float discretizationMultiplier = 256.f;
	
	int imkxWithoutOpacity = int(mkxWithoutOpacity*discretizationMultiplier);
	
	int oldVal = imageAtomicAdd(img, ivec2(gl_FragCoord.xy), imkxWithoutOpacity);
	
	float fOldVal = opacity * float(oldVal)/discretizationMultiplier;
	
	//float diffEnergy = exp(fOldVal) - exp(fNewVal);
	//    exp(fOldVal) - exp(fOldVal + mkx) = exp(fOldVal) * (1.0 - exp(mkx));
	
	float energy = exp(fOldVal) * (1.0f - exp(mkx));
	
	int energyInt = int(energy * 1000.f);
	
	imageAtomicAdd(pimg, Vertex_ID, energyInt);
}*/

