#version 410 core

in vec3 positionWorld;
in vec3 normalWorld;

uniform vec4 color;
uniform vec3 cameraPositionWorld;
uniform float lighting_factor;

out vec4 FragColor;

void main(){
	//material settings
	vec3 ambientColor = color.xyz;
	vec3 diffuseColor = color.xyz;
	vec3 specColor = vec3(1.0, 1.0, 1.0);
	float shininess = 10.0;
	
	//light settings
	vec3 lightColor = vec3(1.0, 1.0, 1.0);
	vec3 lightDirection = vec3(1.0, 0.0, 0.0);
	
	vec3 n = normalize(normalWorld);
	vec3 s = -lightDirection; //s points to light, but because it is at infinity it is the negative direction of lightDirection
	vec3 v = normalize(cameraPositionWorld - positionWorld ); // v points from fragment point to viewer
	vec3 r = reflect( -s, n );
	
	float sDotN = max( dot( s, n ), 0.0 );
	vec3 diffuse = lightColor * diffuseColor * sDotN;
	
	vec3 spec = lightColor * specColor * pow( max( dot(r,v) , 0.0 ), shininess ); 
	
	vec3 phong = diffuse + spec;
	
	FragColor.xyz = (1.0 - lighting_factor) * ambientColor + lighting_factor * phong;
	FragColor.a = color.a;
}