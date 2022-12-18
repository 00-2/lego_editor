#version 450 core
out vec4 FragColor;
in vec3 faceNorm;
in vec4 worldPos;
uniform float lightStatus, light1Status;

uniform vec3 partColor;

uniform vec3 lightPosition, light1Position;
uniform float dim;

void main() {
  vec4 finalColor = vec4(0.0, 0.0, 0.0, 0.0);
  vec3 lightColor = vec3(0.0, 0.0, 0.0);
  FragColor = vec4(0.0, 0.0, 0.0, 0.0);
  vec3 world = worldPos.xyz / worldPos.w;
  vec3 L = normalize(world - lightPosition); // lightDirection
  vec3 L1 = normalize(world - light1Position); // lightDirection
  vec3 E = faceNorm; // we are in Eye Coordinates, so EyePos is (0,0,0)
  vec3 R = normalize(-reflect(L, faceNorm));
  vec3 R1 = normalize(-reflect(L1, faceNorm));
  vec4 Iamb = vec4(lightColor, 1);
  vec4 Idiff = vec4(partColor, 0) * max(dot(E,L), 0.0);
  Idiff = clamp(Idiff, 0.0, 1.0);
  vec4 Ispec =  vec4(1, 1, 1, 0) * pow(max(dot(R,E),0.0),0.3*128);
  Ispec = clamp(Ispec, 0.0, 1.0);
  FragColor += ( (Iamb + Idiff + Ispec) * max(dot(E,L), 0.0) ) * lightStatus;
  vec4 Iamb1 = vec4(lightColor, 1);
  vec4 Idiff1 = vec4(partColor, 0) * max(dot(E,L1), 0.0);
  Idiff1 = clamp(Idiff1, 0.0, 1.0);
  vec4 Ispec1 =  vec4(1, 1, 1, 0) * pow(max(dot(R1,E),0.0),0.3*128);
  Ispec1 = clamp(Ispec1, 0.0, 1.0);
  FragColor += ( (Iamb1 + Idiff1 + Ispec1) * max(dot(E,L1), 0.0) ) * light1Status;
}
