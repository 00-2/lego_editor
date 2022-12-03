#version 450 core
out vec4 FragColor;
in vec3 faceNorm;
in vec4 worldPos;
uniform vec3 partColor;

uniform vec3 lightPosition, light1Position;
uniform float dim;

void main() {
  vec3 world = worldPos.xyz / worldPos.w;
  vec3 lightDirection = world - lightPosition;
  vec3 light1Direction = world - light1Position;
  float pw = length(lightDirection);
  float pw1 = length(light1Direction);

  lightDirection /= pw;
  light1Direction /= pw1;
  pw = pow(dim, pw);
  pw1 = pow(dim, pw1);
  FragColor = max(dot(faceNorm, lightDirection) * pw, dot(faceNorm, light1Direction) * pw1) * vec4(partColor, 1);
  //FragColor = vec4((faceNorm + 1) / 2, 1);
}
