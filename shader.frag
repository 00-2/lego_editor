#version 450 core
out vec4 FragColor;
in vec3 faceNorm;
in vec4 worldPos;
uniform vec3 partColor;

uniform vec3 lightPosition;
uniform float dim;

void main() {
  vec3 world = worldPos.xyz / worldPos.w;
  vec3 lightDirection = world - lightPosition;
  float pw = length(lightDirection);
  lightDirection /= pw;
  pw = pow(dim, pw);
  FragColor = max(dot(faceNorm, lightDirection) * pw, 0.1) * vec4(partColor, 1);
  //FragColor = vec4((faceNorm + 1) / 2, 1);
}
