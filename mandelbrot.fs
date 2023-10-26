#version 330

#define product(a, b) vec2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x)
#define MAX_ITER 1000

// Input vertex attributes (from vertex shader)
in vec2 fragTexCoord;
in vec4 fragColor;

uniform vec2 resolution;
uniform vec2 topLeft;
uniform float scale;
// Output fragment color
out vec4 finalColor;

void main(){
    vec2 uv = ((-resolution.xy + 2.0*(gl_FragCoord.xy))/resolution.y).xy;
    vec2 z = topLeft+uv*scale;
    vec2 c = z;
    float iteration = 0;
    while(length(z)<2 && iteration < 1000){
        z = product(z, z) + c;
        iteration ++;
    }
    finalColor = length(z)<2 ? vec4(0, 0, 0, 1) :vec4(1, 1, 1, 1)*(iteration/100);
}