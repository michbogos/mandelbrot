#version 330

#define product(a, b) vec2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x)
#define MAX_ITER 1000

// Input vertex attributes (from vertex shader)
in vec2 fragTexCoord;
in vec4 fragColor;

//{(struct color){10, 254, 176}, (struct color){56, 12, 86}};

vec4 colors[2] = vec4[2](vec4(56.0f/255.0f, 12.0f/255.0f, 86.0f/255.0f, 1), vec4(10.0f/255.0f, 254.0f/255.0f, 176.0f/255.0f, 1));

uniform vec2 resolution;
uniform vec2 topLeft;
uniform float scale;
// Output fragment color
out vec4 finalColor;

void main(){
    vec2 uv = ((-resolution.xy + 2.0*(gl_FragCoord.xy))/resolution.y).xy;
    vec2 z = topLeft+uv*scale;
    vec2 c = z;
    int iteration = 0;
    while(length(z)<1000 && iteration < 1000){
        z = product(product(z, product(z, z)), product(z, z)) + c;
        iteration ++;
    }
    float v = iteration+1.0f-log(log(length(z))/3)/log(3);
    finalColor = length(z)<1000 ? vec4(0, 0, 0, 1) :mix(colors[int(floor(v))%2], colors[(int(floor(v))+1)%2], v-floor(v));
}