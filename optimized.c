#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include<stdlib.h>
#include<string.h>
#include <complex.h>
#include <pthread.h>

#define WIDTH 1024
#define HEIGHT 1024
#define MAX_ITERATIONS 1000
#define FRAMES 120
#define MAX_MAGNIFICATION 10000000000000

unsigned int Mand (double initx, double inity, unsigned int iterations) {
    
    double cx,cy,xsq,ysq;
    unsigned int iter;
    
    cx=initx+initx*initx - inity*inity;   
    cy=inity+initx*inity+initx*inity; 
    
    for (iter=0;iter<iterations && (ysq=cy*cy)+(xsq=cx*cx)<4;iter++,cy=inity+cx*cy+cx*cy,cx=initx-ysq+xsq) ;
    return (iter);
}

unsigned char colors[4][4] = {{255, 0, 0, 255},{0, 255, 0, 255},{0, 0, 255},{123, 230, 90, 255}};
double dx = -0.0700212907411218; 
double dy = - 0.8224676131988761;



int main(){
    unsigned char* data = malloc(HEIGHT*WIDTH*4*sizeof(unsigned char));
    double scale = 1.0;
    for(int frame = 0; frame < FRAMES; frame++){
        scale /= powf(MAX_MAGNIFICATION, 1.0f/FRAMES);
        for(int i = 0; i< WIDTH; i++){
            for(int j = 0; j < HEIGHT; j++){
                double x = (((double)i/WIDTH) * scale)-0.0700212907411218;
                double y = (((double)j/HEIGHT) * scale)-0.8224676131988761;
                int res = Mand(x, y, MAX_ITERATIONS);
                if(res == 1000){
                    data[(WIDTH*j+i)*4+0] = 0;
                    data[(WIDTH*j+i)*4+1] = 0;
                    data[(WIDTH*j+i)*4+2] = 0;
                    data[(WIDTH*j+i)*4+3] = 255;
                }
                else{
                    float v = sinf(powf((powf((float)res/(float)MAX_ITERATIONS, 0.5)*4), 0.5));
                    data[(WIDTH*j+i)*4+0] = 255*v;
                    data[(WIDTH*j+i)*4+1] = 255*v;
                    data[(WIDTH*j+i)*4+2] = 255*v;
                    data[(WIDTH*j+i)*4+3] = 255;
                }
            }
        }
        char buf[255];
        sprintf(buf, "./frames/%d.png", frame);
        stbi_write_png(buf, WIDTH, HEIGHT, 4, data,WIDTH*4);
        printf("Generated frame: %d @ scale: %lf\n", frame, scale);
    }
    return 0;
}