#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"
#include<stdlib.h>
#include<string.h>
#include<stdio.h>
#include <complex.h>
#include <pthread.h>
#include <omp.h>

#define WIDTH 1024
#define HEIGHT 1024
#define MAX_ITERATIONS 1000
#define FRAMES 120
#define MAX_MAGNIFICATION 10000000000000

float smoothstep( float x )
{
  return x*x*x*(x*(x*6.0-15.0)+10.0);
}

unsigned int Mand (double initx, double inity, float* buf, int sample, unsigned int iterations) {
    
    double cx,cy,xsq,ysq;
    unsigned int iter;
    
    cx=initx+initx*initx - inity*inity;   
    cy=inity+initx*inity+initx*inity; 
    
    for(iter=0;iter<iterations && (ysq=cy*cy)+(xsq=cx*cx)<2;iter++){
        cy=inity+cx*cy+cx*cy;
        cx=initx-ysq+xsq;
        if(fabs(cx) < 1.0f && fabs(cy) < 1.0f){
        int i = fabs(cx)*WIDTH;
        int j = fabs(cy)*HEIGHT;
        buf[(int)(WIDTH*j+i)*3+(sample%3)] += 1.0f;
        // buf[(int)(WIDTH*j+i)*3+1] += 1.0f;
        // buf[(int)(WIDTH*j+i)*3+2] += 1.0f;
        }
    }
    return (iter);
}

unsigned char colors[4][4] = {{255, 0, 0, 255},{0, 255, 0, 255},{0, 0, 255},{123, 230, 90, 255}};
double dx = -0.5f; //-0.0700212907411218; 
double dy = -0.5f; //- 0.8224676131988761;



int main(){
    float* data = malloc(HEIGHT*WIDTH*3*sizeof(float));
    float scale = 10.0f;
    int progress = 0;
    #pragma omp parallel for
    for(int i = 0; i< WIDTH; i++){
        printf("%d/%d\n", progress, WIDTH);
        for(int j = 0; j < HEIGHT; j++){
            double x = (((double)i/WIDTH) * scale);
            double y = (((double)j/HEIGHT) * scale);
            for(int sample = 0; sample < 100; sample++){
                float dx = ((float)rand()/(float)RAND_MAX)*0.1;
                float dy = ((float)rand()/(float)RAND_MAX)*0.1;
                int res = Mand(x+dx, y+dy, data, sample, MAX_ITERATIONS);
            }
            // else{
            //     float v = sinf(powf((powf((float)res/(float)MAX_ITERATIONS, 0.5)*4), 0.5));
            //     data[(WIDTH*j+i)*4+0] = 255*v;
            //     data[(WIDTH*j+i)*4+1] = 255*v;
            //     data[(WIDTH*j+i)*4+2] = 255*v;
            //     data[(WIDTH*j+i)*4+3] = 255;
            // }
        }
        progress++;
    }
    for(int i = 0; i < HEIGHT*WIDTH*3; i++){
        data[i] = smoothstep(smoothstep(log10f(log10f(data[i]))));
    }
    stbi_write_hdr("img.hdr", WIDTH, HEIGHT, 3, data);
    free(data);
    return 0;
}