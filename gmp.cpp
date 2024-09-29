#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<thread>
#include<vector>
#include<queue>
#include<gmp.h>
#define WIDTH 1024
#define HEIGHT 1024
#define MAX_ITERATIONS 1000
#define FRAMES 2400
#define MAX_MAGNIFICATION 0.0000000000001
#define NUM_THREADS 8

unsigned int Mand (mpf_t initx, mpf_t inity, unsigned int iterations) {
    mpf_t cx,cy,xsq,ysq, tmp;
    mpf_init(cx);
    mpf_init(cy);
    mpf_init(xsq);
    mpf_init(ysq);
    mpf_init(tmp);
    unsigned int iter = 0;

    while(true){
        mpf_pow_ui(ysq, cy, 2);
        mpf_pow_ui(xsq, cx, 2);
        mpf_add(tmp, xsq, ysq);
        if(iter > iterations || mpf_cmp_ui(tmp, 4)>0) break;
        iter++;
        mpf_mul(tmp, cx, cy);
        mpf_mul_ui(tmp, tmp, 2);
        mpf_add(cy, inity, tmp);

        mpf_add(tmp, ysq, xsq);
        mpf_sub(cx, initx, tmp);
    }
    return iter;


    // mpf_mul(tmp, initx, initx);
    // mpf_add(cx, cx,tmp);
    // mpf_add(cx, cx,initx);
    // mpf_clear(tmp);
    // mpf_mul(tmp, inity, inity);
    // mpf_sub(cx, cx, tmp);

    // mpf_add(cy, cy, inity);
    // mpf_clear(tmp);
    // mpf_mul(tmp, initx, inity);
    // mpf_add(cy, cy, tmp);
    // mpf_add(cy, cy, tmp);

    // cx=initx+initx*initx - inity*inity;   
    // cy=inity+initx*inity+initx*inity; 
}

unsigned char colors[4][4] = {{255, 0, 0, 255},{0, 255, 0, 255},{0, 0, 255},{123, 230, 90, 255}};
double dx = -0.0700212907411218; 
double dy = - 0.8224676131988761;

std::queue<int> tasks;

void imagen(){
    mpf_t scale, dx, dy, scale_mul, iw, jh, width, height, x, y;
    mpf_init_set_str(dx, "-0.0700212907411218", 10);
    mpf_init_set_str(dy, "-0.8224676131988761", 10);
    mpf_init_set_d(width, ((double)WIDTH));
    mpf_init_set_d(height, ((double)HEIGHT));
    while(!tasks.empty()){
        int frame = tasks.front();
        tasks.pop();
        unsigned char* data = (unsigned char*)malloc(HEIGHT*WIDTH*4*sizeof(unsigned char));
        mpf_init_set_d(scale, 1.0);
        mpf_init_set_d(scale_mul, 0.99);
        for(int i = 0; i < frame; i++){
            mpf_mul(scale, scale, scale_mul);
        }
            for(int i = 0; i< WIDTH; i++){
                mpf_set_d(iw, (double)i/WIDTH);
                for(int j = 0; j < HEIGHT; j++){
                    mpf_set_d(iw, (double)j/HEIGHT);
                    mpf_mul(x, iw, scale);
                    mpf_mul(y, jh, scale);
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
            printf("Generated frame: %d", frame);
        }
    return;
}


int main(){
    mpf_set_default_prec(64);
    int frame = 0;
    for(int i = 0; i < FRAMES; i++){
        tasks.push(i);
    }
    // std::vector<std::thread> threads;
    // for(int i = 0; i < NUM_THREADS; i++){
    //     threads.push_back(std::thread(imagen));
    //     frame ++;
    // }
    // for(int i = 0 ; i < NUM_THREADS; i++){
    //     threads[i].join();
    // }
    imagen();
    return 0;
}