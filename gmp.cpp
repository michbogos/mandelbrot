#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<thread>
#include<vector>
#include<queue>
#include<gmp.h>
#include<omp.h>
#define WIDTH 256
#define HEIGHT 256
#define MAX_ITERATIONS 1000
#define FRAMES 10
#define MAX_MAGNIFICATION 0.0000000000001
#define NUM_THREADS 8

unsigned int Mand (mpf_t initx, mpf_t inity, unsigned int iterations) {
    mpf_t cx,cy,xsq,ysq,ab, cmp;
    mpf_init_set_d(cx,  0.0);
    mpf_init_set_d(cy,  0.0);
    mpf_init_set_d(xsq, 0.0);
    mpf_init_set_d(ysq, 0.0);
    mpf_init_set_d(ab, 0.0);
    mpf_init_set_d(cmp, 0.0);
    unsigned int iter = 0;

    mpf_mul(xsq, initx, initx);
    mpf_mul(ysq, inity, inity);
    mpf_mul(ab,  initx, inity);

    mpf_sub(cx, xsq, ysq);
    mpf_add(cx, cx, initx);

    mpf_add(cy, ab, ab);
    mpf_add(cy, cy, inity);


    while(true){
        mpf_mul(xsq, cx, cx);
        mpf_mul(ysq, cy, cy);
        mpf_add(cmp, xsq, ysq);
        if(iter > iterations || mpf_cmp_ui(cmp, 16)>0) break;
        iter++;
        mpf_mul(ab,  cx, cy);

        mpf_sub(cx, xsq, ysq);
        mpf_add(cx, cx, initx);

        mpf_add(cy, ab, ab);
        mpf_add(cy, cy, inity);
    }
    return iter;

    // cx=initx+initx*initx - inity*inity;   
    // cy=inity+initx*inity+initx*inity; 

    // for (iter=0;iter<iterations && (ysq=cy*cy)+(xsq=cx*cx)<4;iter++,cy=inity+cx*cy+cx*cy,cx=initx-ysq+xsq) ;
}


int main(){
    mpf_set_default_prec(128);
    mpf_t scale, dx, dy, scale_mul, iw, jh, width, height, x, y;
    mpf_init_set_d(dx,-0.026593792304386393);
    mpf_init_set_d(dy,-0.8095285579867694);
    mpf_init_set_d(width, ((double)WIDTH));
    mpf_init_set_d(height, ((double)HEIGHT));
    mpf_init_set_d(scale, 0.01);
    mpf_init_set_d(scale_mul, 0.90);
    mpf_init(iw);
    mpf_init(jh);
    mpf_init(x);
    mpf_init(y);

    for(int frame = 0; frame < FRAMES; frame++){
        unsigned char* data = (unsigned char*)malloc(HEIGHT*WIDTH*4*sizeof(unsigned char));
        mpf_mul(scale, scale, scale_mul);
            for(int i = 0; i< WIDTH; i++){
                mpf_set_d(iw, (double)i/WIDTH);
                for(int j = 0; j < HEIGHT; j++){
                    mpf_set_d(jh, (double)j/HEIGHT);
                    mpf_mul(x, iw, scale);
                    mpf_mul(y, jh, scale);
                    mpf_add(x, x, dx);
                    mpf_add(y, y, dy);
                    int res = Mand(x, y, MAX_ITERATIONS);
                    if(res > 999){
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
    return 0;
}