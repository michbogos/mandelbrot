#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include <complex.h>
#include <pthread.h>
#include<thread>
#include<vector>
#include<queue>
#include<immintrin.h>
#define WIDTH 1024
#define HEIGHT 1024
#define MAX_ITERATIONS 500
#define FRAMES 2400
#define MAX_MAGNIFICATION 0.0000000000001
#define NUM_THREADS 8

__m256i Mand (__m256d _cr, __m256d _ci, __m256i _iterations) {
    __m256d _a,_b,_zi,_zr, _zi2, _zr2, _four, _mask1, _two;
    __m256i _n, _mask2, _one, _c;
    _zr = _mm256_setzero_pd();
    _zi = _mm256_setzero_pd();
    _a = _mm256_setzero_pd();
    _b = _mm256_setzero_pd();
    _mask1 = _mm256_setzero_pd();
    _mask2 = _mm256_setzero_si256();
    _four = _mm256_set1_pd(4.0);
    _one = _mm256_set1_epi64x(1);
    _n = _mm256_set1_epi64x(0);
    _two = _mm256_set1_pd(2.0);

    // cx=initx+initx*initx - inity*inity;   
    // cy=inity+initx*inity+initx*inity; 

    repeat:
        _zr2 = _mm256_mul_pd(_zr, _zr);     // zr * zr
        
        // zi^2 = zi * zi
        _zi2 = _mm256_mul_pd(_zi, _zi);     // zi * zi
        
        // a = zr^2 - zi^2
        _a = _mm256_sub_pd(_zr2, _zi2);     // a = (zr * zr) - (zi * zi)
        // a = a + cr
        _a = _mm256_add_pd(_a, _cr);        // a = ((zr * zr) - (zi * zi)) + cr
        
        // b = zr * zi
        _b = _mm256_mul_pd(_zr, _zi);        // b = zr * zi
        _b = _mm256_fmadd_pd(_b, _two, _ci); // b = (zr * zi) * 2.0 + ci
        
        // zr = a
        _zr = _a;                            // zr = a
        
        // zi = b
        _zi = _b;                            // zr = b        
        
        // a = zr^2 + zi^2
        _a = _mm256_add_pd(_zr2, _zi2);     // a = (zr * zr) + (zi * zi)

        _mask1 = _mm256_cmp_pd(_a, _four, _CMP_LT_OQ); 
        _mask2 = _mm256_cmpgt_epi64(_iterations, _n);  
        _mask2 = _mm256_and_si256(_mask2, _mm256_castpd_si256(_mask1));

        _c = _mm256_and_si256(_one, _mask2);				
        
        _n = _mm256_add_epi64(_n, _c);

        if (_mm256_movemask_pd(_mm256_castsi256_pd(_mask2)) > 0)
            goto repeat;
				
    
    // for (iter=0;iter<iterations && (ysq=cy*cy)+(xsq=cx*cx)<4;iter++,cy=inity+cx*cy+cx*cy,cx=initx-ysq+xsq) ;
    // return (iter);
    return _n;
}

unsigned char colors[4][4] = {{255, 0, 0, 255},{0, 255, 0, 255},{0, 0, 255},{123, 230, 90, 255}};
double dx = -0.0700212907411218; 
double dy = - 0.8224676131988761;

std::queue<int> tasks;

void imagen(){
    __m256i _iter;
    _iter =_mm256_set1_epi64x((long long)(MAX_ITERATIONS));
    while(!tasks.empty()){
        int frame = tasks.front();
        tasks.pop();
        unsigned char* data = (unsigned char*)malloc(HEIGHT*WIDTH*4*sizeof(unsigned char));
        __m256d _xs, _ys;
        double scale = pow(MAX_MAGNIFICATION, (double)frame/FRAMES);
            for(int i = 0; i< WIDTH; i++){
                for(int j = 0; j < HEIGHT; j+=4){
                    _xs = _mm256_set1_pd((((double)i/WIDTH) * scale)-0.0700212907411218);
                    _ys = _mm256_set_pd((((double)(j+3)/HEIGHT) * scale)-0.8224676131988761, (((double)(j+2)/HEIGHT) * scale)-0.8224676131988761,(((double)(j+1)/HEIGHT) * scale)-0.8224676131988761,(((double)(j+0)/HEIGHT) * scale)-0.8224676131988761);
                    // double x = (((double)i/WIDTH) * scale)-0.0700212907411218;
                    // double y = (((double)j/HEIGHT) * scale)-0.8224676131988761;

                    __m256i _res = Mand(_xs, _ys, _iter);
                    for(int idx = 3; idx >= 0; idx--){
                        // data[(WIDTH*(j+idx)+i)*4+0] = 255*sin(_xs[idx]);
                        // data[(WIDTH*(j+idx)+i)*4+1] = 255*sin(_ys[idx]);
                        // data[(WIDTH*(j+idx)+i)*4+2] = 255*0;
                        // data[(WIDTH*(j+idx)+i)*4+3] = 255;
                        if((int)_res[idx] == 1000){
                            data[(WIDTH*(j+idx)+i)*4+0] = 0;
                            data[(WIDTH*(j+idx)+i)*4+1] = 0;
                            data[(WIDTH*(j+idx)+i)*4+2] = 0;
                            data[(WIDTH*(j+idx)+i)*4+3] = 255;
                        }
                        else{
                            float v = sinf(powf((powf((float)_res[idx]/(float)MAX_ITERATIONS, 0.5)*4), 0.5));
                            data[(WIDTH*(j+idx)+i)*4+0] = 255*v;
                            data[(WIDTH*(j+idx)+i)*4+1] = 255*v;
                            data[(WIDTH*(j+idx)+i)*4+2] = 255*v;
                            data[(WIDTH*(j+idx)+i)*4+3] = 255;
                        }
                    }
                }
            }
            char buf[255];
            sprintf(buf, "./frames/%d.png", frame);
            stbi_write_png(buf, WIDTH, HEIGHT, 4, data,WIDTH*4);
            printf("Generated frame: %d @ scale: %lf\n", frame, scale);
        }
    return;
}


int main(){
    int frame = 0;
    for(int i = 0; i < FRAMES; i++){
        tasks.push(i);
    }
    std::vector<std::thread> threads;
    for(int i = 0; i < NUM_THREADS; i++){
        threads.push_back(std::thread(imagen));
        frame ++;
    }
    for(int i = 0 ; i < NUM_THREADS; i++){
        threads[i].join();
    }
    return 0;
}