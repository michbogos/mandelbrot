#include <hip/hip_runtime.h>
#include<hip/hip_complex.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"
#define E (2.7182818284590452353602874713527)
#define HIP_HOST_COHERENT 1

#define SAMPLE_VERSION "HIP-Examples-Application-v1.0"
#define SUCCESS 0
#define FAILURE 1
#define ITERATIONS 5000
#define N_SAMPLES (1024*1024*16)

#define WIDTH  (1080*4)
#define HEIGHT (1920*4)

#define BLOCK 32

using namespace std;

struct color{
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

__device__ struct color color_interp(struct color c1, struct color c2, double t){
    struct color res;
    res.r = (double)c1.r * t + (double)c2.r*(1-t);
    res.g = (double)c1.g * t + (double)c2.g*(1-t);
    res.b = (double)c1.b * t + (double)c2.b*(1-t);
    return res;
}

struct pcg_state_setseq_64 {    // Internals are *Private*.
    uint64_t state;             // RNG state.  All values are possible.
    uint64_t inc;               // Controls which RNG sequence (stream) is
                                // selected. Must *always* be odd.
};
typedef struct pcg_state_setseq_64 pcg32_random_t;

__device__ uint32_t pcg32_random_r(pcg32_random_t* rng)
{
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

__device__ void pcg32_srandom_r(pcg32_random_t* rng, uint64_t initstate, uint64_t initseq)
{
    rng->state = 0U;
    rng->inc = (initseq << 1u) | 1u;
    pcg32_random_r(rng);
    rng->state += initstate;
    pcg32_random_r(rng);
}

__device__ float2 add_imaginary(float2 a, float2 b){
	return a+b;
}

__device__ float2 mul_imaginary(float2 a, float2 b){
	return make_float2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);
}

__device__ float mag_imaginary(float2 a){
	return sqrtf(a.x*a.x+a.y*a.y);
}


__device__ double interpolate(double a, double b, double t){
	return a*(1.0f-t)+b*t;
}

__global__ void MyKernel(float*frame)
{
	int x = threadIdx.x;
	float aspect_ratio = (float)(WIDTH)/(float)(HEIGHT);
	pcg32_random_t rng;
	pcg32_srandom_r(&rng, blockIdx.x*BLOCK+x, threadIdx.x);
	float xcoord = ((float)pcg32_random_r(&rng)/(float)UINT32_MAX)*3.0-1.5;
	float ycoord = ((float)pcg32_random_r(&rng)/(float)UINT32_MAX)*3.0-1.5;
	float2 c = make_float2(xcoord, ycoord);
	float2 i = c;
	bool viable = false;
	for(int n = 0; n < ITERATIONS; n++){
		i = add_imaginary(mul_imaginary(i, i), c);
		if(mag_imaginary(i)>10000.0f){
			viable = true;
			}
		}
	if(!viable)return;

	c = make_float2(xcoord, ycoord);
	i = c;
	for(int n = 0; n < ITERATIONS; n++){
		i = add_imaginary(mul_imaginary(i, i), c);
		if(abs(i.x) < 1.5*((float)WIDTH/HEIGHT) && abs(i.y) < 1.5){
			atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5*aspect_ratio)/(3.0f*aspect_ratio)*WIDTH)))*3+0, 1.0f/ITERATIONS);
			//atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+1, 1.0f);
			//atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+2, 1.0f);
		}
	}
	c = make_float2(xcoord, ycoord);
	i = c;

	viable = false;
	for(int n = 0; n < (int)((float)ITERATIONS*0.01f); n++){
		i = add_imaginary(mul_imaginary(i, i), c);
		if(mag_imaginary(i)>10000.0f){
			viable = true;
			}
		}
	if(!viable)return;

	c = make_float2(xcoord, ycoord);
	i = c;
	for(int n = 0; n < (int)((float)ITERATIONS*0.1f); n++){
		i = add_imaginary(mul_imaginary(i, i), c);
		if(abs(i.x) < 1.5*((float)WIDTH/HEIGHT) && abs(i.y) < 1.5){
			//atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+0, 1.0f);
			atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5*aspect_ratio)/(3.0f*aspect_ratio)*WIDTH)))*3+1, 1.0f/(ITERATIONS*0.1f));
			//atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+2, 1.0f);
		}
	}

	c = make_float2(xcoord, ycoord);
	i = c;

	viable = false;
	for(int n = 0; n < (int)((float)ITERATIONS*0.01f); n++){
		i = add_imaginary(mul_imaginary(i, i), c);
		if(mag_imaginary(i)>10000.0f){
			viable = true;
			}
		}
	if(!viable)return;

	c = make_float2(xcoord, ycoord);
	i = c;
	for(int n = 0; n < (int)((float)ITERATIONS*0.01f); n++){
		i = add_imaginary(mul_imaginary(i, i), c);
		if(abs(i.x) < 1.5*((float)WIDTH/HEIGHT) && abs(i.y) < 1.5){
			//atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+0, 1.0f);
			//atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+1, 1.0f);
			atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5*aspect_ratio)/(3.0f*aspect_ratio)*WIDTH)))*3+2, 1.0f/((float)ITERATIONS*0.01f));
		}
	}
	return;
}

int main()
{
	float* output = (float*)malloc(WIDTH*HEIGHT*3*sizeof(float));

	float* outputBuffer;
	float* deviceData;

	double scale = 2;
	double scale_fac = 0.90;
	float* data = (float*)malloc(BLOCK*N_SAMPLES*2*sizeof(float));

	hipMalloc((void**)&outputBuffer, WIDTH*HEIGHT*3*sizeof(float));
	scale *= scale_fac;

	MyKernel<<<dim3(N_SAMPLES), dim3(BLOCK), 0, 0>>> (outputBuffer);

	hipMemcpy(output, outputBuffer,WIDTH*HEIGHT*3*sizeof(float), hipMemcpyDeviceToHost);
	char buf[255];
	sprintf(buf, "%04d.hdr", 1);
	stbi_write_hdr(buf, WIDTH, HEIGHT, 3, output);
	printf("Rendered frame: %d @ %lf\n", 1, scale);
//}
	hipFree(outputBuffer);
	hipFree(deviceData);
	return SUCCESS;
}