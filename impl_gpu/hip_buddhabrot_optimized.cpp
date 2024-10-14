#include <hip/hip_runtime.h>
#include<hip/hip_complex.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <time.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"
#define E (2.7182818284590452353602874713527)

#define HIP_HOST_COHERENT 1

#define SAMPLE_VERSION "HIP-Examples-Application-v1.0"
#define SUCCESS 0
#define FAILURE 1
#define ITERATIONS 5000
#define N_SAMPLES (1024*1024)

#define WIDTH  (1024)
#define HEIGHT (1024)

#define BLOCK 32ll

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

__global__ void RenderTemplate(float* __restrict__ frame, float scale, float dx, float dy, int N)
{
		int x = threadIdx.x;
		int y = threadIdx.y;
		if(((HEIGHT*(blockIdx.x*BLOCK+x))+(blockIdx.y*BLOCK+y))*3+0 >= WIDTH*HEIGHT*3*sizeof(float)){
			return;
		}
		hipDoubleComplex c = make_hipDoubleComplex(dx+((((double)(blockIdx.x*BLOCK)+x)/((double)WIDTH)))*scale, (dy+(((((double)blockIdx.y*BLOCK+y)/((double)HEIGHT))))*scale)/((float)WIDTH/(float)HEIGHT));
		hipDoubleComplex i = c;
	for(int n = 0; n < N; n++){
		i = hipCadd(hipCmul(i, i), c);
		if(hipCabs(i)>10e10f){
			frame[((WIDTH*(blockIdx.y*BLOCK+y))+(blockIdx.x*BLOCK+x))*3+0] = 1.0f;
			frame[((WIDTH*(blockIdx.y*BLOCK+y))+(blockIdx.x*BLOCK+x))*3+1] = 1.0f;
			frame[((WIDTH*(blockIdx.y*BLOCK+y))+(blockIdx.x*BLOCK+x))*3+2] = 1.0f;
			return;
		}
	}
	frame[((WIDTH*(blockIdx.y*BLOCK+y))+(blockIdx.x*BLOCK+x))*3+0] = 0;
	frame[((WIDTH*(blockIdx.y*BLOCK+y))+(blockIdx.x*BLOCK+x))*3+1] = 0;
	frame[((WIDTH*(blockIdx.y*BLOCK+y))+(blockIdx.x*BLOCK+x))*3+2] = 0;
	return;
}

__global__ void FindEdges(float* __restrict__ frame, int* num_points){
	int x = threadIdx.x;
	int y = threadIdx.y;
	if(!(blockIdx.x*BLOCK+x > 1 && blockIdx.x*BLOCK+x < WIDTH-2 && blockIdx.y*BLOCK+y > 1 && blockIdx.y*BLOCK+y < HEIGHT-2)){
		return;
	}
	if(!(x%BLOCK)){
		// frame[((WIDTH*(blockIdx.y*BLOCK+y))+(blockIdx.x*BLOCK+x))*3+0] = 0.0f;
		// frame[((WIDTH*(blockIdx.y*BLOCK+y))+(blockIdx.x*BLOCK+x))*3+1] = 0.0f;
		// frame[((WIDTH*(blockIdx.y*BLOCK+y))+(blockIdx.x*BLOCK+x))*3+2] = 0.0f;
		return;
	}
	if(!(y%BLOCK)){
		// frame[((WIDTH*(blockIdx.y*BLOCK+y))+(blockIdx.x*BLOCK+x))*3+0] = 0.0f;
		// frame[((WIDTH*(blockIdx.y*BLOCK+y))+(blockIdx.x*BLOCK+x))*3+1] = 0.0f;
		// frame[((WIDTH*(blockIdx.y*BLOCK+y))+(blockIdx.x*BLOCK+x))*3+2] = 0.0f;
		return;
	}
	frame[((WIDTH*(blockIdx.y*BLOCK+y))+(blockIdx.x*BLOCK+x))*3] = 	1*frame[((WIDTH*(blockIdx.y*BLOCK+y-1))+(blockIdx.x*BLOCK+x-1))*3]+
																   	2*frame[((WIDTH*(blockIdx.y*BLOCK+y+0))+(blockIdx.x*BLOCK+x-1))*3]+
																   	1*frame[((WIDTH*(blockIdx.y*BLOCK+y+1))+(blockIdx.x*BLOCK+x-1))*3]+
																   -1*frame[((WIDTH*(blockIdx.y*BLOCK+y-1))+(blockIdx.x*BLOCK+x+1))*3]+
																   -2*frame[((WIDTH*(blockIdx.y*BLOCK+y+0))+(blockIdx.x*BLOCK+x+1))*3]+
																   -1*frame[((WIDTH*(blockIdx.y*BLOCK+y+1))+(blockIdx.x*BLOCK+x+1))*3];
	frame[((WIDTH*(blockIdx.y*BLOCK+y))+(blockIdx.x*BLOCK+x))*3+1] = 0.0f;
	frame[((WIDTH*(blockIdx.y*BLOCK+y))+(blockIdx.x*BLOCK+x))*3+2] = 0.0f;

	if(frame[((WIDTH*(blockIdx.y*BLOCK+y))+(blockIdx.x*BLOCK+x))*3] > 0.0f){
		atomicAdd(&(num_points[0]), 1);
	}
}

__global__ void RenderFractal(float* __restrict__ frame, float* __restrict__ points, int n_points)
{
	int x = threadIdx.x;
	float aspect_ratio = (float)(WIDTH)/(float)(HEIGHT);
	pcg32_random_t rng;
	pcg32_srandom_r(&rng, blockIdx.x*BLOCK+x, threadIdx.x);
	int point_idx = pcg32_random_r(&rng)%n_points;
	// float xcoord = ((float)pcg32_random_r(&rng)/(float)UINT32_MAX)*3.0-1.5;
	// float ycoord = ((float)pcg32_random_r(&rng)/(float)UINT32_MAX)*3.0-1.5;
	float xcoord = points[2*point_idx]+((float)pcg32_random_r(&rng)/(float)UINT32_MAX)-0.5;
	float ycoord = points[2*point_idx+1]+((float)pcg32_random_r(&rng)/(float)UINT32_MAX)-0.5;
	float2 c = make_float2(xcoord, ycoord);
	float2 i = c;
	bool viable1 = false;
	bool viable2 = false;
	bool viable3 = false;
	for(int n = 0; n < ITERATIONS*0.01; n++){
		i = add_imaginary(mul_imaginary(i, i), c);
		if(mag_imaginary(i)>10000.0f){
			viable1 = true;
			viable2 = true;
			viable3 = true;
			}
		}
	for(int n = ITERATIONS*0.01; n < ITERATIONS*0.1 && !viable1; n++){
		i = add_imaginary(mul_imaginary(i, i), c);
		if(mag_imaginary(i)>10000.0f){
			viable2 = true;
			viable3 = true;
			}
		}
	for(int n = ITERATIONS*0.1; n < ITERATIONS && !viable2; n++){
		i = add_imaginary(mul_imaginary(i, i), c);
		if(mag_imaginary(i)>10000.0f){
			viable3 = true;
			}
		}
	if(!viable3) return;

	c = make_float2(xcoord, ycoord);
	i = c;

	if(viable1){
		for(int n = 0; n < (int)((float)ITERATIONS*0.01f); n++){
			i = add_imaginary(mul_imaginary(i, i), c);
			if(abs(i.x) < 1.5*((float)WIDTH/HEIGHT) && abs(i.y) < 1.5){
				//atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+0, 1.0f);
				//atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+1, 1.0f);
				atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5*aspect_ratio)/(3.0f*aspect_ratio)*WIDTH)))*3+2, 1.0f/((float)ITERATIONS*0.01f));
			}
		}
		for(int n = ITERATIONS*0.01f; n < ITERATIONS*0.1f; n++){
			i = add_imaginary(mul_imaginary(i, i), c);
			if(abs(i.x) < 1.5*((float)WIDTH/HEIGHT) && abs(i.y) < 1.5){
				atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5*aspect_ratio)/(3.0f*aspect_ratio)*WIDTH)))*3+0, 1.0f/ITERATIONS);
				//atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+1, 1.0f);
				//atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+2, 1.0f);
			}
		}
		for(int n = ITERATIONS*0.1; n < (int)((float)ITERATIONS); n++){
			i = add_imaginary(mul_imaginary(i, i), c);
			if(abs(i.x) < 1.5*((float)WIDTH/HEIGHT) && abs(i.y) < 1.5){
				//atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+0, 1.0f);
				atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5*aspect_ratio)/(3.0f*aspect_ratio)*WIDTH)))*3+1, 1.0f/(ITERATIONS*0.1f));
				//atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+2, 1.0f);
			}
		}
		return;
	}
	if(viable2){
		for(int n = 0; n < ITERATIONS*0.1f; n++){
			i = add_imaginary(mul_imaginary(i, i), c);
			if(abs(i.x) < 1.5*((float)WIDTH/HEIGHT) && abs(i.y) < 1.5){
				atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5*aspect_ratio)/(3.0f*aspect_ratio)*WIDTH)))*3+0, 1.0f/ITERATIONS);
				//atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+1, 1.0f);
				//atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+2, 1.0f);
			}
		}
		for(int n = ITERATIONS*0.1; n < (int)((float)ITERATIONS); n++){
			i = add_imaginary(mul_imaginary(i, i), c);
			if(abs(i.x) < 1.5*((float)WIDTH/HEIGHT) && abs(i.y) < 1.5){
				//atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+0, 1.0f);
				atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5*aspect_ratio)/(3.0f*aspect_ratio)*WIDTH)))*3+1, 1.0f/(ITERATIONS*0.1f));
				//atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+2, 1.0f);
			}
		}
		return;
	}
	if(viable3){
		for(int n = 0; n < (int)((float)ITERATIONS); n++){
			i = add_imaginary(mul_imaginary(i, i), c);
			if(abs(i.x) < 1.5*((float)WIDTH/HEIGHT) && abs(i.y) < 1.5){
				//atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+0, 1.0f);
				atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5*aspect_ratio)/(3.0f*aspect_ratio)*WIDTH)))*3+1, 1.0f/(ITERATIONS*0.1f));
				//atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+2, 1.0f);
			}
		}
		return;
	}
	return;
}

int main()
{
	float* output = (float*)malloc(WIDTH*HEIGHT*3*sizeof(float));

	float* outputBuffer;
	float* fractalBuffer;
	hipMalloc((void**)&fractalBuffer, WIDTH*HEIGHT*3*sizeof(float));
	float* devicePoints;

	double scale = 2;
	double scale_fac = 0.90;
	int* dnum_points;
	int num_points = 0;

	hipMalloc((void**)&dnum_points, sizeof(int));
	hipMemcpy(dnum_points, &num_points, sizeof(int), hipMemcpyHostToDevice);

	clock_t tic = clock();

	hipMalloc((void**)&outputBuffer, WIDTH*HEIGHT*3*sizeof(float));
	scale *= scale_fac;

	RenderTemplate<<<dim3((WIDTH+BLOCK-1)/BLOCK, (HEIGHT+BLOCK-1)/BLOCK), dim3(BLOCK, BLOCK), 0, 0>>> (outputBuffer, 4.0f, -2.5, -2.0, 100);
	FindEdges<<<dim3((WIDTH+BLOCK-1)/BLOCK, (HEIGHT+BLOCK-1)/BLOCK), dim3(BLOCK, BLOCK), 0, 0>>> (outputBuffer, dnum_points);

	hipMemcpy(&num_points, dnum_points, sizeof(int), hipMemcpyDeviceToHost);
	hipMemcpy(output, outputBuffer,WIDTH*HEIGHT*3*sizeof(float), hipMemcpyDeviceToHost);
	printf("Found: %d\n", num_points);

	float* points = (float*)malloc(2*sizeof(float)*num_points);
	int point_idx = 0;

	for(int j = 0; j < HEIGHT; j++){
		for(int i = 0 ; i < WIDTH; i++){
			if(output[3*((j*WIDTH)+i)] > 0.0f && output[3*((j*WIDTH)+i)+1] < 1.0f){
				points[2*point_idx+0] = (-2.5+((float)i/(float)WIDTH*4))*1.1;
				points[2*point_idx+1] = (-2.0+((float)j/(float)HEIGHT*4)*((float)WIDTH/(float)HEIGHT))*1.1;
				printf("%f %f\n", -2.5+((float)i/(float)WIDTH*4), -2.0+((float)j/(float)HEIGHT*4)/((float)WIDTH/(float)HEIGHT));
			}
		}
	}

	hipFree(outputBuffer);
	hipMalloc((void**)&devicePoints, 2*num_points*sizeof(float));
	hipMemcpy(devicePoints, points, 2*num_points*sizeof(float), hipMemcpyHostToDevice);
	RenderFractal<<<dim3(N_SAMPLES), dim3(BLOCK), 0, 0>>> (fractalBuffer, devicePoints, num_points);
	float* fractal = (float*)malloc(3*WIDTH*HEIGHT*sizeof(float));
	hipMemcpy(fractal, fractalBuffer,WIDTH*HEIGHT*3*sizeof(float), hipMemcpyDeviceToHost);
	clock_t toc = clock();

	printf("Kernel finished in: %fs\n", (double)(toc-tic)/CLOCKS_PER_SEC);
	char buf[255];
	sprintf(buf, "%04d.hdr", 1);
	tic = clock();
	if(stbi_write_hdr(buf, WIDTH, HEIGHT, 3, fractal)){
		printf("Rendered frame: %d @ %lf\n", 1, scale);
	}
	toc = clock();
	printf("Image written in: %fs\n", (double)(toc-tic)/CLOCKS_PER_SEC);
//}
	hipFree(devicePoints);
	return SUCCESS;
}