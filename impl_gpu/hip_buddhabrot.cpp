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

#define SAMPLE_VERSION "HIP-Examples-Application-v1.0"
#define SUCCESS 0
#define FAILURE 1
#define ITERATIONS 100
#define N_SAMPLES 1024*31

#define WIDTH  (512)
#define HEIGHT (512)

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

__global__ void MyKernel(float*frame, float* data)
{
	int x = threadIdx.x;
	float2 c = make_float2(data[2*blockIdx.x*BLOCK+x],data[2*blockIdx.x*BLOCK+x+1]);
	float2 i = c;
	bool viable = false;
	for(int n = 0; n < ITERATIONS; n++){
		i = add_imaginary(mul_imaginary(i, i), c);
		if(mag_imaginary(i)>10000.0f){
			viable = true;
			}
		}
	if(!viable)return;

	c = make_float2(data[2*blockIdx.x*BLOCK+x],data[2*blockIdx.x*BLOCK+x+1]);
	i = c;
	for(int n = 0; n < ITERATIONS; n++){
		i = add_imaginary(mul_imaginary(i, i), c);
		if(abs(i.x) < 1.5 && abs(i.y) < 1.5){
			atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+0, 1.0f/log(N_SAMPLES));
			//atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+1, 1.0f);
			//atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+2, 1.0f);
		}
	}
	c = make_float2(data[2*blockIdx.x*BLOCK+x],data[2*blockIdx.x*BLOCK+x+1]);
	i = c;

	viable = false;
	for(int n = 0; n < ITERATIONS-50; n++){
		i = add_imaginary(mul_imaginary(i, i), c);
		if(mag_imaginary(i)>10000.0f){
			viable = true;
			}
		}
	if(!viable)return;

	c = make_float2(data[2*blockIdx.x*BLOCK+x],data[2*blockIdx.x*BLOCK+x+1]);
	i = c;
	for(int n = 0; n < ITERATIONS-50; n++){
		i = add_imaginary(mul_imaginary(i, i), c);
		if(abs(i.x) < 1.5 && abs(i.y) < 1.5){
			//atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+0, 1.0f);
			atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+1, 1.0f/log(N_SAMPLES));
			//atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+2, 1.0f);
		}
	}

	c = make_float2(data[2*blockIdx.x*BLOCK+x],data[2*blockIdx.x*BLOCK+x+1]);
	i = c;

	viable = false;
	for(int n = 0; n < ITERATIONS-80; n++){
		i = add_imaginary(mul_imaginary(i, i), c);
		if(mag_imaginary(i)>10000.0f){
			viable = true;
			}
		}
	if(!viable)return;

	c = make_float2(data[2*blockIdx.x*BLOCK+x],data[2*blockIdx.x*BLOCK+x+1]);
	i = c;
	for(int n = 0; n < ITERATIONS-80; n++){
		i = add_imaginary(mul_imaginary(i, i), c);
		if(abs(i.x) < 1.5 && abs(i.y) < 1.5){
			//atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+0, 1.0f);
			//atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+1, 1.0f);
			atomicAdd((float*)frame+((int)(((i.y+1.5)/3.0f)*HEIGHT)*WIDTH + (int)(((i.x+1.5)/3.0f)*WIDTH))*3+2, 1.0f/log(N_SAMPLES));
		}
	}
	return;
}

int main(int argc, char* argv[])
{
    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    cout << " System minor " << devProp.minor << endl;
    cout << " System major " << devProp.major << endl;
    cout << " agent prop name " << devProp.name << endl;


	float *output = (float*) malloc(WIDTH*HEIGHT*3*sizeof(float));

	float* outputBuffer;
	float* deviceData;

	double scale = 2;
	double scale_fac = 0.90;
	float data[BLOCK*N_SAMPLES*2];

	for(int i = 0; i < BLOCK*N_SAMPLES*2; i++){
		data[i]=((float)rand()/(float)(RAND_MAX))*3.0f-1.5f;
	}

	hipMalloc((void**)&outputBuffer, WIDTH*HEIGHT*3*sizeof(float));
	hipMalloc((void**)&deviceData, N_SAMPLES*2*BLOCK*sizeof(float));
	hipMemcpy(deviceData, data, 2*BLOCK*N_SAMPLES*sizeof(float), hipMemcpyHostToDevice);

	//for(int i =0; i < (int)(floor(log(0.0000000000001)/log(scale_fac))); i++){
	scale *= scale_fac;

	MyKernel<<<dim3(N_SAMPLES), dim3(BLOCK), 0, 0>>> (outputBuffer, deviceData);

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