#include <hip/hip_runtime.h>
#include<hip/hip_complex.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define SAMPLE_VERSION "HIP-Examples-Application-v1.0"
#define SUCCESS 0
#define FAILURE 1

#define WIDTH 512
#define HEIGHT 512

using namespace std;


__device__ float interpolate(float a, float b, float t){
	return a*(1.0f-t)+b*t;
}

__global__ void MyKernel(unsigned char*frame, float scale, int N)
{
	float colors[6][3] = {{255, 156, 87},{87, 165, 187},{198, 98, 123},{76, 71, 237},{143, 97, 198},{198, 34, 243}};
	hipDoubleComplex c = make_hipDoubleComplex((blockIdx.x/256.0f-1.0f)*scale-0.0700432019411218, (blockIdx.y/256.0f-1.0f)*scale-0.8224676332988761);
	hipDoubleComplex i = make_hipDoubleComplex((blockIdx.x/256.0f-1.0f)*scale-0.0700432019411218, (blockIdx.y/256.0f-1.0f)*scale-0.8224676332988761);
	for(int n = 0; n < N; n++){
		i = hipCadd(hipCmul(i, i), c);
		if(hipCabs(i)>10e10f){
			frame[(512*blockIdx.x+blockIdx.y)*3+0] = (int)exp(sqrt(sqrt(n)));
			frame[(512*blockIdx.x+blockIdx.y)*3+1] = (int)exp(sqrt(sqrt(n)));
			frame[(512*blockIdx.x+blockIdx.y)*3+2] = (int)exp(sqrt(sqrt(n)));
			return;
		}
	}
	frame[(512*blockIdx.x+blockIdx.y)*3+0] = 0;
	frame[(512*blockIdx.x+blockIdx.y)*3+1] = 0;
	frame[(512*blockIdx.x+blockIdx.y)*3+2] = 0;
	return;
}

int main(int argc, char* argv[])
{
    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    cout << " System minor " << devProp.minor << endl;
    cout << " System major " << devProp.major << endl;
    cout << " agent prop name " << devProp.name << endl;


	unsigned char *output = (unsigned char*) malloc(512*512*3);

	unsigned char* outputBuffer;
    hipMalloc((void**)&outputBuffer, 512*512*3*sizeof(unsigned char));

	MyKernel<<<dim3(512, 512), dim3(1, 1), 0, 0>>> (outputBuffer, 10e-14, 1000);

	hipMemcpy(output, outputBuffer,512*512*3, hipMemcpyDeviceToHost);
	stbi_write_png("hip.png", 512, 512, 3, output, 512*3);
    hipFree(outputBuffer);
	return SUCCESS;
}