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

__global__ void MyKernel(unsigned char*frame, int N)
{
	hipComplex c = make_hipComplex(blockIdx.x/256.0f-1.0f-0.0700432019411218f, blockIdx.y/256.0f-1.0f-0.8224676332988761f);
	hipComplex i = make_hipComplex(blockIdx.x/256.0f-1.0f-0.0700432019411218f, blockIdx.y/256.0f-1.0f-0.8224676332988761f);
	for(int n = 0; n < N; n++){
		i = hipCadd(hipCmul(i, i), c);
	}
	if(hipCabs(i)<100.0f){
		frame[(512*blockIdx.x+blockIdx.y)*3+0] = 255;
		frame[(512*blockIdx.x+blockIdx.y)*3+1] = 255;
		frame[(512*blockIdx.x+blockIdx.y)*3+2] = 255;
	}
	else{
		frame[(512*blockIdx.x+blockIdx.y)*3+0] = 0;
		frame[(512*blockIdx.x+blockIdx.y)*3+1] = 0;
		frame[(512*blockIdx.x+blockIdx.y)*3+2] = 0;
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


	unsigned char *output = (unsigned char*) malloc(512*512*4);

	unsigned char* outputBuffer;
    hipMalloc((void**)&outputBuffer, 512*512*4*sizeof(unsigned char));

	MyKernel<<<dim3(512, 512), dim3(1, 1), 0, 0>>> (outputBuffer, 1000, );

	hipMemcpy(output, outputBuffer,512*512*4, hipMemcpyDeviceToHost);
	stbi_write_png("hip.png", 512, 512, 3, output, 512*3);
    hipFree(outputBuffer);
	return SUCCESS;
}