

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

#define WIDTH  (4096*4)
#define HEIGHT (4096*4)

#define BLOCK 32

using namespace std;


__device__ float interpolate(float a, float b, float t){
	return a*(1.0f-t)+b*t;
}

__global__ void MyKernel(unsigned char*frame, float scale, int N)
{
	// for(int x = 0; (x<BLOCK)&&(blockIdx.x*BLOCK+x<HEIGHT);x++){
	// 	for(int y = 0; (y<BLOCK)&&(blockIdx.y*BLOCK+y<WIDTH);y++){
		int x = threadIdx.x;
		int y = threadIdx.y;
	// 			hipDoubleComplex c = make_hipDoubleComplex(((BLOCK*blockIdx.x+x)/(WIDTH*0.5f)-1.0f)*scale-0.0700432019411218, (blockIdx.y/(HEIGHT*0.5f)-1.0f)*scale-0.8224676332988761);
	// hipDoubleComplex i = make_hipDoubleComplex((blockIdx.x/(WIDTH*0.5f)-1.0f)*scale-0.0700432019411218, (blockIdx.y/(HEIGHT*0.5f)-1.0f)*scale-0.8224676332988761);
		hipDoubleComplex c = make_hipDoubleComplex(((((float)blockIdx.x*BLOCK+x)/((float)HEIGHT/255.0))/256.0), (((((float)blockIdx.y*BLOCK+y)/((float)WIDTH/255.0)))/256.0));
		hipDoubleComplex i = c;
		// frame[((HEIGHT*(blockIdx.x*BLOCK+x))+(blockIdx.y*BLOCK+y))*3+0] = (int)(hipCreal(c)*256);
		// frame[((HEIGHT*(blockIdx.x*BLOCK+x))+(blockIdx.y*BLOCK+y))*3+1] = (int)(hipCimag(c)*256);
		// frame[((HEIGHT*(blockIdx.x*BLOCK+x))+(blockIdx.y*BLOCK+y))*3+2] = (int)0;
	// 	}
	// }
	for(int n = 0; n < N; n++){
		i = hipCadd(hipCmul(i, i), c);
		if(hipCabs(i)>10e10f){
			frame[((HEIGHT*(blockIdx.x*BLOCK+x))+(blockIdx.y*BLOCK+y))*3+0] = n;
			frame[((HEIGHT*(blockIdx.x*BLOCK+x))+(blockIdx.y*BLOCK+y))*3+1] = n;
			frame[((HEIGHT*(blockIdx.x*BLOCK+x))+(blockIdx.y*BLOCK+y))*3+2] = n;
			return;
		}
	}
	frame[((HEIGHT*(blockIdx.x*BLOCK+x))+(blockIdx.y*BLOCK+y))*3+0] = 0;
	frame[((HEIGHT*(blockIdx.x*BLOCK+x))+(blockIdx.y*BLOCK+y))*3+1] = 0;
	frame[((HEIGHT*(blockIdx.x*BLOCK+x))+(blockIdx.y*BLOCK+y))*3+2] = 0;
	return;
}

int main(int argc, char* argv[])
{
    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    cout << " System minor " << devProp.minor << endl;
    cout << " System major " << devProp.major << endl;
    cout << " agent prop name " << devProp.name << endl;


	unsigned char *output = (unsigned char*) malloc(WIDTH*HEIGHT*3);

	unsigned char* outputBuffer;
    hipMalloc((void**)&outputBuffer, WIDTH*HEIGHT*3*sizeof(unsigned char));

	MyKernel<<<dim3((WIDTH)/BLOCK, (HEIGHT)/BLOCK), dim3(BLOCK, BLOCK), 0, 0>>> (outputBuffer, 1, 1000);

	hipMemcpy(output, outputBuffer,WIDTH*HEIGHT*3, hipMemcpyDeviceToHost);
	stbi_write_png("hip.png", WIDTH, HEIGHT, 3, output, HEIGHT*3);
    hipFree(outputBuffer);
	return SUCCESS;
}