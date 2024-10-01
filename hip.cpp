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
#define E (2.7182818284590452353602874713527)

#define SAMPLE_VERSION "HIP-Examples-Application-v1.0"
#define SUCCESS 0
#define FAILURE 1

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


__device__ double interpolate(double a, double b, double t){
	return a*(1.0f-t)+b*t;
}

__global__ void MyKernel(unsigned char*frame, double scale, double dx, double dy ,int N)
{
	color colors[2] = {(struct color){10, 254, 176}, (struct color){56, 12, 86}};
	// for(int x = 0; (x<BLOCK)&&(blockIdx.x*BLOCK+x<HEIGHT);x++){
	// 	for(int y = 0; (y<BLOCK)&&(blockIdx.y*BLOCK+y<WIDTH);y++){
		int x = threadIdx.x;
		int y = threadIdx.y;
	// 			hipDoubleComplex c = make_hipDoubleComplex(((BLOCK*blockIdx.x+x)/(WIDTH*0.5f)-1.0f)*scale-0.0700432019411218, (blockIdx.y/(HEIGHT*0.5f)-1.0f)*scale-0.8224676332988761);
	// hipDoubleComplex i = make_hipDoubleComplex((blockIdx.x/(WIDTH*0.5f)-1.0f)*scale-0.0700432019411218, (blockIdx.y/(HEIGHT*0.5f)-1.0f)*scale-0.8224676332988761);
		hipDoubleComplex c = make_hipDoubleComplex(dx+((((double)(blockIdx.x*BLOCK)+x)/((double)HEIGHT/255.0))/255.0)*scale, dy+(((((double)blockIdx.y*BLOCK+y)/((double)WIDTH/255.0)))/255.0)*scale);
		hipDoubleComplex i = c;
		// frame[((HEIGHT*(blockIdx.x*BLOCK+x))+(blockIdx.y*BLOCK+y))*3+0] = (int)(hipCreal(c)*256);
		// frame[((HEIGHT*(blockIdx.x*BLOCK+x))+(blockIdx.y*BLOCK+y))*3+1] = (int)(hipCimag(c)*256);
		// frame[((HEIGHT*(blockIdx.x*BLOCK+x))+(blockIdx.y*BLOCK+y))*3+2] = (int)0;
	// 	}
	// }
	for(int n = 0; n < N; n++){
		i = hipCadd(hipCmul(i, i), c);
		if(hipCabs(i)>10e10f){
			double logzn = logf(hipCabs(i))/2.0f;
			double nu = logf(logzn / logf(2)) / logf(2);
			double iter = (double)n+1-nu;
			struct color col = color_interp(colors[((int)floorf(iter))%2], colors[(((int)floorf(iter))+1)%2], iter-floorf(iter));
			frame[((HEIGHT*(blockIdx.x*BLOCK+x))+(blockIdx.y*BLOCK+y))*3+0] = col.r;
			frame[((HEIGHT*(blockIdx.x*BLOCK+x))+(blockIdx.y*BLOCK+y))*3+1] = col.g;
			frame[((HEIGHT*(blockIdx.x*BLOCK+x))+(blockIdx.y*BLOCK+y))*3+2] = col.b;
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

	MyKernel<<<dim3((WIDTH)/BLOCK, (HEIGHT)/BLOCK), dim3(BLOCK, BLOCK), 0, 0>>> (outputBuffer, 4, -3, -2, 1000);

	hipMemcpy(output, outputBuffer,WIDTH*HEIGHT*3, hipMemcpyDeviceToHost);
	stbi_write_png("hip.png", WIDTH, HEIGHT, 3, output, HEIGHT*3);
    hipFree(outputBuffer);
	return SUCCESS;
}