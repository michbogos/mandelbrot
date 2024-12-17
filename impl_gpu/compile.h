#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

static constexpr auto kernel_source{
    R"(
    extern "C"
    __global__ void vector_add(float* output, float* input1, float* input2, size_t size) {
      int i = threadIdx.x;
      if (i < size) {
        output[i] = input1[i] + input2[i];
      }
    }
)"};