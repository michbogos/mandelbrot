
#include <iostream>
#include <memory>
#include <vector>

#include <kompute/Kompute.hpp>
#include <shader.hpp>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define WIDTH (1920*2)
#define HEIGHT (1080*2)
#define WARP_SZIE 32


int main() {
  kp::Manager mgr(0, {}, { "VK_EXT_shader_atomic_float" });

  auto tensorOut =
    mgr.tensor(WIDTH*HEIGHT*3, 4, kp::Memory::DataTypes::eFloat);
  
    auto tensorOut2 =
    mgr.tensor(WIDTH*HEIGHT*3, 4, kp::Memory::DataTypes::eFloat);
  

  const std::vector<std::shared_ptr<kp::Memory>> params = { tensorOut };

  const std::vector<uint32_t> shader = std::vector<uint32_t>(
    shader::SHADER_COMP_SPV.begin(), shader::SHADER_COMP_SPV.end());
  std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, shader, {(WIDTH+WARP_SZIE-1)/WARP_SZIE, (HEIGHT+WARP_SZIE-1)/WARP_SZIE, 32}, {WIDTH, HEIGHT}, {WIDTH, HEIGHT});
  

  mgr.sequence()
    ->record<kp::OpSyncDevice>(params)
    ->record<kp::OpAlgoDispatch>(algo, std::vector<uint32_t>{WIDTH, HEIGHT})
    ->record<kp::OpSyncLocal>(params)
    ->eval();
  


  stbi_write_hdr("output.hdr", WIDTH, HEIGHT, 3, tensorOut->vector<float>().data());
}