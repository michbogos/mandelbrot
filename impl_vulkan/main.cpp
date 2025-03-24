
#include <iostream>
#include <memory>
#include <vector>

#include <kompute/Kompute.hpp>
#include <shader.hpp>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>


int
main()
{
    kp::Manager mgr;

   auto tensorOut =
      mgr.tensor(512*512*3, 4, kp::Memory::DataTypes::eFloat);

    const std::vector<std::shared_ptr<kp::Memory>> params = { tensorOut };

    const std::vector<uint32_t> shader = std::vector<uint32_t>(
      shader::SHADER_COMP_SPV.begin(), shader::SHADER_COMP_SPV.end());
    std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, shader, {512, 512, 1});

    mgr.sequence()
      ->record<kp::OpSyncDevice>(params)
      ->record<kp::OpAlgoDispatch>(algo)
      ->record<kp::OpSyncLocal>(params)
      ->eval();

      stbi_write_hdr("output.hdr", 512, 512, 3, tensorOut->vector<float>().data());

    // // prints "Output {  0  4  12  }"
    // std::cout << "Output: {  ";
    // for (const float& elem : tensorOut->vector()) {
    //     std::cout << elem << "  ";
    // }
    // std::cout << "}" << std::endl;

    // if (tensorOut->vector() != std::vector<float>{ 0, 4, 12 }) {
    //     throw std::runtime_error("Result does not match");
    // }
}