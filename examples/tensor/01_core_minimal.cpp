#include <iostream>
#include <vix/ai/tensor/Version.hpp>
#include <vix/ai/tensor/Engine.hpp>
#include <vix/ai/tensor/Tensor.hpp>

using namespace vix::ai::tensor;

int main()
{
  std::cout << "vix_ai_tensor version: " << version() << "\n";
  Tensor t({2, 2, 2});
  Engine e{Device::from_string("cpu")};
  std::cout << e.compute(t) << "\n";
}
