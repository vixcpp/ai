#include <iostream>
#include <vix/ai/tensor/Engine.hpp>
#include <vix/ai/tensor/Version.hpp>

int main()
{
  using namespace vix::ai::core;
  std::cout << "vix-ai-core " << version() << "\n";

  Tensor t = Tensor::ones({2, 3});
  Engine e{Device::from_string("cpu")};
  std::cout << e.info() << "\n";
  std::cout << e.compute(t) << "\n";
  return 0;
}
