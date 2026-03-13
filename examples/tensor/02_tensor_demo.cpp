#include <iostream>
#include <vix/ai/tensor/Tensor.hpp>
#include <vix/ai/tensor/Engine.hpp>

using namespace vix::ai::tensor;

int main()
{
  Tensor t = Tensor::ones({3, 3});
  Engine e{Device::from_string("cpu")};
  std::cout << "Demo: Tensor::ones 3x3\n";
  std::cout << e.compute(t) << "\n";

  t.reshape({1, 9});
  std::cout << "After reshape to 1x9 -> " << e.compute(t) << "\n";
}
