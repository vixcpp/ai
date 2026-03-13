#include <iostream>
#include <vix/ai/tensor/Tensor.hpp>
#include <vix/ai/tensor/Engine.hpp>
#include <vix/ai/tensor/Device.hpp>

int main()
{
  vix::ai::tensor::Device dev;
  vix::ai::tensor::Engine eng(dev);
  vix::ai::tensor::Tensor t({2, 3});
  t.fill(1.0f);
  std::cout << eng.compute(t) << "\n";
  return 0;
}
