#include <iostream>
#include <vix/ai/tensor/Engine.hpp>

using namespace vix::ai::tensor;

int main()
{
  Engine e{Device::from_string("cpu")};
  std::cout << e.info() << "\n";
}
