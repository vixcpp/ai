#include <iostream>
#include <vector>
#include <vix/ai/tensor/Device.hpp>

using namespace vix::ai::tensor;

int main()
{
  std::vector<std::string> devs = {"cpu", "cuda", "cuda:1"};
  for (auto &s : devs)
  {
    Device d = Device::from_string(s);
    std::cout << "Parsed \"" << s << "\" -> " << d.name() << "\n";
  }
}
