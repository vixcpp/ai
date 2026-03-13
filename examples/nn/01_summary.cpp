#include <iostream>
#include <vix/ai/nn/Network.hpp>

using namespace vix::ai::nn;

int main()
{
  Network net;
  net.add_layer("Dense(8->16)");
  net.add_layer("Conv2D(k=3)");
  net.add_layer("Transformer(h=8)");
  std::cout << net.summary() << "\n";
}
