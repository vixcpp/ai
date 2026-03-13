#include <iostream>
#include <vix/ai/vision/Tracker.hpp>
using namespace vix::ai::vision;

int main()
{
  Tracker tr;
  for (int i = 0; i < 3; ++i)
  {
    auto fid = tr.update(Image{64, 64});
    std::cout << "frame=" << fid << "\n";
  }
}
