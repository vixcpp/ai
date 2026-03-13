#include <iostream>
#include <vix/ai/vision/Image.hpp>

using namespace vix::ai::vision;

int main()
{
  Image img(1280, 720);
  std::cout << "Image: " << img.width() << "x" << img.height()
            << " area=" << img.area() << "\n";
}
