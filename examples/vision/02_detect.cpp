#include <iostream>
#include <vix/ai/vision/Detector.hpp>
using namespace vix::ai::vision;

int main()
{
  Detector det;
  auto boxes = det.detect(Image{320, 240});
  std::cout << "detected boxes: " << boxes.size() << "\n";
  for (auto &b : boxes)
    std::cout << "bbox: " << b[0] << "," << b[1] << "," << b[2] << "," << b[3] << "\n";
}
