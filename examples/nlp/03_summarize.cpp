#include <iostream>
#include <vix/ai/nlp/Pipeline.hpp>

using namespace vix::ai::nlp;

int main()
{
  Pipeline p;
  std::string s = "Mini demo text.";
  std::cout << "summary: " << p.summarize(s) << "\n";
}
