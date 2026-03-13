#include <iostream>
#include <vix/ai/nlp/Tokenizer.hpp>

using namespace vix::ai::nlp;

int main()
{
  Tokenizer tok;
  auto v = tok.tokenize("Hello Vix AI world!");
  std::cout << "tokens=" << v.size() << "\n";
  for (auto &t : v)
    std::cout << "- " << t << "\n";
}
