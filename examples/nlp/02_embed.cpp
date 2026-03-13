#include <iostream>
#include <vix/ai/nlp/Embedding.hpp>

using namespace vix::ai::nlp;

int main()
{
  Embedding emb;
  auto e = emb.encode("vix");
  std::cout << "embedding_dim=" << e.size() << " val=" << e[0] << "\n";
}
