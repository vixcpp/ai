<table>
<tr>
<td valign="top" width="70%">

<h1>Vix.AI</h1>

<p>
  <img src="https://img.shields.io/badge/C++20-Standard-blue">
  <img src="https://img.shields.io/github/stars/vixcpp/ai?style=flat">
  <img src="https://img.shields.io/github/forks/vixcpp/ai?style=flat">
  <img src="https://img.shields.io/badge/License-MIT-green">
</p>

<p>
<b>Vix.AI</b> is a modern <b>C++ AI framework</b> for machine learning,
deep learning, NLP, and computer vision — designed for
<b>native performance</b>, <b>modularity</b>, and
<b>production-grade reliability</b>.
</p>

<p>
Part of the <b>Vix.cpp ecosystem</b>, bringing
<b>Python-like AI workflows</b> to <b>high-performance C++</b>.
</p>

<p>
📘 <a href="https://vixcpp.com/docs">Documentation</a><br>
🌍 <a href="https://vixcpp.com">vixcpp.com</a>
</p>

</td>

<td valign="middle" width="30%" align="right">
<img
src="https://res.cloudinary.com/dwjbed2xb/image/upload/v1762524349/vixai_snfafp.png"
alt="Vix.AI Logo"
width="200"
style="border-radius:50%;"
/>
</td>
</tr>
</table>

## AI at Native Speed

Most AI tools rely on Python layers and runtime overhead.

**Vix.AI** is designed to bring **AI workloads directly into modern C++**,
with predictable performance and full control over memory and execution.

Key goals:

- ⚡ Native performance
- 🧩 Modular architecture
- 🧠 Production-grade ML & Deep Learning
- 🌐 Distributed AI systems
- 🛠 Seamless integration with **Vix.cpp**

# Modular Architecture

Vix.AI is composed of independent modules.

| Module | Description |
|------|-------------|
| **core** | Tensor primitives, device abstraction, memory |
| **ml** | Classical ML algorithms |
| **nn** | Neural network layers and optimizers |
| **nlp** | Natural language processing |
| **vision** | Computer vision utilities |
| **distributed** | Distributed AI training |

All modules are developed as **independent repositories** and linked using **Git submodules**.

# Installation

Clone the repository with all modules:

```bash
git clone --recurse-submodules https://github.com/vixcpp/ai.git
cd ai
```

## Build the framework:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run tests:
```bash
cd build
ctest --output-on-failure
```

## Your First Vix.AI Program

### Example: Linear Regression
```cpp
#include <vix/ai/ml/Regression.hpp>
#include <iostream>

using namespace vix::ai::ml;

int main() {

  Mat X = {
    {1}, {2}, {3}, {4}, {5}
  };

  Vec y = {
    3, 5, 7, 9, 11
  };

  LinearRegression lr;
  lr.fit(X, y);

  auto pred = lr.predict({{6}});

  std::cout << "Prediction: " << pred[0] << "\n";
}
```

### Example: Clustering
```cpp
#include <vix/ai/ml/Clustering.hpp>

using namespace vix::ai::ml;

int main() {

  Mat data = {
    {1,1},{1.2,1.1},{5,5},{5.2,4.9}
  };

  KMeans km(2);
  km.fit(data);

}
```

## Roadmap

| Phase | Focus |
|------|------|
| ✅ Phase 1 | Core ML algorithms |
| 🚧 Phase 2 | Neural networks |
| 🔜 Phase 3 | NLP & Vision |
| 🌍 Phase 4 | Distributed AI |
| 💫 Phase 5 | Unified AI Runtime |

## Part of the Vix Ecosystem

| Project | Description |
|--------|-------------|
| Vix.cpp | High-performance backend runtime |
| Vix.AI | Artificial intelligence framework |
| Vix.ORM | Modern C++ ORM |
| Vix.CLI | Developer CLI tools |

## Contributing

Contributions are welcome.

If you're interested in high-performance AI systems in C++,
you’ll feel at home here.

Please read the contributing guide before opening a PR.

⭐ If this project resonates with you, consider starring the repository.

## License

MIT License
