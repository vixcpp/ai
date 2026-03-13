# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Aggregation of all Vix AI modules:
  - ML: datasets, regression, clustering
  - Tensor: device/tensor/engine primitives
  - NN: neural network layers, training, inference
  - NLP: tokenization, text preprocessing, sequence models
  - Vision: image loading, transformations, simple models
  - Distributed: P2P networking, distributed computation
- Unified CMake build system for AI submodules.
- Examples demonstrating integration between modules (e.g., distributed training with NN and Tensor).
- Unit tests across all submodules for interoperability.

### Changed
- N/A

### Fixed
- N/A

---

## [0.1.0] - 2026-03-13

### Added
- Initial release of Vix AI ecosystem.
- Core modules:
  - `ml/`: machine learning utilities
  - `tensor/`: tensors and computation primitives
  - `nn/`: neural networks
  - `nlp/`: natural language processing
  - `vision/`: computer vision utilities
  - `distributed/`: distributed P2P computation
- CMake options for building, testing, warnings, and installation.
- Examples and test programs for each submodule.
- Fully compatible with all Vix AI submodules, enabling cross-module pipelines.

### Changed
- N/A

### Fixed
- N/A
