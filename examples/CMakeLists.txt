cmake_minimum_required(VERSION 3.20)

#[[
===============================================================================
Vix.AI — Examples Collection
===============================================================================
This CMake script builds demonstration programs for various Vix.AI modules.

Each example illustrates how to use a specific subsystem:
  • Tensor/Core (tensors, devices, engine basics)
  • ML (machine learning primitives)
  • NN (neural network components)
  • NLP (tokenizers, embeddings, pipelines)
  • Vision
  • Distributed

Examples for optional modules are only built if the corresponding CMake
target exists (e.g. vix::ai::ml) and the example folder is present.
===============================================================================
]]

message(STATUS "Building Vix.AI examples")

# ---- Tensor/Core examples ---------------------------------------------------
if (TARGET vix::ai::tensor)
  add_executable(vix_ai_example_01_tensor_minimal 01_minimal.cpp)
  target_link_libraries(vix_ai_example_01_tensor_minimal PRIVATE vix::ai::tensor)
  target_compile_features(vix_ai_example_01_tensor_minimal PRIVATE cxx_std_20)

  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tensor/CMakeLists.txt")
    add_subdirectory(tensor)
  endif()
else()
  message(STATUS "Skipping Tensor/Core examples (vix::ai::tensor not found)")
endif()

# ---- ML examples ------------------------------------------------------------
if (TARGET vix::ai::ml)
  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/02_ml_minimal.cpp")
    add_executable(vix_ai_example_02_ml_minimal 02_ml_minimal.cpp)
    target_link_libraries(vix_ai_example_02_ml_minimal PRIVATE vix::ai::ml)
    target_compile_features(vix_ai_example_02_ml_minimal PRIVATE cxx_std_20)
  endif()

  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ml/CMakeLists.txt")
    add_subdirectory(ml)
  endif()
else()
  message(STATUS "Skipping ML examples (vix::ai::ml not found)")
endif()

# ---- NLP examples -----------------------------------------------------------
if (TARGET vix::ai::nlp)
  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nlp/CMakeLists.txt")
    add_subdirectory(nlp)
  else()
    message(STATUS "NLP examples folder missing (examples/nlp)")
  endif()
else()
  message(STATUS "Skipping NLP examples (vix::ai::nlp not found)")
endif()

# ---- NN examples ------------------------------------------------------------
if (TARGET vix::ai::nn)
  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nn/CMakeLists.txt")
    add_subdirectory(nn)
  else()
    message(STATUS "NN examples folder missing (examples/nn)")
  endif()
else()
  message(STATUS "Skipping NN examples (vix::ai::nn not found)")
endif()

# ---- Vision examples --------------------------------------------------------
if (TARGET vix::ai::vision)
  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vision/CMakeLists.txt")
    add_subdirectory(vision)
  else()
    message(STATUS "Vision examples folder missing (examples/vision)")
  endif()
else()
  message(STATUS "Skipping Vision examples (vix::ai::vision not found)")
endif()

# ---- Distributed examples ---------------------------------------------------
if (TARGET vix::ai::distributed)
  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/distributed/CMakeLists.txt")
    add_subdirectory(distributed)
  else()
    message(STATUS "Distributed examples folder missing (examples/distributed)")
  endif()
else()
  message(STATUS "Skipping Distributed examples (vix::ai::distributed not found)")
endif()
