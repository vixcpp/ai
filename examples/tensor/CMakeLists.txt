cmake_minimum_required(VERSION 3.20)

#[[
===============================================================================
Vix.AI — Tensor (tensor) Examples
===============================================================================
This CMake script builds demonstration programs for the Tensor module of Vix.AI.

The Tensor module provides foundational abstractions:
  • Device  — hardware selection (CPU / CUDA)
  • Tensor  — lightweight multidimensional array
  • Engine  — execution and computation interface

These examples are always built as long as the Tensor library (vix::ai::tensor)
exists in the current build graph.
===============================================================================
]]

message(STATUS "Building vix.ai.tensor examples")

# Vérifie que la lib tensor existe
if (NOT TARGET vix::ai::tensor)
  message(WARNING "Skipping Tensor examples (vix::ai::tensor not found)")
  return()
endif()

# 01 — minimal
add_executable(vix_ai_example_tensor_minimal 01_tensor_minimal.cpp)
target_link_libraries(vix_ai_example_tensor_minimal PRIVATE vix::ai::tensor)
target_compile_features(vix_ai_example_tensor_minimal PRIVATE cxx_std_20)

# 02 — tensor demo (fill/reshape/compute)
add_executable(vix_ai_example_tensor_demo 02_tensor_demo.cpp)
target_link_libraries(vix_ai_example_tensor_demo PRIVATE vix::ai::tensor)
target_compile_features(vix_ai_example_tensor_demo PRIVATE cxx_std_20)

# 03 — device parse & info
add_executable(vix_ai_example_tensor_device_parse 03_device_parse.cpp)
target_link_libraries(vix_ai_example_tensor_device_parse PRIVATE vix::ai::tensor)
target_compile_features(vix_ai_example_tensor_device_parse PRIVATE cxx_std_20)

# 04 — engine info
add_executable(vix_ai_example_tensor_engine_info 04_engine_info.cpp)
target_link_libraries(vix_ai_example_tensor_engine_info PRIVATE vix::ai::tensor)
target_compile_features(vix_ai_example_tensor_engine_info PRIVATE cxx_std_20)
