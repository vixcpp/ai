cmake_minimum_required(VERSION 3.20)

#[[
Vix.AI — NN Examples
Built only if Vix::AI::nn is available.
]]

message(STATUS "Building Vix.AI.NN examples")

if (NOT TARGET vix::ai::nn)
  message(STATUS "Skipping NN examples (vix::ai::nn not found)")
  return()
endif()

add_executable(vix_ai_example_nn_summary 01_summary.cpp)
target_link_libraries(vix_ai_example_nn_summary PRIVATE vix::ai::nn)
target_compile_features(vix_ai_example_nn_summary PRIVATE cxx_std_20)
