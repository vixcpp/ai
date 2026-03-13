cmake_minimum_required(VERSION 3.20)

#[[
Vix.AI — Distributed Examples
Built only if Vix::ai::distributed is available.
]]

message(STATUS "Building Vix.ai.Distributed examples")

# Vérifie que la lib existe
if (NOT TARGET Vix::ai::distributed)
  message(STATUS "Skipping Distributed examples (Vix::ai::distributed not found)")
  return()
endif()

# Exemple minimal hello
add_executable(vix_ai_example_distrib_hello 01_hello.cpp)
target_link_libraries(vix_ai_example_distrib_hello PRIVATE Vix::ai::distributed)
target_compile_features(vix_ai_example_distrib_hello PRIVATE cxx_std_20)
