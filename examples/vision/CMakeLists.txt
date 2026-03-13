cmake_minimum_required(VERSION 3.20)

#[[
Vix.AI — Vision Examples
Built only if vix::ai::vision is available.
]]

message(STATUS "Building Vix.AI.Vision examples")

if (NOT TARGET vix::ai::vision)
  message(STATUS "Skipping Vision examples (Vix::AI::vision not found)")
  return()
endif()

add_executable(vix_ai_example_vision_info 01_info.cpp)
target_link_libraries(vix_ai_example_vision_info PRIVATE vix::ai::vision)
target_compile_features(vix_ai_example_vision_info PRIVATE cxx_std_20)

add_executable(vix_ai_example_vision_detect 02_detect.cpp)
target_link_libraries(vix_ai_example_vision_detect PRIVATE vix::ai::vision)
target_compile_features(vix_ai_example_vision_detect PRIVATE cxx_std_20)

add_executable(vix_ai_example_vision_track 03_track.cpp)
target_link_libraries(vix_ai_example_vision_track PRIVATE vix::ai::vision)
target_compile_features(vix_ai_example_vision_track PRIVATE cxx_std_20)
