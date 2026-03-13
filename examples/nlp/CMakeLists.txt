cmake_minimum_required(VERSION 3.20)

#[[
Vix.AI — NLP Examples
Built only if vix::ai::nlp is available.
]]

message(STATUS "Building Vix.AI.NLP examples")

if (NOT TARGET vix::ai::nlp)
  message(STATUS "Skipping NLP examples (Vix::AI::nlp not found)")
  return()
endif()

add_executable(vix_ai_example_nlp_tokenize 01_tokenize.cpp)
target_link_libraries(vix_ai_example_nlp_tokenize PRIVATE vix::ai::nlp)
target_compile_features(vix_ai_example_nlp_tokenize PRIVATE cxx_std_20)

add_executable(vix_ai_example_nlp_embed 02_embed.cpp)
target_link_libraries(vix_ai_example_nlp_embed PRIVATE vix::ai::nlp)
target_compile_features(vix_ai_example_nlp_embed PRIVATE cxx_std_20)

add_executable(vix_ai_example_nlp_summarize 03_summarize.cpp)
target_link_libraries(vix_ai_example_nlp_summarize PRIVATE vix::ai::nlp)
target_compile_features(vix_ai_example_nlp_summarize PRIVATE cxx_std_20)
