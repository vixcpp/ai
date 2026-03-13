cmake_minimum_required(VERSION 3.20)

#[[
===============================================================================
Vix.AI — ML Examples
===============================================================================
Builds demonstration and test programs for the Machine Learning module.
===============================================================================
]]

option(VIX_AI_BUILD_EXAMPLES "Build Vix.AI examples" ON)
if (NOT VIX_AI_BUILD_EXAMPLES)
  message(STATUS "Skipping all examples (VIX_AI_BUILD_EXAMPLES=OFF)")
  return()
endif()

message(STATUS "Building Vix.ai.ML examples")

# Vérifie que la lib existe
if (NOT TARGET Vix::ai::ml)
  message(STATUS "Skipping ML examples (Vix::ai::ml not found)")
  return()
endif()

# 1) Exemple: regression + clustering
add_executable(vix_ai_example_ml_regression_kmeans 02_regression_kmeans.cpp)
target_compile_features(vix_ai_example_ml_regression_kmeans PRIVATE cxx_std_20)
target_link_libraries(vix_ai_example_ml_regression_kmeans PRIVATE Vix::ai::ml)
if (TARGET Vix::ai::tensor)
  target_link_libraries(vix_ai_example_ml_regression_kmeans PRIVATE Vix::ai::tensor)
endif()

# 2) Ne pas redéfinir les tests: ils vivent dans ml/tests
if (TARGET vix_ai_ml_smoke_test)
  message(STATUS "Target vix_ai_ml_smoke_test already defined (from tests) — skipping here.")
endif()
if (TARGET vix_ai_ml_test)
  message(STATUS "Target vix_ai_ml_test already defined (from tests) — skipping here.")
endif()

# 3) Cible pratique pour lancer l’exemple
add_custom_target(run_ml_example
  COMMAND $<TARGET_FILE:vix_ai_example_ml_regression_kmeans>
  DEPENDS vix_ai_example_ml_regression_kmeans
  COMMENT "Running ML regression+kmeans example"
)

message(STATUS "Vix.ai.ML example configured successfully.")
