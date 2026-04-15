#  @file CMakeLists.txt
#  @author Gaspard Kirira
#
#  Copyright 2025, Gaspard Kirira. All rights reserved.
#  https://github.com/vixcpp/vix
#  Use of this source code is governed by a MIT license
#  that can be found in the License file.
#
#  @brief Vix AI umbrella project.
#
#  Umbrella build for the Vix AI modules.
#  Provides optional integration of tensor, ml, nn, nlp, vision,
#  and distributed modules under a single top-level project.
#
#  Public targets:
#    - vix_ai_all
#    - vix::ai::all
#
#  Installation:
#    Installs the umbrella package configuration and exported targets
#    for find_package(vix_ai CONFIG REQUIRED).

cmake_minimum_required(VERSION 3.20)

project(vix_ai VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(VIX_AI_BUILD_EXAMPLES "Build umbrella examples" ON)
option(VIX_AI_BUILD_TENSOR "Build tensor module" ON)
option(VIX_AI_BUILD_ML "Build ml module" OFF)
option(VIX_AI_BUILD_NN "Build nn module" OFF)
option(VIX_AI_BUILD_NLP "Build nlp module" OFF)
option(VIX_AI_BUILD_VISION "Build vision module" OFF)
option(VIX_AI_BUILD_DISTRIBUTED "Build distributed module" OFF)
option(VIX_AI_INSTALL "Install vix_ai umbrella targets" ON)

if(MSVC)
  add_compile_options(/W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(VIX_AI_VERSION ${PROJECT_VERSION})
set(VIX_AI_UMBRELLA_BUILD ON)

function(vix_ai_add_module dir option_name)
  if(${option_name} AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt")
    message(STATUS "[vix_ai] Adding module: ${dir}")
    add_subdirectory(${dir})
  else()
    message(STATUS "[vix_ai] Skipping module: ${dir}")
  endif()
endfunction()

vix_ai_add_module(tensor VIX_AI_BUILD_TENSOR)
vix_ai_add_module(ml VIX_AI_BUILD_ML)
vix_ai_add_module(nn VIX_AI_BUILD_NN)
vix_ai_add_module(nlp VIX_AI_BUILD_NLP)
vix_ai_add_module(vision VIX_AI_BUILD_VISION)
vix_ai_add_module(distributed VIX_AI_BUILD_DISTRIBUTED)

if(VIX_AI_BUILD_EXAMPLES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
  add_subdirectory(examples)
endif()

add_library(vix_ai_all INTERFACE)
add_library(vix::ai::all ALIAS vix_ai_all)

target_compile_features(vix_ai_all INTERFACE cxx_std_20)

if(TARGET vix_ai_tensor)
  target_link_libraries(vix_ai_all INTERFACE vix::ai::tensor)
endif()

if(TARGET vix_ai_ml)
  target_link_libraries(vix_ai_all INTERFACE vix::ai::ml)
endif()

if(TARGET vix_ai_nn)
  target_link_libraries(vix_ai_all INTERFACE vix::ai::nn)
endif()

if(TARGET vix_ai_nlp)
  target_link_libraries(vix_ai_all INTERFACE vix::ai::nlp)
endif()

if(TARGET vix_ai_vision)
  target_link_libraries(vix_ai_all INTERFACE vix::ai::vision)
endif()

if(TARGET vix_ai_distributed)
  target_link_libraries(vix_ai_all INTERFACE vix::ai::distributed)
endif()

set_target_properties(vix_ai_all PROPERTIES
  EXPORT_NAME ai::all
)

if(VIX_AI_INSTALL)
  install(TARGETS vix_ai_all
    EXPORT vix_aiTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  install(EXPORT vix_aiTargets
    FILE vix_aiTargets.cmake
    NAMESPACE vix::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vix_ai
  )

  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vix_aiConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/vix_aiConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/vix_ai"
  )

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/vix_aiConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
  )

  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/vix_aiConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/vix_aiConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/vix_ai"
  )
endif()

if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt")
  add_subdirectory(test)
endif()

message(STATUS "vix::ai configured (${PROJECT_VERSION})")
message(STATUS "Build tensor: ${VIX_AI_BUILD_TENSOR}")
message(STATUS "Build ml: ${VIX_AI_BUILD_ML}")
message(STATUS "Build nn: ${VIX_AI_BUILD_NN}")
message(STATUS "Build nlp: ${VIX_AI_BUILD_NLP}")
message(STATUS "Build vision: ${VIX_AI_BUILD_VISION}")
message(STATUS "Build distributed: ${VIX_AI_BUILD_DISTRIBUTED}")
message(STATUS "Build examples: ${VIX_AI_BUILD_EXAMPLES}")
message(STATUS "Install: ${VIX_AI_INSTALL}")
