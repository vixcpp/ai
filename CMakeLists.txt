cmake_minimum_required(VERSION 3.20)

#[[
===============================================================================
Vix AI - Umbrella CMake Project
===============================================================================
This is the top-level build configuration for the **Vix.AI** ecosystem.

It aggregates multiple AI-oriented C++ modules developed under the Vix project:
  • tensor         — Core tensor, device & engine primitives
  • ml             — Machine learning utilities and algorithms
  • nn             — Neural network primitives
  • nlp            — Natural language processing tools
  • vision         — Computer vision components
  • distributed    — Distributed / parallel computing

Each subproject can be built or skipped independently using -DVIX_AI_BUILD_* options.
This structure allows modular development and incremental builds while providing
a unified installation and package configuration (`find_package(vix_ai CONFIG)`).

Usage:
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVIX_AI_BUILD_TENSOR=ON
  cmake --build build
  cmake --install build --prefix /usr/local
===============================================================================
]]

project(vix_ai
  VERSION 0.1.0
  DESCRIPTION "Vix AI umbrella: tensor, ml, nn, nlp, vision, distributed"
  LANGUAGES CXX)

include(CTest)

# ---- Global build options ---------------------------------------------------
option(VIX_AI_BUILD_EXAMPLES   "Build umbrella examples" ON)
option(VIX_AI_BUILD_TENSOR     "Build tensor module"       ON)
option(VIX_AI_BUILD_ML         "Build ml module"           OFF)
option(VIX_AI_BUILD_NN         "Build nn module"           OFF)
option(VIX_AI_BUILD_NLP        "Build nlp module"          OFF)
option(VIX_AI_BUILD_VISION     "Build vision module"       OFF)
option(VIX_AI_BUILD_DISTRIB    "Build distributed module"  OFF)
option(VIX_AI_INSTALL          "Install umbrella package"  ON)

# ---- Compiler configuration -------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (MSVC)
  add_compile_options(/W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(VIX_AI_VERSION ${PROJECT_VERSION})

# ---- Helper function for optional subdirectories ---------------------------
function(add_dir_if exists_flag dir opt_var)
  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}" AND ${opt_var})
    message(STATUS "Adding subdirectory: ${dir}")
    add_subdirectory(${dir})
  else()
    message(STATUS "Skipping ${dir} (missing or disabled)")
  endif()
endfunction()

# ---- Subprojects ------------------------------------------------------------
add_dir_if(exists_flag tensor        VIX_AI_BUILD_TENSOR)
add_dir_if(exists_flag ml            VIX_AI_BUILD_ML)
add_dir_if(exists_flag nn            VIX_AI_BUILD_NN)
add_dir_if(exists_flag nlp           VIX_AI_BUILD_NLP)
add_dir_if(exists_flag vision        VIX_AI_BUILD_VISION)
add_dir_if(exists_flag distributed   VIX_AI_BUILD_DISTRIB)

if (VIX_AI_BUILD_EXAMPLES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples")
  add_subdirectory(examples)
endif()

# ---- Meta target linking all available modules -----------------------------
add_library(vix_ai_all INTERFACE)
add_library(Vix::AI::all ALIAS vix_ai_all)

if (TARGET Vix::ai::tensor)
  target_link_libraries(vix_ai_all INTERFACE Vix::ai::tensor)
endif()
if (TARGET Vix::ai::ml)
  target_link_libraries(vix_ai_all INTERFACE Vix::ai::ml)
endif()
if (TARGET Vix::ai::nn)
  target_link_libraries(vix_ai_all INTERFACE Vix::ai::nn)
endif()
if (TARGET Vix::ai::nlp)
  target_link_libraries(vix_ai_all INTERFACE Vix::ai::nlp)
endif()
if (TARGET Vix::ai::vision)
  target_link_libraries(vix_ai_all INTERFACE Vix::ai::vision)
endif()
if (TARGET Vix::ai::distributed)
  target_link_libraries(vix_ai_all INTERFACE Vix::ai::distributed)
endif()

# ---- Installation and Package Export ---------------------------------------
if (VIX_AI_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  install(TARGETS vix_ai_all
    EXPORT vix_aiTargets
  )

  install(EXPORT vix_aiTargets
    FILE vix_aiTargets.cmake
    NAMESPACE Vix::AI::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vix_ai
  )

  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/vix_aiConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
  )

  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/vix_aiConfig.cmake
"include(\"\${CMAKE_CURRENT_LIST_DIR}/vix_aiTargets.cmake\")\n"
"# Load subpackage targets if installed\n"
"foreach(_comp IN ITEMS tensor ml nn nlp vision distributed)\n"
"  set(_p \"\${CMAKE_CURRENT_LIST_DIR}/../\${_comp}/\${_comp}Targets.cmake\")\n"
"  if (EXISTS \"\${_p}\")\n"
"    include(\"\${_p}\")\n"
"  endif()\n"
"endforeach()\n"
  )

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/vix_aiConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/vix_aiConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vix_ai
  )
endif()

# ---- Global testing placeholder --------------------------------------------
if (BUILD_TESTING)
  # add_subdirectory(tests)
endif()
