cmake_minimum_required(VERSION 3.20)

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
add_library(vix::ai::all ALIAS vix_ai_all)

# Lier uniquement les modules qui existent
if (TARGET vix_ai_tensor)
  target_link_libraries(vix_ai_all INTERFACE vix_ai_tensor)
endif()
if (TARGET vix_ai_ml)
  target_link_libraries(vix_ai_all INTERFACE vix_ai_ml)
endif()
if (TARGET vix_ai_nn)
  target_link_libraries(vix_ai_all INTERFACE vix_ai_nn)
endif()
if (TARGET vix_ai_nlp)
  target_link_libraries(vix_ai_all INTERFACE vix_ai_nlp)
endif()
if (TARGET vix_ai_vision)
  target_link_libraries(vix_ai_all INTERFACE vix_ai_vision)
endif()
if (TARGET vix_ai_distributed)
  target_link_libraries(vix_ai_all INTERFACE vix_ai_distributed)
endif()

# ---- Installation and Package Export ---------------------------------------
if (VIX_AI_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  # Installer chaque module individuellement s'il existe
  foreach(_mod IN ITEMS tensor ml nn nlp vision distributed)
    if (TARGET vix_ai_${_mod})
      install(TARGETS vix_ai_${_mod}
        EXPORT vix_aiTargets
      )
    endif()
  endforeach()

  # Installer l'export global
  install(EXPORT vix_aiTargets
    FILE vix_aiTargets.cmake
    NAMESPACE vix::ai::
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
