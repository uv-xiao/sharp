cmake_minimum_required(VERSION 3.20.0)

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

project(sharp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Debug")
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type (default Debug)" FORCE)
endif()

# Options
option(SHARP_INCLUDE_TOOLS "Build Sharp tools" ON)
option(SHARP_INCLUDE_TESTS "Generate Sharp test targets" ON)
option(SHARP_INCLUDE_DOCS "Generate Sharp documentation" OFF)
option(SHARP_BINDINGS_PYTHON_ENABLED "Enable Python bindings" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(SHARP_ENABLE_ASSERTIONS "Enable assertions" ON)

# Find MLIR
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Find CIRCT
find_package(CIRCT CONFIG)
if(CIRCT_FOUND)
  message(STATUS "Using CIRCTConfig.cmake in: ${CIRCT_DIR}")
else()
  message(STATUS "CIRCT not found, some features may be disabled")
endif()

# Configure include paths and definitions
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Set up LLVM/MLIR includes
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

# Add CIRCT includes if available
if(CIRCT_FOUND)
  include_directories(${CIRCT_INCLUDE_DIRS})
endif()

# Set up definitions
add_definitions(${LLVM_DEFINITIONS})

# Configure RPATH
if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Set up tablegen
set(SHARP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SHARP_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(SHARP_INCLUDE_DIR ${PROJECT_BINARY_DIR}/include)
set(SHARP_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)

# Configure Python bindings
if(SHARP_BINDINGS_PYTHON_ENABLED)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

# Add subdirectories
add_subdirectory(include/sharp)
add_subdirectory(lib)

if(SHARP_INCLUDE_TOOLS)
  add_subdirectory(tools)
endif()

if(SHARP_INCLUDE_TESTS)
  enable_testing()
  add_subdirectory(test)
  add_subdirectory(unittests)
endif()

if(SHARP_INCLUDE_DOCS)
  add_subdirectory(docs)
endif()

# Export targets
export(EXPORT SharpTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/SharpTargets.cmake"
  NAMESPACE sharp::
)

# Create SharpConfig.cmake
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SharpConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/SharpConfig.cmake @ONLY)

# Installation
install(DIRECTORY include/sharp
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.td"
  PATTERN "*.def"
)

install(DIRECTORY ${SHARP_INCLUDE_DIR}/sharp
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.inc"
  PATTERN "*.gen"
)

install(EXPORT SharpTargets
  FILE SharpTargets.cmake
  NAMESPACE sharp::
  DESTINATION lib/cmake/sharp
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SharpConfig.cmake
  DESTINATION lib/cmake/sharp
)