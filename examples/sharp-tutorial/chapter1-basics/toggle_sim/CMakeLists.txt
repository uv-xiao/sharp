cmake_minimum_required(VERSION 3.16)
project(Toggle_simulation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(Toggle_sim
  Toggle_sim.cpp
  SimulationBase.h
  SimulationBase.cpp
)

# Include directories
target_include_directories(Toggle_sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
