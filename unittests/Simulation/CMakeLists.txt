add_sharp_unittest(SharpSimulationTest
  basic.cpp
)

target_link_libraries(SharpSimulationTest
  PRIVATE
  SharpSimulationCore
  MLIRIR
  MLIRSupport
)