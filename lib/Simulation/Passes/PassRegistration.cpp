//===- PassRegistration.cpp - Register Simulation Passes ------------------===//
//
// This file implements the registration of simulation passes.
//
//===----------------------------------------------------------------------===//

#include "sharp/Simulation/Passes.h"

namespace sharp {

// Generate the definitions of the passes
#define GEN_PASS_REGISTRATION
#include "sharp/Simulation/Passes.h.inc"

void registerSimulationPasses() {
  registerSharpSimulationPasses();
}

} // namespace sharp