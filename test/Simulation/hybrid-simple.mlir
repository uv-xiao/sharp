// RUN: sharp-opt %s -sharp-hybrid-sim | FileCheck %s

// CHECK: // Generated Hybrid TL-RTL Simulation Code
// CHECK: #include "sharp/Simulation/Hybrid/HybridBridge.h"
// CHECK: // Bridge configuration
// CHECK: const char* BRIDGE_CONFIG = R"({
// CHECK:   "sync_mode": "lockstep",
// CHECK:   "module_mappings": [
// CHECK:     {
// CHECK:       "tl_module": "Adder",
// CHECK:       "rtl_module": "Adder_rtl"
// CHECK:     }
// CHECK:   ],
// CHECK:   "method_mappings": [
// CHECK:     {
// CHECK:       "method_name": "add",
// CHECK:       "input_signals": ["add_arg0", "add_arg1"],
// CHECK:       "output_signals": ["add_result"],
// CHECK:       "enable_signal": "add_en",
// CHECK:       "ready_signal": "add_rdy"
// CHECK:     }
// CHECK:   ]
// CHECK: })";

// CHECK: // Transaction-level simulation for Adder
// CHECK: class AdderTLSim : public SimModule {
// CHECK: public:
// CHECK:   AdderTLSim() : SimModule("Adder") {
// CHECK:     registerMethod("add", [this](const std::vector<uint64_t>& args) -> std::vector<uint64_t> {
// CHECK:       // TL implementation of add
// CHECK:       return {0}; // Placeholder
// CHECK:     });
// CHECK:   }
// CHECK: };

// CHECK: int main(int argc, char** argv) {
// CHECK:   std::cout << "Starting Hybrid TL-RTL Simulation" << std::endl;
// CHECK:   // Create TL simulator
// CHECK:   auto tlSimulator = std::make_unique<Simulator>();
// CHECK:   // Create hybrid bridge
// CHECK:   auto bridge = HybridBridgeFactory::create("", SyncMode::Lockstep);
// CHECK:   bridge->configure(BRIDGE_CONFIG);

txn.module @Adder attributes {moduleName = "Adder"} {
  txn.value_method @add(%a: i32, %b: i32) -> i32 {
    %sum = arith.addi %a, %b : i32
    txn.return %sum : i32
  }
  
  txn.schedule [@add]
}