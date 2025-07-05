// RUN: sharp-opt %s -sharp-hybrid-sim 2>&1 | FileCheck %s
// RUN: sharp-opt %s -sharp-hybrid-sim 2>&1 | FileCheck %s --check-prefix=OUTPUT

// CHECK: Hybrid Simulation Setup:
// CHECK: 1. Run --sharp-arcilator to generate RTL representation
// CHECK: 2. Compile the generated C++ code with simulation libraries
// CHECK: 3. Run the executable for hybrid TL-RTL simulation

// OUTPUT: // Generated Hybrid TL-RTL Simulation Code
// OUTPUT: #include "sharp/Simulation/Hybrid/HybridBridge.h"
// OUTPUT: // Bridge configuration
// OUTPUT: const char* BRIDGE_CONFIG = R"({
// OUTPUT:   "sync_mode": "lockstep",
// OUTPUT:   "module_mappings": [
// OUTPUT:     {
// OUTPUT:       "tl_module": "Counter",
// OUTPUT:       "rtl_module": "Counter_rtl"
// OUTPUT:     }
// OUTPUT:   ],
// OUTPUT:   "method_mappings": [
// OUTPUT:     {
// OUTPUT:       "method_name": "getValue",
// OUTPUT:       "input_signals": [],
// OUTPUT:       "output_signals": ["getValue_result"],
// OUTPUT:       "enable_signal": "getValue_en",
// OUTPUT:       "ready_signal": "getValue_rdy"
// OUTPUT:     },
// OUTPUT:     {
// OUTPUT:       "method_name": "getStatus",
// OUTPUT:       "input_signals": [],
// OUTPUT:       "output_signals": ["getStatus_result"],
// OUTPUT:       "enable_signal": "getStatus_en",
// OUTPUT:       "ready_signal": "getStatus_rdy"
// OUTPUT:     }
// OUTPUT:   ]
// OUTPUT: })";

txn.module @Counter attributes {moduleName = "Counter"} {
  txn.value_method @getValue() -> i32 attributes {timing = "combinational"} {
    %c0 = arith.constant 0 : i32
    txn.return %c0 : i32
  }
  
  txn.value_method @getStatus() -> i32 attributes {timing = "combinational"} {
    %c1 = arith.constant 1 : i32
    txn.return %c1 : i32
  }
  
  txn.schedule [@getValue, @getStatus]
}