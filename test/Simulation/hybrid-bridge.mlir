// RUN: sharp-opt %s -sharp-arcilator -sharp-hybrid-sim -output=%t.cpp | FileCheck %s
// RUN: sharp-opt %s -sharp-arcilator -sharp-hybrid-sim | FileCheck %s --check-prefix=OUTPUT

// CHECK: Successfully converted to Arc dialect for RTL simulation
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
// OUTPUT:       "method_name": "increment",
// OUTPUT:       "input_signals": [],
// OUTPUT:       "output_signals": [],
// OUTPUT:       "enable_signal": "increment_en",
// OUTPUT:       "ready_signal": "increment_rdy"
// OUTPUT:     }
// OUTPUT:   ]
// OUTPUT: })";

txn.module @Counter attributes {moduleName = "Counter"} {
  txn.value_method @getValue() -> i32 attributes {timing = "combinational"} {
    %c0 = arith.constant 0 : i32
    txn.return %c0 : i32
  }
  
  txn.action_method @increment() attributes {timing = "static(1)"} {
    txn.yield
  }
  
  txn.rule @autoIncrement {
    %true = arith.constant 1 : i1
    txn.yield %true : i1
  }
  
  txn.schedule [@autoIncrement, @increment] {
    conflict_matrix = {
      "autoIncrement,autoIncrement" = 2 : i32,
      "increment,autoIncrement" = 3 : i32
    }
  }
}