// RUN: sharp-opt %s -sharp-simulate=mode=translation | FileCheck %s

// Simple adder module for testing TxnSimulate pass
txn.module @Adder {
    txn.value_method @add(%a: i32, %b: i32) -> i32 {
        %sum = arith.addi %a, %b : i32
        txn.return %sum : i32
    }
    
    txn.schedule [@add]
}

// CHECK: // Generated Txn Module Simulation
// CHECK: #include <iostream>
// CHECK: #include <memory>
// CHECK: #include <vector>
// CHECK: #include <map>
// CHECK: #include <string>
// CHECK: #include <functional>
// CHECK: #include <cassert>
// CHECK: #include <chrono>
// CHECK: #include <queue>

// CHECK: class AdderModule : public SimModule {
// CHECK: public:
// CHECK:   AdderModule() : SimModule("Adder") {
// CHECK:     // Register methods
// CHECK:     registerValueMethod("add", 
// CHECK:       [this](const std::vector<int64_t>& args) -> std::vector<int64_t> {
// CHECK:         return add(args[0], args[1]);
// CHECK:       });
// CHECK:   }

// CHECK:   // Value method: add
// CHECK:   std::vector<int64_t> add(int64_t arg0, int64_t arg1) {
// CHECK:     int64_t _0 = arg0 + arg1;
// CHECK:     return {_0};
// CHECK:   }