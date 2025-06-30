// RUN: sharp-opt --sharp-pre-synthesis-check %s --allow-unregistered-dialect 2>&1 | FileCheck %s

// Test checking for allowed/disallowed operations in synthesizable code

// Test case: module with allowed arith operations (should pass)
txn.module @ModuleWithArith {
  txn.rule @compute {
    %c42 = arith.constant 42 : i32
    %sum = arith.addi %c42, %c42 : i32
    txn.return
  }
  
  txn.schedule [@compute] {conflict_matrix = {}}
}
// CHECK-NOT: error: Module 'ModuleWithArith'

// Test case: module with disallowed operations
txn.module @ModuleWithBadOps {
  txn.rule @useUnregistered {
    // CHECK-DAG: error: operation 'test.unregistered' is not allowed in synthesizable code
    %result = "test.unregistered"() : () -> i32
    txn.return
  }
  
  txn.schedule [@useUnregistered] {conflict_matrix = {}}
}
// CHECK-DAG: error: Module 'ModuleWithBadOps' is non-synthesizable: contains non-synthesizable operations

// Test case: good module with only allowed operations
txn.module @GoodModule {
  %inst = txn.instance @prim of @GoodPrimitive : !txn.module<"GoodPrimitive">
  
  txn.rule @useInstance {
    %val = txn.call @prim::@read() : () -> i32
    %c1 = arith.constant 1 : i32
    %inc = arith.addi %val, %c1 : i32
    txn.call @prim::@write(%inc) : (i32) -> ()
    txn.return
  }
  
  txn.value_method @getValue() -> i32 {
    %val = txn.call @prim::@read() : () -> i32
    txn.return %val : i32
  }
  
  txn.schedule [@useInstance, @getValue] {conflict_matrix = {}}
}
// CHECK-NOT: error: Module 'GoodModule'

// Test case: good primitive (referenced by GoodModule)
txn.primitive @GoodPrimitive type = "hw" interface = !txn.module<"GoodPrimitive"> {
  txn.fir_value_method @read() {firrtl.port = "read_data"} : () -> i32
  txn.fir_action_method @write() {firrtl.data_port = "write_data", firrtl.enable_port = "write_enable"} : (i32) -> ()
  txn.clock_by @clk
  txn.reset_by @rst
  txn.schedule [@read, @write] {conflict_matrix = {
    "read,read" = 3 : i32,
    "read,write" = 3 : i32,
    "write,read" = 3 : i32,
    "write,write" = 2 : i32
  }}
} {firrtl.impl = "GoodPrimitive_impl"}
// CHECK-NOT: error: Module 'GoodPrimitive'