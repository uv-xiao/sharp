// RUN: sharp-opt %s | FileCheck %s

// Test basic primitive operations

// CHECK-LABEL: txn.primitive @RegisterPrimitive 
// CHECK-SAME: type = "hw"
// CHECK-SAME: interface = !txn.module<"RegisterPrimitive">
txn.primitive @RegisterPrimitive type = "hw" interface = !txn.module<"RegisterPrimitive"> {
  // CHECK: txn.fir_value_method @read() : () -> i32
  txn.fir_value_method @read() : () -> i32
  
  // CHECK: txn.fir_action_method @write() : (i32) -> ()
  txn.fir_action_method @write() : (i32) -> ()
  
  // CHECK: txn.clock_by @clk
  txn.clock_by @clk
  
  // CHECK: txn.reset_by @rst
  txn.reset_by @rst
  
  // CHECK: txn.schedule [@read, @write] {
  // CHECK-DAG: "read,read" = 3 : i32
  // CHECK-DAG: "read,write" = 3 : i32
  // CHECK-DAG: "write,read" = 3 : i32
  // CHECK-DAG: "write,write" = 2 : i32
  // CHECK: }
  txn.schedule [@read, @write] {
    conflict_matrix = {
      "read,read" = 3 : i32,    // CF
      "read,write" = 3 : i32,   // CF
      "write,read" = 3 : i32,   // CF
      "write,write" = 2 : i32   // C
    }
  }
}

// CHECK-LABEL: txn.primitive @WirePrimitive
// CHECK-SAME: type = "hw"
// CHECK-SAME: interface = !txn.module<"WirePrimitive">
txn.primitive @WirePrimitive type = "hw" interface = !txn.module<"WirePrimitive"> {
  // CHECK: txn.fir_value_method @read() : () -> i32
  txn.fir_value_method @read() : () -> i32
  
  // CHECK: txn.fir_action_method @write() : (i32) -> ()
  txn.fir_action_method @write() : (i32) -> ()
  
  // CHECK: txn.clock_by @clk
  txn.clock_by @clk
  
  // CHECK: txn.reset_by @rst
  txn.reset_by @rst
  
  // CHECK: txn.schedule [@read, @write] {
  // CHECK-DAG: "read,read" = 3 : i32
  // CHECK-DAG: "read,write" = 0 : i32
  // CHECK-DAG: "write,read" = 1 : i32
  // CHECK-DAG: "write,write" = 2 : i32
  // CHECK: }
  txn.schedule [@read, @write] {
    conflict_matrix = {
      "read,read" = 3 : i32,    // CF
      "read,write" = 0 : i32,   // SB (read before write)
      "write,read" = 1 : i32,   // SA
      "write,write" = 2 : i32   // C
    }
  }
}

// CHECK-LABEL: txn.primitive @SpecPrimitive
// CHECK-SAME: type = "spec"
txn.primitive @SpecPrimitive type = "spec" interface = !txn.module<"SpecPrimitive"> {
  // Spec primitives use regular txn operations
  
  // CHECK: txn.value_method @getValue() -> i32
  txn.value_method @getValue() -> i32 {
    %c42 = arith.constant 42 : i32
    txn.return %c42 : i32
  }
  
  // CHECK: txn.action_method @setValue(%{{.*}}: i32)
  txn.action_method @setValue(%val: i32) {
    txn.return
  }
  
  // CHECK: txn.schedule [@getValue, @setValue]
  txn.schedule [@getValue, @setValue]
}

// Test module using primitives
// CHECK-LABEL: txn.module @ModuleWithPrimitives
txn.module @ModuleWithPrimitives {
  // CHECK: %{{.*}} = txn.instance @reg of @RegisterPrimitive : !txn.module<"RegisterPrimitive">
  %reg = txn.instance @reg of @RegisterPrimitive : !txn.module<"RegisterPrimitive">
  
  // CHECK: %{{.*}} = txn.instance @wire of @WirePrimitive : !txn.module<"WirePrimitive">
  %wire = txn.instance @wire of @WirePrimitive : !txn.module<"WirePrimitive">
  
  txn.rule @transfer {
    // Read from register
    %val = txn.call @reg.read() : () -> i32
    // Write to wire
    txn.call @wire.write(%val) : (i32) -> ()
    txn.yield
  }
  
  txn.schedule [@transfer]
}