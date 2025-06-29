// RUN: sharp-opt %s -allow-unregistered-dialect | sharp-opt -allow-unregistered-dialect | FileCheck %s

// Test basic TXN dialect functionality with updated syntax

// CHECK-LABEL: txn.module @BasicModule {
txn.module @BasicModule {
  // CHECK: %{{.*}} = txn.instance @helper of @Helper : !txn.module<"Helper">
  %helper = txn.instance @helper of @Helper : !txn.module<"Helper">
  
  // CHECK: txn.value_method @getValue() -> i32
  txn.value_method @getValue() -> i32 {
    %c42 = arith.constant 42 : i32
    txn.return %c42 : i32
  }
  
  // CHECK: txn.action_method @setValue(%{{.*}}: i32) -> i32
  txn.action_method @setValue(%val: i32) -> i32 {
    txn.return %val : i32
  }
  
  // CHECK: txn.schedule [@getValue, @setValue]
  txn.schedule [@getValue, @setValue]
}

// CHECK-LABEL: txn.module @ModuleWithPrimitive
txn.module @ModuleWithPrimitive {
  // CHECK: txn.primitive @storage type = "hw" interface = !txn.module<"StorageInterface">
  txn.primitive @storage type = "hw" interface = !txn.module<"StorageInterface"> {
    // Primitive body
  }
  
  // CHECK: txn.primitive @spec type = "spec" interface = !txn.module<"SpecInterface">
  txn.primitive @spec type = "spec" interface = !txn.module<"SpecInterface"> {
    // Specification body
  }
  
  txn.schedule []
}

// CHECK-LABEL: txn.module @ModuleWithRule
txn.module @ModuleWithRule {
  // CHECK: txn.rule @simpleRule
  txn.rule @simpleRule {
    %c1 = arith.constant 1 : i32
  }
  
  // CHECK: txn.rule @conditionalRule
  txn.rule @conditionalRule {
    %cond = arith.constant true
    txn.if %cond {
      "test.action"() : () -> ()
    } else {
      // Do nothing
    }
  }
  
  txn.schedule [@simpleRule, @conditionalRule]
}

// CHECK-LABEL: txn.module @Helper
txn.module @Helper {
  txn.value_method @help() -> i32 {
    %c0 = arith.constant 0 : i32
    txn.return %c0 : i32
  }
  
  txn.schedule [@help]
}