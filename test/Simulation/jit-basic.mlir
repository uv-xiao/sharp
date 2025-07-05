// RUN: not sharp-opt %s -sharp-simulate=mode=jit 2>&1 | FileCheck %s

// CHECK: error: cannot be converted to LLVM IR: missing `LLVMTranslationDialectInterface`
txn.module @BasicCounter attributes {moduleName = "BasicCounter"} {
  txn.value_method @getValue() -> i32 {
    %c42 = arith.constant 42 : i32
    txn.return %c42 : i32
  }
  
  txn.action_method @setValue(%val: i32) {
    txn.yield
  }
  
  txn.schedule [@getValue, @setValue]
}