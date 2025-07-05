// RUN: not sharp-opt %s -sharp-simulate=mode=jit 2>&1 | FileCheck %s

// CHECK: error: cannot be converted to LLVM IR: missing `LLVMTranslationDialectInterface`
txn.module @SimpleCounter attributes {moduleName = "SimpleCounter"} {
  txn.value_method @read() -> i32 {
    %zero = arith.constant 0 : i32
    txn.return %zero : i32
  }
  
  txn.action_method @increment() {
    txn.yield
  }
  
  txn.rule @auto_incr {
    %true = arith.constant true
    txn.yield %true : i1
  }
  
  txn.schedule [@auto_incr, @read, @increment]
}