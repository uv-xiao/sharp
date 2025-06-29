// RUN: sharp-opt %s | FileCheck %s

// CHECK-LABEL: txn.module @SimpleModule
txn.module @SimpleModule {
  txn.schedule []
}

// CHECK-LABEL: txn.module @ModuleWithAttrs
// CHECK-SAME: conflict_matrix = {
txn.module @ModuleWithAttrs attributes {conflict_matrix = {"rule1,rule2" = 2 : i32}} {
  txn.schedule []
}