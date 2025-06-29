// RUN: sharp-opt %s -mlir-print-op-generic | sharp-opt -mlir-print-op-generic | FileCheck %s

// Test attributes using generic format

// CHECK: "txn.module"
// CHECK-SAME: conflict_matrix = {
"txn.module"() ({
  // CHECK: "txn.rule"
  // CHECK-SAME: timing = "combinational"
  "txn.rule"() ({
    "txn.yield"() : () -> ()
  }) {sym_name = "rule1", timing = "combinational"} : () -> ()

  // CHECK: "txn.rule"
  // CHECK-SAME: timing = "static(3)"
  "txn.rule"() ({
    "txn.yield"() : () -> ()
  }) {sym_name = "rule2", timing = "static(3)"} : () -> ()

  // CHECK: "txn.action_method"
  // CHECK-SAME: timing = "dynamic"
  "txn.action_method"() ({
    "txn.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "method1", timing = "dynamic"} : () -> ()

  // CHECK: "txn.value_method"
  // CHECK-SAME: timing = "combinational"
  "txn.value_method"() ({
    %c0 = "arith.constant"() {value = 42 : i32} : () -> i32
    "txn.return"(%c0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "valueMethod", timing = "combinational"} : () -> ()

  "txn.schedule"() {methods = [@rule1, @rule2, @method1]} : () -> ()
}) {sym_name = "ModuleWithConflictMatrix", conflict_matrix = {"rule1,rule2" = 2 : i32, "rule1,method1" = 0 : i32, "rule2,method1" = 1 : i32}} : () -> ()

// CHECK: "txn.module"
"txn.module"() ({
  "txn.rule"() ({
    "txn.yield"() : () -> ()
  }) {sym_name = "simpleRule"} : () -> ()

  "txn.action_method"() ({
    "txn.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "simpleMethod"} : () -> ()

  "txn.schedule"() {methods = [@simpleRule, @simpleMethod]} : () -> ()
}) {sym_name = "ModuleWithoutAttributes"} : () -> ()