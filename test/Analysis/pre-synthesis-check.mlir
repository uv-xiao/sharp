// RUN: sharp-opt --sharp-pre-synthesis-check %s 2>&1 | FileCheck %s

// Test pre-synthesis checking pass

// Test case: spec primitive
txn.primitive @SpecPrimitive type = "spec" interface = !txn.module<"SpecPrimitive"> {
  txn.fir_value_method @getValue() {firrtl.port = "data"} : () -> i32
  txn.schedule [@getValue] {conflict_matrix = {}}
}

// Test case: multi-cycle rule
"txn.module"() ({
  "txn.rule"() ({
    "txn.return"() : () -> ()
  }) {sym_name = "multiCycleRule", timing = "static(2)"} : () -> ()
  
  "txn.schedule"() {actions = [@multiCycleRule]} : () -> ()
}) {sym_name = "MultiCycleRuleModule"} : () -> ()

// Test case: multi-cycle method
"txn.module"() ({
  "txn.value_method"() ({
    %c0 = "arith.constant"() {value = 0 : i32} : () -> i32
    "txn.return"(%c0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "multiCycleMethod", timing = "dynamic"} : () -> ()
  
  "txn.schedule"() {actions = [@multiCycleMethod]} : () -> ()
}) {sym_name = "MultiCycleMethodModule"} : () -> ()

// Test case: primitive without firrtl.impl
txn.primitive @NoFirrtlImpl type = "hw" interface = !txn.module<"NoFirrtlImpl"> {
  txn.fir_value_method @read() {firrtl.port = "data"} : () -> i32
  txn.schedule [@read] {conflict_matrix = {}}
}

// Test case: module instantiating non-synthesizable module
txn.module @ParentModule {
  %inst = txn.instance @spec_inst of @SpecPrimitive : !txn.module<"SpecPrimitive">
  
  txn.rule @useSpec {
    %val = txn.call @spec_inst::@getValue() : () -> i32
    txn.return
  }
  
  txn.schedule [@useSpec] {conflict_matrix = {}}
}

// Test case: synthesizable primitive
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

// Test case: synthesizable module with combinational timing
txn.module @GoodModule {
  %inst = txn.instance @prim of @GoodPrimitive : !txn.module<"GoodPrimitive">
  
  // Combinational rule (default timing)
  txn.rule @combRule {
    %val = txn.call @prim::@read() : () -> i32
    txn.return
  }
  
  // Explicit combinational timing using generic syntax
  "txn.value_method"() ({
    %val = "txn.call"() {callee = @prim::@read} : () -> i32
    "txn.return"(%val) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "combMethod", timing = "combinational"} : () -> ()
  
  txn.schedule [@combRule, @combMethod] {conflict_matrix = {}}
}

// CHECK-DAG: error: multi-cycle rules are not yet supported for synthesis
// CHECK-DAG: error: multi-cycle methods are not yet supported for synthesis
// CHECK-DAG: error: synthesizable primitive lacks firrtl.impl attribute
// CHECK-DAG: error: Module 'SpecPrimitive' is non-synthesizable: spec primitive type
// CHECK-DAG: error: Module 'MultiCycleRuleModule' is non-synthesizable: contains multi-cycle operations
// CHECK-DAG: error: Module 'MultiCycleMethodModule' is non-synthesizable: contains multi-cycle operations
// CHECK-DAG: error: Module 'NoFirrtlImpl' is non-synthesizable: spec primitive type
// CHECK-DAG: error: Module 'ParentModule' is non-synthesizable: instantiates non-synthesizable module 'SpecPrimitive'
// CHECK-NOT: error: Module 'GoodPrimitive'
// CHECK-NOT: error: Module 'GoodModule'