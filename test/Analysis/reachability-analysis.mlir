// RUN: sharp-opt --sharp-reachability-analysis %s | FileCheck %s

// Test reachability analysis for method calls in actions

// CHECK-LABEL: txn.module @SimpleConditional
txn.module @SimpleConditional {
  %reg = txn.instance @state of @Register : !txn.module<"Register">
  
  // CHECK-LABEL: txn.rule @conditionalRule
  txn.rule @conditionalRule {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %val = txn.call @state::@read() : () -> i32
    %cond = arith.cmpi "eq", %val, %c0 : i32
    
    txn.if %cond {
      // CHECK: txn.call @state::@write({{.*}}) {{.*}} {reachability_condition = "cond_0"}
      txn.call @state::@write(%c1) : (i32) -> ()
      txn.yield
    } else {
      // CHECK: txn.call @state::@write({{.*}}) {{.*}} {reachability_condition = "!cond_0"}
      txn.call @state::@write(%c0) : (i32) -> ()
      txn.yield
    }
    txn.return
  }
  txn.schedule [@conditionalRule]
}

// CHECK-LABEL: txn.module @NestedConditionals
txn.module @NestedConditionals {
  %reg1 = txn.instance @r1 of @Register : !txn.module<"Register">
  %reg2 = txn.instance @r2 of @Register : !txn.module<"Register">
  
  // CHECK-LABEL: txn.action_method @nestedAction
  txn.action_method @nestedAction(%arg0: i32) -> () {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    
    %cond1 = arith.cmpi "eq", %arg0, %c0 : i32
    txn.if %cond1 {
      // CHECK: txn.call @r1::@write({{.*}}) {{.*}} {reachability_condition = "cond_{{[0-9]+}}"}
      txn.call @r1::@write(%c0) : (i32) -> ()
      txn.yield
    } else {
      %cond2 = arith.cmpi "sgt", %arg0, %c10 : i32
      txn.if %cond2 {
        // CHECK: txn.call @r2::@write(%arg0) {{.*}} {reachability_condition = "!cond_{{[0-9]+}} && cond_{{[0-9]+}}"}
        txn.call @r2::@write(%arg0) : (i32) -> ()
        // CHECK: txn.call @r1::@write({{.*}}) {{.*}} {reachability_condition = "!cond_{{[0-9]+}} && cond_{{[0-9]+}}"}
        txn.call @r1::@write(%c10) : (i32) -> ()
        txn.yield
      } else {
        txn.yield
      }
      txn.yield
    }
    txn.return
  }
  txn.schedule [@nestedAction]
}

// CHECK-LABEL: txn.module @UnconditionalCalls
txn.module @UnconditionalCalls {
  %wire = txn.instance @w of @Wire : !txn.module<"Wire">
  
  // CHECK-LABEL: txn.value_method @getValue
  txn.value_method @getValue() -> i32 {
    // CHECK: txn.call @w::@read() {{.*}}
    %val = txn.call @w::@read() : () -> i32
    txn.return %val : i32
  }
  
  // CHECK-LABEL: txn.rule @alwaysRule
  txn.rule @alwaysRule {
    %c42 = arith.constant 42 : i32
    // CHECK: txn.call @w::@write({{.*}}) {{.*}} {reachability_condition = "true"}
    txn.call @w::@write(%c42) : (i32) -> ()
    txn.return
  }
  txn.schedule [@getValue, @alwaysRule]
}

// CHECK-LABEL: txn.module @ComplexControlFlow
txn.module @ComplexControlFlow {
  %reg = txn.instance @counter of @Register : !txn.module<"Register">
  
  // CHECK-LABEL: txn.rule @complexRule
  txn.rule @complexRule {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c5 = arith.constant 5 : i32
    %c10 = arith.constant 10 : i32
    
    %val = txn.call @counter::@read() : () -> i32
    
    // First level condition
    %cond1 = arith.cmpi "slt", %val, %c5 : i32
    txn.if %cond1 {
      // CHECK: txn.call @counter::@write({{.*}}) {{.*}} {reachability_condition = "cond_{{[0-9]+}}"}
      txn.call @counter::@write(%c0) : (i32) -> ()
      txn.yield
    } else {
      // Second level condition in else branch
      %cond2 = arith.cmpi "slt", %val, %c10 : i32
      txn.if %cond2 {
        %inc = arith.addi %val, %c1 : i32
        // CHECK: txn.call @counter::@write({{.*}}) {{.*}} {reachability_condition = "!cond_{{[0-9]+}} && cond_{{[0-9]+}}"}
        txn.call @counter::@write(%inc) : (i32) -> ()
        txn.yield
      } else {
        // CHECK: txn.call @counter::@write({{.*}}) {{.*}} {reachability_condition = "!cond_{{[0-9]+}} && !cond_{{[0-9]+}}"}
        txn.call @counter::@write(%c10) : (i32) -> ()
        txn.yield
      }
      txn.yield
    }
    
    txn.return
  }
  txn.schedule [@complexRule]
}

// Primitive modules (simplified for testing)
txn.module @Register {
  txn.value_method @read() -> i32 {
    %0 = arith.constant 0 : i32
    txn.return %0 : i32
  }
  txn.action_method @write(%val: i32) -> () {
    txn.return
  }
  txn.schedule [@read, @write]
}

txn.module @Wire {
  txn.value_method @read() -> i32 {
    %0 = arith.constant 0 : i32
    txn.return %0 : i32
  }
  txn.action_method @write(%val: i32) -> () {
    txn.return
  }
  txn.schedule [@read, @write]
}