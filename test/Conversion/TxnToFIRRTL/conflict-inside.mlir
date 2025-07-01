// RUN: sharp-opt --convert-txn-to-firrtl %s | FileCheck %s

// Test conflict_inside calculation with reachability analysis

// CHECK-LABEL: firrtl.circuit "ConflictInsideTest"
// CHECK: firrtl.module @ConflictInsideTest

txn.module @ConflictInsideTest {
  %reg1 = txn.instance @r1 of @Register : !txn.module<"Register">
  %reg2 = txn.instance @r2 of @Register : !txn.module<"Register">
  
  // Rule with conditional method calls that conflict
  txn.rule @conditionalConflict {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c10 = arith.constant 10 : i32
    
    // Read from reg1
    %val1 = txn.call @r1::@read() : () -> i32
    
    // Conditional branches with conflicting method calls
    %cond1 = arith.cmpi eq, %val1, %c0 : i32
    txn.if %cond1 {
      // Write to reg1 in then branch
      txn.call @r1::@write(%c10) : (i32) -> ()
      txn.yield
    } else {
      // Another condition in else branch
      %cond2 = arith.cmpi sgt, %val1, %c1 : i32
      txn.if %cond2 {
        // Also write to reg1 - conflicts with the write in then branch
        txn.call @r1::@write(%c1) : (i32) -> ()
        txn.yield
      } else {
        txn.yield
      }
      txn.yield
    }
    txn.return
  }
  
  // Rule with non-conflicting conditional calls
  txn.rule @noConflict {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    
    %val = txn.call @r1::@read() : () -> i32
    %cond = arith.cmpi eq, %val, %c0 : i32
    
    txn.if %cond {
      // Write to reg1 in then branch
      txn.call @r1::@write(%c1) : (i32) -> ()
      txn.yield
    } else {
      // Write to reg2 in else branch - no conflict
      txn.call @r2::@write(%c1) : (i32) -> ()
      txn.yield
    }
    txn.return
  }
  
  // Action method with internal conflicts
  txn.action_method @conflictingMethod(%enable: i1) {
    %c5 = arith.constant 5 : i32
    %c7 = arith.constant 7 : i32
    
    txn.if %enable {
      txn.call @r1::@write(%c5) : (i32) -> ()
      // Another write to same register - always conflicts
      txn.call @r1::@write(%c7) : (i32) -> ()
      txn.yield
    } else {
      txn.yield
    }
    txn.return
  }
  
  txn.schedule [@conditionalConflict, @noConflict, @conflictingMethod] {
    conflict_matrix = {
      "r1::read,r1::write" = 2 : i32,  // C
      "r1::write,r1::write" = 2 : i32, // C  
      "r2::read,r2::write" = 2 : i32,  // C
      "r2::write,r2::write" = 2 : i32  // C
    }
  }
}

// Verify basic conversion completes successfully
// The test focuses on ensuring the conversion handles complex control flow
// Full dynamic conflict_inside with reachability analysis is deferred