// RUN: sharp-opt %s --sharp-simulate | FileCheck %s
// RUN: sharp-opt %s --sharp-simulate="mode=translation output=test.cpp" | FileCheck %s --check-prefix=TRANS

// Simple counter module for testing TxnSimulate pass
txn.module @Counter {
    txn.state @count : i32
    
    txn.value_method @getValue() -> i32 {
        %v = txn.read @count : !txn.ref<i32>
        txn.return %v : i32
    }
    
    txn.action_method @increment() {
        %v = txn.read @count : !txn.ref<i32>
        %one = arith.constant 1 : i32
        %next = arith.addi %v, %one : i32
        txn.write @count, %next : !txn.ref<i32>, i32
    }
    
    txn.rule @autoIncrement {
        %v = txn.call @getValue() : () -> i32
        %max = arith.constant 100 : i32
        %cond = arith.cmpi ult, %v, %max : i32
        txn.if %cond {
            txn.call @increment() : () -> ()
        }
    }
    
    %s = txn.schedule @Counter conflicts {
        "autoIncrement" = []
    } {
        txn.scheduled "autoIncrement"
    }
}

// CHECK: txn.module @Counter
// TRANS: Generated C++ code to test.cpp