// Example 1: Primitive instance method calls
txn.module @PrimitiveCallExample {
  %reg = txn.instance @reg of @Register<i32> : !txn.module<"Register">
  
  txn.action_method @writeToReg() {
    %c42 = arith.constant 42 : i32
    // This call fails to convert properly
    txn.call @reg::@write(%c42) : (i32) -> ()
    txn.return
  }
  
  txn.rule @doWrite {
    txn.call @writeToReg() : () -> ()
    txn.yield
  }
  
  txn.schedule [@doWrite]
}

// Example 2: txn.if with yields
txn.module @IfYieldExample {
  txn.rule @conditionalRule {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %cond = arith.cmpi slt, %c1, %c2 : i32
    
    txn.if %cond {
      // Do something
      txn.yield  // This yield causes issues
    } else {
      // Do something else  
      txn.yield  // This yield causes issues
    }
    txn.yield  // Rule terminator
  }
  
  txn.schedule [@conditionalRule]
}

// Example 3: Missing schedule
txn.module @NoScheduleExample {
  txn.rule @myRule {
    txn.yield
  }
  // Missing: txn.schedule [@myRule]
}