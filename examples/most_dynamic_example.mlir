// Example demonstrating most-dynamic will-fire mode
// This shows how primitive-level conflicts are handled

// Define a simple Register primitive
txn.primitive @Register type = "hw" interface = !txn.module<"Register"> {
  txn.fir_value_method @read() : () -> i32
  txn.fir_action_method @write() : (i32) -> ()
  txn.clock_by @clk
  txn.reset_by @rst
  txn.schedule [@write] {
    conflict_matrix = {
      "read,read" = 3 : i32,    // CF - multiple reads OK
      "read,write" = 3 : i32,   // CF - read doesn't block write
      "write,read" = 3 : i32,   // CF - write doesn't block read
      "write,write" = 2 : i32   // C - only one write per cycle
    }
  }
} {firrtl.impl = "Register_impl"}

// Module using the primitive
txn.module @MostDynamicExample {
  %reg1 = txn.instance @reg1 of @Register : !txn.module<"Register">
  %reg2 = txn.instance @reg2 of @Register : !txn.module<"Register">
  
  // Action that writes to reg1
  txn.action_method @writeReg1(%val: i32) {
    txn.call @reg1::@write(%val) : (i32) -> ()
    txn.yield
  }
  
  // Action that conditionally writes to both registers
  txn.action_method @conditionalWrite(%val: i32, %cond: i1) {
    txn.if %cond {
      txn.call @reg1::@write(%val) : (i32) -> ()
      txn.yield
    } else {
      txn.call @reg2::@write(%val) : (i32) -> ()
      txn.yield
    }
    txn.yield
  }
  
  // Action that always writes to reg2
  txn.action_method @writeReg2(%val: i32) {
    txn.call @reg2::@write(%val) : (i32) -> ()
    txn.yield
  }
  
  // Schedule with most-dynamic mode enabled
  txn.schedule [@writeReg1, @conditionalWrite, @writeReg2] {
    conflict_matrix = {
      // These actions may conflict depending on runtime conditions
      "writeReg1,conditionalWrite" = 3 : i32,  // CF at module level
      "conditionalWrite,writeReg2" = 3 : i32,  // CF at module level
      "writeReg1,writeReg2" = 3 : i32          // CF - different registers
    }
  }
}