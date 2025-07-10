// A module that toggles between 0 and 1
txn.module @Toggle {
  // We'll use a Register primitive to store state
  %state = txn.instance @state of @Register<i1> : !txn.module<"Register">
  
  // Value method to read the current state
  txn.value_method @read() -> i1 {
    %val = txn.call @state::@read() : () -> i1
    txn.return %val : i1
  }
  
  // Action method to toggle the state
  txn.action_method @toggle() {
    %current = txn.call @state::@read() : () -> i1
    %one = arith.constant 1 : i1
    %new = arith.xori %current, %one : i1
    txn.call @state::@write(%new) : (i1) -> ()
    txn.yield
  }

  txn.rule @default {
    %current = txn.call @state::@read() : () -> i1
    txn.call @state::@write(%current) : (i1) -> ()
  }
  
  // Schedule declares all methods/rules
  txn.schedule [@toggle, @default]
}