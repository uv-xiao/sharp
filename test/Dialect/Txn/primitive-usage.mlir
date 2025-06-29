// RUN: sharp-opt %s | FileCheck %s

// Test using primitives in modules

// CHECK-LABEL: txn.module @SystemWithPrimitives
txn.module @SystemWithPrimitives {
  // Instantiate a register primitive
  // CHECK: %[[REG:.*]] = txn.instance @dataReg of @RegisterPrimitive : !txn.module<"RegisterPrimitive">
  %dataReg = txn.instance @dataReg of @RegisterPrimitive : !txn.module<"RegisterPrimitive">
  
  // Instantiate a wire primitive  
  // CHECK: %[[WIRE:.*]] = txn.instance @dataWire of @WirePrimitive : !txn.module<"WirePrimitive">
  %dataWire = txn.instance @dataWire of @WirePrimitive : !txn.module<"WirePrimitive">
  
  // Rule that transfers data from register to wire
  // CHECK: txn.rule @transferData
  txn.rule @transferData {
    // Read from register
    // CHECK: %[[VAL:.*]] = txn.call @dataReg.read() : () -> i32
    %val = txn.call @dataReg.read() : () -> i32
    
    // Write to wire
    // CHECK: txn.call @dataWire.write(%[[VAL]]) : (i32) -> ()
    txn.call @dataWire.write(%val) : (i32) -> ()
    
    txn.yield
  }
  
  // Action method that writes to register
  // CHECK: txn.action_method @storeValue(%[[ARG:.*]]: i32)
  txn.action_method @storeValue(%value: i32) {
    // CHECK: txn.call @dataReg.write(%[[ARG]]) : (i32) -> ()
    txn.call @dataReg.write(%value) : (i32) -> ()
    txn.return
  }
  
  // Value method that reads from wire
  // CHECK: txn.value_method @readOutput() -> i32
  txn.value_method @readOutput() -> i32 {
    // CHECK: %[[OUT:.*]] = txn.call @dataWire.read() : () -> i32
    %output = txn.call @dataWire.read() : () -> i32
    // CHECK: txn.return %[[OUT]] : i32
    txn.return %output : i32
  }
  
  // CHECK: txn.schedule [@transferData, @storeValue, @readOutput]
  txn.schedule [@transferData, @storeValue, @readOutput]
}

// Primitives need to be defined for the above to work
txn.primitive @RegisterPrimitive type = "hw" interface = !txn.module<"RegisterPrimitive"> {
  txn.fir_value_method @read() : () -> i32
  txn.fir_action_method @write() : (i32) -> ()
  txn.clock_by @clk
  txn.reset_by @rst
  txn.schedule [@read, @write] {
    conflict_matrix = {
      "read,read" = 3 : i32,
      "read,write" = 3 : i32,
      "write,read" = 3 : i32,
      "write,write" = 2 : i32
    }
  }
}

txn.primitive @WirePrimitive type = "hw" interface = !txn.module<"WirePrimitive"> {
  txn.fir_value_method @read() : () -> i32
  txn.fir_action_method @write() : (i32) -> ()
  txn.clock_by @clk
  txn.reset_by @rst
  txn.schedule [@read, @write] {
    conflict_matrix = {
      "read,read" = 3 : i32,
      "read,write" = 0 : i32,
      "write,read" = 1 : i32,
      "write,write" = 2 : i32
    }
  }
}