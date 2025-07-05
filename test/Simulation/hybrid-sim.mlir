// RUN: not sharp-opt %s --sharp-simulate-hybrid 2>&1 | FileCheck %s

// This test is a placeholder for future hybrid simulation functionality
// CHECK: Unknown command line argument '--sharp-simulate-hybrid'
txn.module @TestBench {
  // Instance of RTL DUT (will be converted to RTL)
  %dut = txn.instance @dut of @Adder attributes {sharp.impl = "rtl"}
  
  // Test vector generation
  txn.rule @generate_tests {
    %tests = arith.constant dense<[[1, 2], [5, 7], [10, 15]]> : tensor<3x2xi32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    
    scf.for %i = %c0 to %c3 step %c1 {
      %test = tensor.extract %tests[%i, %c0] : tensor<3x2xi32>
      %a = tensor.extract %test[%c0] : tensor<2xi32>
      %b = tensor.extract %test[%c1] : tensor<2xi32>
      
      // Call RTL module through bridge
      %sum = txn.call %dut::@add(%a, %b) : (i32, i32) -> i32
      
      // Verify result
      %expected = arith.addi %a, %b : i32
      %correct = arith.cmpi eq, %sum, %expected : i32
      cf.assert %correct, "Addition result incorrect"
    }
  }
}

// Simple adder to be implemented in RTL
txn.module @Adder attributes {sharp.rtl_export = true} {
  // Will be converted to RTL combinational logic
  txn.value_method @add(%a: i32, %b: i32) -> i32 attributes {timing = "combinational"} {
    %sum = arith.addi %a, %b : i32
    txn.return %sum : i32
  }
}

// Bridge configuration
sharp.bridge @tl_to_rtl {
  // Map TL method calls to RTL signals
  method_mapping = {
    "add" : {
      inputs = ["a", "b"],
      outputs = ["sum"],
      handshake = "none"  // Combinational
    }
  }
}

// Hybrid simulation configuration
sharp.sim @hybrid {
  testbench = @TestBench,      // TL simulation
  rtl_modules = [@Adder],      // Convert to RTL
  bridge = @tl_to_rtl,         // Bridge config
  max_cycles = 100 : i64,
  // Arcilator settings
  arcilator_opts = {
    optimize = true,
    trace = true,
    trace_format = "vcd"
  }
}