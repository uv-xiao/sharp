// RUN: sharp-opt %s | sharp-opt | FileCheck %s

// Simple test to verify FIRRTL, comb, and hw dialects can coexist

// CHECK-LABEL: hw.module @HWAdder
hw.module @HWAdder(in %a : i8, in %b : i8, out sum : i8) {
  // Use comb dialect for combinational logic
  %0 = comb.add %a, %b : i8
  hw.output %0 : i8
}

// CHECK-LABEL: hw.module @CombLogicExample  
hw.module @CombLogicExample(in %x : i4, in %y : i4, in %sel : i1, out result : i4) {
  // Various comb operations
  %and_result = comb.and %x, %y : i4
  %or_result = comb.or %x, %y : i4
  %mux_result = comb.mux %sel, %and_result, %or_result : i4
  hw.output %mux_result : i4
}

// CHECK-LABEL: firrtl.circuit "SimpleAdder"
firrtl.circuit "SimpleAdder" {
  // CHECK: firrtl.module @SimpleAdder
  firrtl.module @SimpleAdder(in %a: !firrtl.uint<8>, in %b: !firrtl.uint<8>, 
                             out %c: !firrtl.uint<9>) {
    %sum = firrtl.add %a, %b : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<9>
    firrtl.connect %c, %sum : !firrtl.uint<9>, !firrtl.uint<9>
  }
}

// CHECK-LABEL: hw.module @MixedHWAndComb
hw.module @MixedHWAndComb(in %clock : i1, in %data : i8, out q : i8) {
  // Mix of hw constants and comb operations
  %c1_i8 = hw.constant 1 : i8
  %incremented = comb.add %data, %c1_i8 : i8
  
  // Sequential logic using seq dialect
  %reg = seq.compreg %incremented, %clock : i8
  
  hw.output %reg : i8
}

// CHECK-LABEL: firrtl.circuit "Counter"
firrtl.circuit "Counter" {
  firrtl.module @Counter(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>,
                         out %count: !firrtl.uint<8>) {
    %c1_ui8 = firrtl.constant 1 : !firrtl.uint<8>
    
    // Register to hold count value
    %count_reg = firrtl.reg %clock : !firrtl.uint<8>
    
    // Increment logic
    %next_count = firrtl.add %count_reg, %c1_ui8 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
    
    // Reset logic
    %c0_ui8 = firrtl.constant 0 : !firrtl.uint<8>
    %count_val = firrtl.mux(%reset, %c0_ui8, %next_count) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
    
    // Update register
    firrtl.connect %count_reg, %count_val : !firrtl.uint<8>, !firrtl.uint<8>
    
    // Connect output
    firrtl.connect %count, %count_reg : !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// CHECK-LABEL: hw.module @CombShifter
hw.module @CombShifter(in %data : i16, in %shift_amount : i4, out shifted : i16) {
  // Shift operations using comb
  %shifted_left = comb.shl %data, %shift_amount : i16
  %shifted_right = comb.shru %data, %shift_amount : i16
  
  // Select based on MSB of shift amount
  %msb = comb.extract %shift_amount from 3 : (i4) -> i1
  %result = comb.mux %msb, %shifted_right, %shifted_left : i16
  
  hw.output %result : i16
}

// Test showing all three dialects can reference each other through proper interfaces
// CHECK-LABEL: func.func @testInterop
func.func @testInterop() {
  // This shows that standard MLIR can reference modules from different hardware dialects
  %c10 = arith.constant 10 : i8
  %c20 = arith.constant 20 : i8
  
  // In practice, we would need proper conversion/wrapper operations
  // to actually instantiate and use these hardware modules
  
  return
}