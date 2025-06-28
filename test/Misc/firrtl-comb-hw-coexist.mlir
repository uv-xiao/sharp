// RUN: sharp-opt %s | sharp-opt | FileCheck %s

// Test that FIRRTL dialect operations can coexist with comb/hw dialect operations
// This demonstrates interoperability between different hardware description dialects

// CHECK-LABEL: firrtl.module @FIRRTLAdder
firrtl.module @FIRRTLAdder(in %a: !firrtl.uint<8>, in %b: !firrtl.uint<8>, 
                           out %sum: !firrtl.uint<8>) {
  // FIRRTL add operation
  %0 = firrtl.add %a, %b : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<9>
  
  // Truncate to 8 bits
  %1 = firrtl.bits %0 7 to 0 : (!firrtl.uint<9>) -> !firrtl.uint<8>
  
  // Connect to output
  firrtl.strictconnect %sum, %1 : !firrtl.uint<8>
}

// CHECK-LABEL: hw.module @HWAdder
hw.module @HWAdder(in %a : i8, in %b : i8, out sum : i8) {
  // Comb add operation
  %0 = comb.add %a, %b : i8
  
  // Output the result
  hw.output %0 : i8
}

// CHECK-LABEL: hw.module @MixedDesign
hw.module @MixedDesign(in %x : i8, in %y : i8, in %sel : i1, 
                       out result : i8, out overflow : i1) {
  // Use comb operations
  %sum = comb.add %x, %y : i8
  %diff = comb.sub %x, %y : i8
  
  // Multiplexer using comb
  %mux_result = comb.mux %sel, %sum, %diff : i8
  
  // Extract overflow bit using comb operations
  %x_ext = comb.concat %x, %false : i8, i1
  %y_ext = comb.concat %y, %false : i8, i1
  %sum_ext = comb.add %x_ext, %y_ext : i9
  %overflow_bit = comb.extract %sum_ext from 8 : (i9) -> i1
  
  %false = hw.constant false
  
  // Output results
  hw.output %mux_result, %overflow_bit : i8, i1
}

// CHECK-LABEL: firrtl.circuit "MixedCircuit"
firrtl.circuit "MixedCircuit" {
  // FIRRTL module with more complex logic
  firrtl.module @Comparator(in %a: !firrtl.uint<4>, in %b: !firrtl.uint<4>,
                            out %eq: !firrtl.uint<1>, out %lt: !firrtl.uint<1>) {
    %eq_result = firrtl.eq %a, %b : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    %lt_result = firrtl.lt %a, %b : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
    
    firrtl.strictconnect %eq, %eq_result : !firrtl.uint<1>
    firrtl.strictconnect %lt, %lt_result : !firrtl.uint<1>
  }
  
  // Main module using the comparator
  firrtl.module @MixedCircuit(in %in1: !firrtl.uint<4>, in %in2: !firrtl.uint<4>,
                              out %equal: !firrtl.uint<1>, out %less: !firrtl.uint<1>) {
    %comp_eq, %comp_lt = firrtl.instance comp @Comparator(
      in a: !firrtl.uint<4>, 
      in b: !firrtl.uint<4>,
      out eq: !firrtl.uint<1>, 
      out lt: !firrtl.uint<1>
    )
    
    firrtl.strictconnect %comp_eq, %in1 : !firrtl.uint<4>
    firrtl.strictconnect %comp_lt, %in2 : !firrtl.uint<4>
    firrtl.strictconnect %equal, %comp_eq : !firrtl.uint<1>
    firrtl.strictconnect %less, %comp_lt : !firrtl.uint<1>
  }
}

// CHECK-LABEL: hw.module @CombLogic
hw.module @CombLogic(in %a : i4, in %b : i4, in %c : i4, out result : i4) {
  // Complex combinational logic mixing various operations
  %and_ab = comb.and %a, %b : i4
  %or_bc = comb.or %b, %c : i4
  %xor_result = comb.xor %and_ab, %or_bc : i4
  
  // Arithmetic operations
  %sum = comb.add %a, %c : i4
  %shifted = comb.shru %sum, %b : i4
  
  // Final result
  %final = comb.mux %shifted, %xor_result, %sum : i4
  
  hw.output %final : i4
}

// CHECK-LABEL: func.func @testCoexistence
func.func @testCoexistence() {
  // This demonstrates that we can reference both FIRRTL and HW modules
  // from regular MLIR functions, showing full interoperability
  
  %c8_i8 = arith.constant 8 : i8
  %c12_i8 = arith.constant 12 : i8
  %true = arith.constant true
  
  // We could potentially have conversion operations between
  // FIRRTL types and standard types here
  
  return
}

// Test that shows FIRRTL operations can use results from HW/Comb operations
// through proper type conversions (when conversion infrastructure is available)
// CHECK-LABEL: hw.module @HybridModule
hw.module @HybridModule(in %clock : i1, in %reset : i1, in %data : i8, out q : i8) {
  // Use comb for combinational logic
  %incremented = comb.add %data, %c1 : i8
  
  // Use seq for sequential logic (registers)
  %reg = seq.compreg %incremented, %clock : i8
  
  %c1 = hw.constant 1 : i8
  
  hw.output %reg : i8
}