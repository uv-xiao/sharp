// RUN: sharp-opt %s | sharp-opt | FileCheck %s

// Test demonstrating that FIRRTL, comb, hw, and seq dialects can coexist
// in the same IR, showing interoperability between different hardware dialects

// CHECK-LABEL: hw.module @HWCombExample
hw.module @HWCombExample(in %a : i8, in %b : i8, in %sel : i1, out result : i8) {
  // Combinational logic using comb dialect
  %sum = comb.add %a, %b : i8
  %diff = comb.sub %a, %b : i8
  %result = comb.mux %sel, %sum, %diff : i8
  hw.output %result : i8
}

// CHECK-LABEL: hw.module @SequentialExample  
hw.module @SequentialExample(in %clock : !seq.clock, in %reset : i1, in %data : i8, out q : i8) {
  // Constants
  %c0_i8 = hw.constant 0 : i8
  %c1_i8 = hw.constant 1 : i8
  
  // Combinational logic
  %incremented = comb.add %data, %c1_i8 : i8
  
  // Sequential logic - register with reset
  %reg = seq.compreg %incremented, %clock reset %reset, %c0_i8 : i8
  
  hw.output %reg : i8
}

// CHECK-LABEL: firrtl.circuit "FIRRTLAdder"
firrtl.circuit "FIRRTLAdder" {
  // CHECK: firrtl.module @FIRRTLAdder
  firrtl.module @FIRRTLAdder(in %a: !firrtl.uint<8>, in %b: !firrtl.uint<8>, 
                             out %sum: !firrtl.uint<9>) {
    // FIRRTL addition produces wider result
    %0 = firrtl.add %a, %b : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<9>
    firrtl.connect %sum, %0 : !firrtl.uint<9>, !firrtl.uint<9>
  }
  
  // CHECK: firrtl.module @FIRRTLMux
  firrtl.module @FIRRTLMux(in %sel: !firrtl.uint<1>, in %a: !firrtl.uint<4>, 
                           in %b: !firrtl.uint<4>, out %out: !firrtl.uint<4>) {
    %result = firrtl.mux(%sel, %a, %b) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
    firrtl.connect %out, %result : !firrtl.uint<4>, !firrtl.uint<4>
  }
}

// CHECK-LABEL: hw.module @ComplexCombLogic
hw.module @ComplexCombLogic(in %a : i16, in %b : i16, out result : i16) {
  // Bitwise operations
  %and_result = comb.and %a, %b : i16
  %or_result = comb.or %a, %b : i16
  %xor_result = comb.xor %a, %b : i16
  
  // Arithmetic operations
  %sum = comb.add %a, %b : i16
  
  // Comparison
  %is_equal = comb.icmp eq %a, %b : i16
  
  // Conditional selection
  %selected = comb.mux %is_equal, %and_result, %xor_result : i16
  
  hw.output %selected : i16
}

// CHECK-LABEL: firrtl.circuit "FIRRTLCounter"
firrtl.circuit "FIRRTLCounter" {
  firrtl.module @FIRRTLCounter(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>,
                               in %enable: !firrtl.uint<1>, out %count: !firrtl.uint<8>) {
    // Constants
    %c0_ui8 = firrtl.constant 0 : !firrtl.uint<8>
    %c1_ui8 = firrtl.constant 1 : !firrtl.uint<8>
    
    // Register for counter
    %count_reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
    
    // Increment logic
    %incremented_wide = firrtl.add %count_reg, %c1_ui8 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<9>
    
    // Truncate to 8 bits
    %incremented = firrtl.bits %incremented_wide 7 to 0 : (!firrtl.uint<9>) -> !firrtl.uint<8>
    
    // Conditional increment based on enable
    %next_val = firrtl.mux(%enable, %incremented, %count_reg) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
    
    // Reset logic
    %count_val = firrtl.mux(%reset, %c0_ui8, %next_val) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
    
    // Update register
    firrtl.connect %count_reg, %count_val : !firrtl.uint<8>, !firrtl.uint<8>
    
    // Output
    firrtl.connect %count, %count_reg : !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// CHECK-LABEL: hw.module @MixedBitOperations
hw.module @MixedBitOperations(in %data : i32, out high : i16, out low : i16) {
  // Extract high and low parts using comb
  %high_bits = comb.extract %data from 16 : (i32) -> i16
  %low_bits = comb.extract %data from 0 : (i32) -> i16
  
  // Could also concatenate
  %reconstructed = comb.concat %high_bits, %low_bits : i16, i16
  
  hw.output %high_bits, %low_bits : i16, i16
}

// CHECK-LABEL: func.func @demonstrateCoexistence
func.func @demonstrateCoexistence() {
  // This function shows that all these hardware dialects can coexist
  // in the same module. In practice, conversions between FIRRTL types
  // and hw/comb types would require specific lowering passes.
  
  %c42 = arith.constant 42 : i32
  %c7 = arith.constant 7 : i8
  
  // The actual instantiation and connection of these hardware modules
  // would require appropriate instance operations and type conversions
  
  return
}

// Additional test with seq dialect features
// CHECK-LABEL: hw.module @SequentialStateMachine
hw.module @SequentialStateMachine(in %clock : !seq.clock, in %reset : i1, 
                                  in %input : i2, out state : i2) {
  // State encoding
  %IDLE = hw.constant 0 : i2
  %STATE1 = hw.constant 1 : i2
  %STATE2 = hw.constant 2 : i2
  %STATE3 = hw.constant 3 : i2
  
  // State register
  %current_state = seq.compreg %next_state, %clock reset %reset, %IDLE : i2
  
  // Next state logic using comb operations
  %is_idle = comb.icmp eq %current_state, %IDLE : i2
  %is_state1 = comb.icmp eq %current_state, %STATE1 : i2
  %is_state2 = comb.icmp eq %current_state, %STATE2 : i2
  
  // State transitions based on input
  %trans_from_idle = comb.mux %is_idle, %STATE1, %current_state : i2
  %trans_from_s1 = comb.mux %is_state1, %STATE2, %trans_from_idle : i2
  %trans_from_s2 = comb.mux %is_state2, %STATE3, %trans_from_s1 : i2
  
  // Input-dependent transitions
  %input_is_zero = comb.icmp eq %input, %IDLE : i2
  %next_state = comb.mux %input_is_zero, %IDLE, %trans_from_s2 : i2
  
  hw.output %current_state : i2
}