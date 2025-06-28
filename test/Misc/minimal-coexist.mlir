// RUN: sharp-opt %s | FileCheck %s

// Minimal test showing FIRRTL and HW/Comb dialects can coexist

// CHECK: hw.module @HWModule
hw.module @HWModule(in %a : i8, in %b : i8, out sum : i8) {
  %0 = comb.add %a, %b : i8
  hw.output %0 : i8
}

// CHECK: firrtl.circuit "FIRRTLModule"
firrtl.circuit "FIRRTLModule" {
  firrtl.module @FIRRTLModule(in %a: !firrtl.uint<8>, in %b: !firrtl.uint<8>, out %sum: !firrtl.uint<9>) {
    %0 = firrtl.add %a, %b : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<9>
    firrtl.connect %sum, %0 : !firrtl.uint<9>
  }
}

// Key observation: Both hardware description styles can exist in the same IR