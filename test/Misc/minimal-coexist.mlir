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

// CHECK-LABEL: txn.module @BasicModule {
txn.module @BasicModule {
  // CHECK: %{{.*}} = txn.instance @helper of @Helper : !txn.module<"Helper">
  %helper = txn.instance @helper of @Helper : !txn.module<"Helper">
  
  // CHECK: txn.value_method @getValue() -> i32
  txn.value_method @getValue() -> i32 {
    %c42 = arith.constant 42 : i32
    txn.return %c42 : i32
  }
  
  // CHECK: txn.action_method @setValue(%{{.*}}: !firrtl.uint<8>) -> !firrtl.uint<8>
  txn.action_method @setValue(%val: !firrtl.uint<8>) -> !firrtl.uint<8> {
    txn.return %val : !firrtl.uint<8>
  }
  
  // CHECK: txn.schedule [@getValue, @setValue]
  txn.schedule [@getValue, @setValue]
}
// Key observation: Both hardware description styles can exist in the same IR