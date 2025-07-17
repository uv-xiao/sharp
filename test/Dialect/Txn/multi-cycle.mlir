// RUN: sharp-opt %s | FileCheck %s
// RUN: sharp-opt %s --mlir-print-op-generic | FileCheck %s --check-prefix=GENERIC

// Define primitives used in tests
txn.primitive @Register type = "hw" interface = index {
  txn.fir_value_method @read() : () -> i32
  txn.fir_action_method @write() : (i32) -> ()
  txn.clock_by @clk
  txn.reset_by @rst
  txn.schedule [@write] {
    conflict_matrix = {
      "read,read" = 3 : i32,
      "read,write" = 3 : i32,
      "write,read" = 3 : i32,
      "write,write" = 2 : i32
    }
  }
} {firrtl.impl = "Register_impl"}

txn.primitive @SpecFIFO type = "spec" interface = index {
  txn.fir_value_method @notEmpty() : () -> i1
  txn.fir_value_method @notFull() : () -> i1
  txn.fir_action_method @enqueue() : (i32) -> ()
  txn.fir_action_method @dequeue() : () -> i32
  txn.clock_by @clk
  txn.reset_by @rst
  txn.schedule [@enqueue, @dequeue] {
    conflict_matrix = {
      "enqueue,dequeue" = 2 : i32,
      "dequeue,enqueue" = 2 : i32,
      "enqueue,enqueue" = 2 : i32,
      "dequeue,dequeue" = 2 : i32
    }
  }
} {firrtl.impl = "SpecFIFO_impl", software_semantics = {fifo_empty = true, fifo_data = []}}

txn.primitive @Memory type = "hw" interface = index {
  txn.fir_value_method @read() : (i32) -> i32
  txn.fir_action_method @write() : (i32, i32) -> ()
  txn.clock_by @clk
  txn.reset_by @rst
  txn.schedule [@write] {
    conflict_matrix = {
      "read,read" = 3 : i32,
      "read,write" = 3 : i32,
      "write,read" = 3 : i32,
      "write,write" = 2 : i32
    }
  }
} {firrtl.impl = "Memory_impl"}

// Test basic future and launch operations
txn.module @MultiCycleExample {
    %reg = txn.instance @reg of @Register : index
    %fifo = txn.instance @fifo of @SpecFIFO : index
    
    // CHECK-LABEL: txn.action_method @staticLaunch
    txn.action_method @staticLaunch(%data: i32) {
        // Per-cycle action
        txn.call @reg::@write(%data) : (i32) -> ()
        
        // Multi-cycle region
        // CHECK: txn.future {
        txn.future {
            // Static latency launch
            // CHECK: %{{.*}} = txn.launch after 3 {
            %done1 = txn.launch after 3 {
                %c42 = arith.constant 42 : i32
                txn.call @reg::@write(%c42) : (i32) -> ()
                txn.yield
            }
        }
        txn.return
    }
    
    // CHECK-LABEL: txn.action_method @dynamicLaunch
    txn.action_method @dynamicLaunch(%data: i32) {
        // CHECK: txn.future {
        txn.future {
            // Static launch first
            // CHECK: %[[DONE1:.*]] = txn.launch after 2 {
            %done1 = txn.launch after 2 {
                txn.call @fifo::@enqueue(%data) : (i32) -> ()
                txn.yield
            }
            
            // Dynamic launch waiting for done1
            // CHECK: %[[DONE2:.*]] = txn.launch until %[[DONE1]] {
            %done2 = txn.launch until %done1 {
                %v = txn.call @fifo::@dequeue() : () -> i32
                txn.call @reg::@write(%v) : (i32) -> ()
                txn.yield
            }
        }
        txn.return
    }
    
    // CHECK-LABEL: txn.action_method @combinedLaunch
    txn.action_method @combinedLaunch(%data: i32) -> i1 {
        // CHECK: txn.future {
        txn.future {
            // First static launch
            // CHECK: %[[DONE1:.*]] = txn.launch after 1 {
            %done1 = txn.launch after 1 {
                txn.call @fifo::@enqueue(%data) : (i32) -> ()
                txn.yield
            }
            
            // Combined: wait for done1 then 2 more cycles
            // CHECK: %[[DONE2:.*]] = txn.launch until %[[DONE1]] after 2 {
            %done2 = txn.launch until %done1 after 2 {
                %v = txn.call @fifo::@dequeue() : () -> i32
                %c10 = arith.constant 10 : i32
                %sum = arith.addi %v, %c10 : i32
                txn.call @reg::@write(%sum) : (i32) -> ()
                txn.yield
            }
            
            // Dynamic launch with completion signal
            // CHECK: %[[DONE3:.*]] = txn.launch until %[[DONE2]] {
            %done3 = txn.launch until %done2 {
                %final = txn.call @reg::@read() : () -> i32
                txn.call @fifo::@enqueue(%final) : (i32) -> ()
                txn.yield
            }
        }
        
        // Return completion status
        %true = arith.constant 1 : i1
        txn.return %true : i1
    }
    
    // CHECK-LABEL: txn.rule @multiCycleRule
    txn.rule @multiCycleRule {
        %guard = arith.constant 1 : i1
        
        // CHECK: txn.future {
        txn.future {
            // Dequeue with retry - store in register first
            // CHECK: %[[DONE1:.*]] = txn.launch after 1 {
            %done1 = txn.launch after 1 {
                %v = txn.call @fifo::@dequeue() : () -> i32
                txn.call @reg::@write(%v) : (i32) -> ()
                txn.yield
            }
            
            // Process after 3 cycles
            // CHECK: %[[DONE2:.*]] = txn.launch until %[[DONE1]] after 3 {
            %done2 = txn.launch until %done1 after 3 {
                %data = txn.call @reg::@read() : () -> i32
                %doubled = arith.muli %data, %data : i32
                txn.call @reg::@write(%doubled) : (i32) -> ()
                txn.yield
            }
        }
    }
    
    txn.schedule [@staticLaunch, @dynamicLaunch, @combinedLaunch, @multiCycleRule]
}

// Test empty future block
txn.module @EmptyFuture {
    // CHECK-LABEL: txn.action_method @emptyFuture
    txn.action_method @emptyFuture() {
        // CHECK: txn.future {
        // CHECK-NEXT: }
        txn.future {
        }
        txn.return
    }
    
    txn.schedule [@emptyFuture]
}

// Test nested launches
txn.module @NestedLaunches {
    %mem = txn.instance @mem of @Memory : index
    
    // CHECK-LABEL: txn.action_method @nestedLaunch
    txn.action_method @nestedLaunch(%addr: i32) {
        // CHECK: txn.future {
        txn.future {
            // Outer launch
            // CHECK: %[[OUTER:.*]] = txn.launch after 1 {
            %outer = txn.launch after 1 {
                // Read from memory
                %data = txn.call @mem::@read(%addr) : (i32) -> i32
                
                // Inner future with launches
                // CHECK: txn.future {
                txn.future {
                    // CHECK: %[[INNER:.*]] = txn.launch after 2 {
                    %inner = txn.launch after 2 {
                        %c1 = arith.constant 1 : i32
                        %incremented = arith.addi %data, %c1 : i32
                        txn.call @mem::@write(%addr, %incremented) : (i32, i32) -> ()
                        txn.yield
                    }
                }
                txn.yield
            }
        }
        txn.return
    }
    
    txn.schedule [@nestedLaunch]
}

// GENERIC-LABEL: "txn.module"
// GENERIC: "txn.future"
// GENERIC: "txn.launch"