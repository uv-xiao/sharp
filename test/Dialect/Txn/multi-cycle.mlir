// RUN: sharp-opt %s | FileCheck %s
// RUN: sharp-opt %s --mlir-print-op-generic | FileCheck %s --check-prefix=GENERIC

// Test basic future and launch operations
txn.module @MultiCycleExample {
    %reg = txn.instance @reg of @Register<i32> : !txn.module<"Register">
    %fifo = txn.instance @fifo of @SpecFIFO<i32> : !txn.module<"SpecFIFO">
    
    // CHECK-LABEL: txn.action_method @staticLaunch
    txn.action_method @staticLaunch(%data: i32) {
        // Per-cycle action
        txn.call @reg.write(%data) : (i32) -> ()
        
        // Multi-cycle region
        // CHECK: txn.future {
        txn.future {
            // Static latency launch
            // CHECK: %{{.*}} = txn.launch {latency = 3 : i32} {
            %done1 = txn.launch {latency=3} {
                %c42 = arith.constant 42 : i32
                txn.call @reg.write(%c42) : (i32) -> ()
                txn.yield
            }
            // CHECK: txn.yield
        }
        txn.return
    }
    
    // CHECK-LABEL: txn.action_method @dynamicLaunch
    txn.action_method @dynamicLaunch(%data: i32) {
        // CHECK: txn.future {
        txn.future {
            // Static launch first
            // CHECK: %[[DONE1:.*]] = txn.launch {latency = 2 : i32} {
            %done1 = txn.launch {latency=2} {
                txn.call @fifo.enqueue(%data) : (i32) -> ()
                txn.yield
            }
            
            // Dynamic launch waiting for done1
            // CHECK: %[[DONE2:.*]] = txn.launch until %[[DONE1]] {
            %done2 = txn.launch until %done1 {
                %v = txn.call @fifo.dequeue() : () -> i32
                txn.call @reg.write(%v) : (i32) -> ()
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
            // CHECK: %[[DONE1:.*]] = txn.launch {latency = 1 : i32} {
            %done1 = txn.launch {latency=1} {
                txn.call @fifo.enqueue(%data) : (i32) -> ()
                txn.yield
            }
            
            // Combined: wait for done1 then 2 more cycles
            // CHECK: %[[DONE2:.*]] = txn.launch until %[[DONE1]] {latency = 2 : i32} {
            %done2 = txn.launch until %done1 {latency=2} {
                %v = txn.call @fifo.dequeue() : () -> i32
                %c10 = arith.constant 10 : i32
                %sum = arith.addi %v, %c10 : i32
                txn.call @reg.write(%sum) : (i32) -> ()
                txn.yield
            }
            
            // Dynamic launch with completion signal
            // CHECK: %[[DONE3:.*]] = txn.launch until %[[DONE2]] {
            %done3 = txn.launch until %done2 {
                %final = txn.call @reg.read() : () -> i32
                txn.call @fifo.enqueue(%final) : (i32) -> ()
                txn.yield
            }
        }
        
        // Return completion status
        %true = arith.constant true : i1
        txn.return %true : i1
    }
    
    // CHECK-LABEL: txn.rule @multiCycleRule
    txn.rule @multiCycleRule {
        %guard = txn.call @fifo.notEmpty() : () -> i1
        
        txn.if %guard {
            // CHECK: txn.future {
            txn.future {
                // Dequeue with retry - store in register first
                // CHECK: %[[DONE1:.*]] = txn.launch {latency = 1 : i32} {
                %done1 = txn.launch {latency=1} {
                    %v = txn.call @fifo.dequeue() : () -> i32
                    txn.call @reg.write(%v) : (i32) -> ()
                    txn.yield
                }
                
                // Process after 3 cycles
                // CHECK: %[[DONE2:.*]] = txn.launch until %[[DONE1]] {latency = 3 : i32} {
                %done2 = txn.launch until %done1 {latency=3} {
                    %data = txn.call @reg.read() : () -> i32
                    %doubled = arith.muli %data, %data : i32
                    txn.call @reg.write(%doubled) : (i32) -> ()
                    txn.yield
                }
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
    %mem = txn.instance @mem of @Memory<i32> : !txn.module<"Memory">
    
    // CHECK-LABEL: txn.action_method @nestedLaunch
    txn.action_method @nestedLaunch(%addr: i32) {
        // CHECK: txn.future {
        txn.future {
            // Outer launch
            // CHECK: %[[OUTER:.*]] = txn.launch {latency = 1 : i32} {
            %outer = txn.launch {latency=1} {
                // Read from memory
                %data = txn.call @mem.read(%addr) : (i32) -> i32
                
                // Inner future with launches
                // CHECK: txn.future {
                txn.future {
                    // CHECK: %[[INNER:.*]] = txn.launch {latency = 2 : i32} {
                    %inner = txn.launch {latency=2} {
                        %c1 = arith.constant 1 : i32
                        %incremented = arith.addi %data, %c1 : i32
                        txn.call @mem.write(%addr, %incremented) : (i32, i32) -> ()
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