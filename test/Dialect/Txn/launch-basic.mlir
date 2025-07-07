// RUN: sharp-opt %s | FileCheck %s

txn.module @LaunchTest {
    %reg = txn.instance @reg of @Register<i32> : !txn.module<"Register">
    
    // Test 1: Static latency only
    txn.action_method @staticOnly() {
        txn.future {
            // CHECK: %{{.*}} = txn.launch {latency = 3 : i32} {
            %done = txn.launch {latency=3} {
                %c42 = arith.constant 42 : i32
                txn.call @reg.write(%c42) : (i32) -> ()
                txn.yield
            }
        }
        txn.return
    }
    
    // Test 2: Dynamic condition only
    txn.action_method @dynamicOnly() {
        %true = arith.constant 1 : i1
        txn.future {
            // CHECK: %{{.*}} = txn.launch until %{{.*}} {
            %done = txn.launch until %true {
                %c99 = arith.constant 99 : i32
                txn.call @reg.write(%c99) : (i32) -> ()
                txn.yield
            }
        }
        txn.return
    }
    
    // Test 3: Both condition and latency
    txn.action_method @both() {
        %false = arith.constant 0 : i1
        txn.future {
            // CHECK: %{{.*}} = txn.launch until %{{.*}} {latency = 5 : i32} {
            %done = txn.launch until %false {latency=5} {
                %c111 = arith.constant 111 : i32
                txn.call @reg.write(%c111) : (i32) -> ()
                txn.yield
            }
        }
        txn.return
    }
    
    txn.schedule [@staticOnly, @dynamicOnly, @both]
}