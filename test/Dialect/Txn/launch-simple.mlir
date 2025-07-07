// RUN: sharp-opt %s

txn.module @Simple {
    %reg = txn.instance @reg of @Register<i32> : !txn.module<"Register">
    
    txn.action_method @test() {
        // First test: Just future block with yield
        %cond = arith.constant 1 : i1
        txn.future {
            %done1 = txn.launch after 2 {
                txn.yield
            }
            %done2 = txn.launch until %cond {
                txn.yield
            }
            %done3 = txn.launch until %done1 after 1 {
                txn.yield
            }
        }
        txn.return
    }
    
    txn.schedule [@test]
}