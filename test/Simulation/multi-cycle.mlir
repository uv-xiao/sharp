// RUN: sharp-opt %s --sharp-simulate | FileCheck %s

// Module with multi-cycle operations
// CHECK-LABEL: txn.module @Pipeline
txn.module @Pipeline {
  %fifo_in = txn.primitive @SpecFIFO<i32>
  %fifo_out = txn.primitive @SpecFIFO<i32>
  
  // Multi-cycle processing operation
  // CHECK: txn.action_method @process
  txn.action_method @process() attributes {sharp.latency = 5 : i32} {
    // Check if we can dequeue
    %can_deq = txn.call %fifo_in::@canDeq() : () -> i1
    scf.if %can_deq {
      // Start multi-cycle operation
      %data = txn.call %fifo_in::@deq() : () -> i32
      
      // Simulate multi-cycle processing with spec primitive
      %processed = txn.spec.multi_cycle<5> {
        // Stage 1: Fetch
        %stage1 = arith.muli %data, %data : i32
        
        // Stage 2-4: Process (implicit)
        
        // Stage 5: Write back
        txn.yield %stage1 : i32
      } : i32
      
      // Enqueue result
      txn.call %fifo_out::@enq(%processed) : (i32) -> ()
    }
  }
  
  // Input method
  txn.action_method @input(%data: i32) {
    txn.call %fifo_in::@enq(%data) : (i32) -> ()
  }
  
  // Output method  
  txn.value_method @output() -> i32 {
    %can_deq = txn.call %fifo_out::@canDeq() : () -> i1
    %result = scf.if %can_deq -> i32 {
      %data = txn.call %fifo_out::@deq() : () -> i32
      scf.yield %data : i32
    } else {
      %zero = arith.constant 0 : i32
      scf.yield %zero : i32
    }
    txn.return %result : i32
  }
}

// CHECK-LABEL: txn.module @PipelineTest
txn.module @PipelineTest {
  %pipeline = txn.instance @pipe of @Pipeline
  
  // Generate test data
  txn.rule @generate_data {
    %i = arith.constant 1 : i32
    %limit = arith.constant 10 : i32
    
    scf.for %idx = %i to %limit step %i {
      txn.call %pipeline::@input(%idx) : (i32) -> ()
    }
  }
  
  // Process data
  txn.rule @process {
    txn.call %pipeline::@process() : () -> ()
  }
  
  // Check output
  txn.rule @check_output {
    %result = txn.call %pipeline::@output() : () -> i32
    %zero = arith.constant 0 : i32
    %has_output = arith.cmpi ne, %result, %zero : i32
    
    scf.if %has_output {
      // Verify it's a square
      txn.spec.assert {
        %sqrt = math.sqrt %result : i32
        %squared = arith.muli %sqrt, %sqrt : i32
        %eq = arith.cmpi eq, %squared, %result : i32
        txn.yield %eq : i1
      } message "Output should be a perfect square"
    }
  }
}

// Simulation with performance tracking
sharp.sim @perf_test {
  module = @PipelineTest,
  max_cycles = 1000 : i64,
  profile = true,
  // Track method execution counts and latencies
  track_methods = ["Pipeline::process", "Pipeline::input", "Pipeline::output"]
}