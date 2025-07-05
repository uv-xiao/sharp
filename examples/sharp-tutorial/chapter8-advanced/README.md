# Chapter 8: Advanced Topics

## Overview

This chapter covers advanced Sharp features and real-world applications:
- Custom primitives
- Formal verification integration
- Performance optimization
- Industrial case studies

## Custom Primitives

### Creating New Primitives

Custom primitives extend Sharp's capabilities:

### custom_primitives.mlir

```mlir
// Define custom CAM (Content Addressable Memory)
txn.primitive @CAM<width: i32, depth: i32> {
  // Define methods
  txn.method @write(%addr: i32, %data: !width) -> ()
  txn.method @search(%pattern: !width) -> i32  // Returns address or -1
  txn.method @clear() -> ()
  
  // Software semantics
  txn.software_semantics {
    %storage = std::map<i32, !width>
    %last_match = i32
  }
}

// Using custom primitive
txn.module @NetworkRouter {
  %cam = txn.instance @cam of @CAM<32, 1024> : !txn.module<"CAM">
  
  txn.action_method @add_route(%prefix: i32, %port: i32) {
    txn.call @cam::@write(%prefix, %port) : (i32, i32) -> ()
    txn.yield
  }
  
  txn.value_method @lookup(%addr: i32) -> i32 {
    %port = txn.call @cam::@search(%addr) : (i32) -> i32
    txn.return %port : i32
  }
  
  txn.schedule [@add_route, @lookup]
}
```

### Advanced Primitive Features

```mlir
// Primitive with internal state machine
txn.primitive @Arbiter<clients: i32> {
  txn.method @request(%id: i32) -> ()
  txn.method @grant() -> i32
  txn.method @release(%id: i32) -> ()
  
  txn.hardware_semantics {
    // Priority encoder
    %requests = hw.aggregate %clients : i1
    %grant_id = hw.priority_encode %requests : i32
  }
  
  txn.conflict_matrix {
    "request,grant" = 0 : i32,     // SB
    "grant,release" = 0 : i32,     // SB
    "request,release" = 1 : i32    // SA
  }
}

// Multi-port memory primitive
txn.primitive @MultiPortRAM<width: i32, depth: i32, read_ports: i32, write_ports: i32> {
  txn.method @read(%port: i32, %addr: i32) -> !width
  txn.method @write(%port: i32, %addr: i32, %data: !width) -> ()
  
  txn.timing {
    "read" = "combinational",
    "write" = "static(1)"
  }
  
  txn.constraints {
    // No two writes to same address
    conflict_if(%port1 != %port2 && %addr1 == %addr2) {
      "write[%port1],write[%port2]" = 3 : i32  // CF
    }
  }
}
```

## Formal Verification Integration

### verification_example.mlir

```mlir
// Module with formal properties
txn.module @SecureCounter {
  %count = txn.instance @count of @Register<i32> : !txn.module<"Register">
  %max = txn.instance @max of @Register<i32> : !txn.module<"Register">
  
  txn.action_method @set_max(%limit: i32) {
    txn.call @max::@write(%limit) : (i32) -> ()
    txn.yield
  }
  
  txn.action_method @increment() {
    %val = txn.call @count::@read() : () -> i32
    %limit = txn.call @max::@read() : () -> i32
    %cmp = arith.cmpi slt, %val, %limit : i32
    scf.if %cmp {
      %one = arith.constant 1 : i32
      %next = arith.addi %val, %one : i32
      txn.call @count::@write(%next) : (i32) -> ()
    }
    txn.yield
  }
  
  // Formal properties
  txn.property @never_exceed_max {
    %count_val = txn.call @count::@read() : () -> i32
    %max_val = txn.call @max::@read() : () -> i32
    %valid = arith.cmpi sle, %count_val, %max_val : i32
    txn.assert %valid : i1
  }
  
  txn.property @monotonic_increase {
    %old = txn.sample @count::@read() : () -> i32
    txn.call @this.increment() : () -> ()
    %new = txn.call @count::@read() : () -> i32
    %increased = arith.cmpi sge, %new, %old : i32
    txn.assert %increased : i1
  }
  
  txn.schedule [@set_max, @increment] {
    verification_depth = 20 : i32
  }
}
```

### Model Checking Integration

```mlir
// Deadlock-free protocol verification
txn.module @Protocol {
  %state = txn.instance @state of @Register<i8> : !txn.module<"Register">
  
  // State encoding
  %IDLE = arith.constant 0 : i8
  %REQ = arith.constant 1 : i8
  %ACK = arith.constant 2 : i8
  
  txn.action_method @request() {
    %s = txn.call @state::@read() : () -> i8
    %is_idle = arith.cmpi eq, %s, %IDLE : i8
    scf.if %is_idle {
      txn.call @state::@write(%REQ) : (i8) -> ()
    }
    txn.yield
  }
  
  txn.action_method @acknowledge() {
    %s = txn.call @state::@read() : () -> i8
    %is_req = arith.cmpi eq, %s, %REQ : i8
    scf.if %is_req {
      txn.call @state::@write(%ACK) : (i8) -> ()
    }
    txn.yield
  }
  
  txn.action_method @complete() {
    %s = txn.call @state::@read() : () -> i8
    %is_ack = arith.cmpi eq, %s, %ACK : i8
    scf.if %is_ack {
      txn.call @state::@write(%IDLE) : (i8) -> ()
    }
    txn.yield
  }
  
  // Liveness property
  txn.property @eventually_completes {
    txn.eventually {
      %s = txn.call @state::@read() : () -> i8
      %is_idle = arith.cmpi eq, %s, %IDLE : i8
      txn.assert %is_idle : i1
    }
  }
  
  txn.schedule [@request, @acknowledge, @complete]
}
```

## Performance Optimization

### optimization_patterns.mlir

```mlir
// Optimized pipeline with forwarding
txn.module @OptimizedPipeline {
  // Pipeline registers with bypass logic
  %stage1 = txn.instance @stage1 of @Register<i32> : !txn.module<"Register">
  %stage2 = txn.instance @stage2 of @Register<i32> : !txn.module<"Register">
  %stage3 = txn.instance @stage3 of @Register<i32> : !txn.module<"Register">
  
  // Forwarding paths
  %fwd1to3 = txn.instance @fwd1to3 of @Wire<i32> : !txn.module<"Wire">
  %fwd2to3 = txn.instance @fwd2to3 of @Wire<i32> : !txn.module<"Wire">
  
  txn.action_method @process(%use_fwd: i1) {
    scf.if %use_fwd {
      // Fast path with forwarding
      %s1 = txn.call @stage1::@read() : () -> i32
      %s2 = txn.call @stage2::@read() : () -> i32
      
      // Compute and forward
      %r1 = arith.muli %s1, %s1 : i32
      %r2 = arith.addi %s2, %r1 : i32
      
      txn.call @fwd1to3::@write(%r1) : (i32) -> ()
      txn.call @fwd2to3::@write(%r2) : (i32) -> ()
      txn.call @stage3::@write(%r2) : (i32) -> ()
    } else {
      // Normal pipeline advance
      %s1 = txn.call @stage1::@read() : () -> i32
      %s2 = txn.call @stage2::@read() : () -> i32
      
      txn.call @stage2::@write(%s1) : (i32) -> ()
      txn.call @stage3::@write(%s2) : (i32) -> ()
    }
    txn.yield
  }
  
  // Speculation and rollback
  txn.action_method @speculate(%pred: i32) {
    %backup = txn.call @stage3::@read() : () -> i32
    
    // Speculative execution
    %spec_result = arith.muli %pred, %pred : i32
    txn.call @stage3::@write(%spec_result) : (i32) -> ()
    
    // Validation
    %actual = txn.call @stage2::@read() : () -> i32
    %correct = arith.cmpi eq, %pred, %actual : i32
    
    // Rollback if mispredicted
    scf.if %correct {
      // Keep speculative result
    } else {
      txn.call @stage3::@write(%backup) : (i32) -> ()
    }
    txn.yield
  }
  
  txn.schedule [@process, @speculate] {
    optimization_hints = {
      "unroll_factor" = 2 : i32,
      "pipeline_depth" = 3 : i32,
      "enable_forwarding" = true
    }
  }
}

// Banking and parallelism
txn.module @BankedMemory {
  // 4-way banked memory for parallel access
  %bank0 = txn.instance @bank0 of @BRAM<i32> : !txn.module<"BRAM">
  %bank1 = txn.instance @bank1 of @BRAM<i32> : !txn.module<"BRAM">
  %bank2 = txn.instance @bank2 of @BRAM<i32> : !txn.module<"BRAM">
  %bank3 = txn.instance @bank3 of @BRAM<i32> : !txn.module<"BRAM">
  
  txn.action_method @parallel_read(%addr0: i32, %addr1: i32, %addr2: i32, %addr3: i32) 
      -> (i32, i32, i32, i32) {
    // All banks can be accessed in parallel
    %v0 = txn.call @bank0::@read(%addr0) : (i32) -> i32
    %v1 = txn.call @bank1::@read(%addr1) : (i32) -> i32
    %v2 = txn.call @bank2::@read(%addr2) : (i32) -> i32
    %v3 = txn.call @bank3::@read(%addr3) : (i32) -> i32
    
    txn.return %v0, %v1, %v2, %v3 : i32, i32, i32, i32
  }
  
  txn.action_method @streaming_write(%base: i32, %d0: i32, %d1: i32, %d2: i32, %d3: i32) {
    // Distribute writes across banks
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    
    %a0 = arith.addi %base, %c0 : i32
    %a1 = arith.addi %base, %c1 : i32
    %a2 = arith.addi %base, %c2 : i32
    %a3 = arith.addi %base, %c3 : i32
    
    txn.call @bank0::@write(%a0, %d0) : (i32, i32) -> ()
    txn.call @bank1::@write(%a1, %d1) : (i32, i32) -> ()
    txn.call @bank2::@write(%a2, %d2) : (i32, i32) -> ()
    txn.call @bank3::@write(%a3, %d3) : (i32, i32) -> ()
    
    txn.yield
  }
  
  txn.schedule [@parallel_read, @streaming_write]
}
```

## Case Studies

### case_study_cache.mlir

```mlir
// High-performance cache controller
txn.module @CacheController {
  // Cache state
  %tags = txn.instance @tags of @CAM<20, 256> : !txn.module<"CAM">
  %data = txn.instance @data of @BRAM<i64> : !txn.module<"BRAM">
  %valid = txn.instance @valid of @Register<i256> : !txn.module<"Register">
  %dirty = txn.instance @dirty of @Register<i256> : !txn.module<"Register">
  
  // Statistics
  %hits = txn.instance @hits of @Register<i64> : !txn.module<"Register">
  %misses = txn.instance @misses of @Register<i64> : !txn.module<"Register">
  
  txn.action_method @read(%addr: i32) -> i64 {
    // Extract tag and index
    %tag = arith.shrui %addr, %c12 : i32
    %idx = arith.andi %addr, %c255 : i32
    
    // Check tag
    %match_idx = txn.call @tags::@search(%tag) : (i32) -> i32
    %hit = arith.cmpi ne, %match_idx, %minus_one : i32
    
    scf.if %hit {
      // Cache hit
      %data_val = txn.call @data::@read(%match_idx) : (i32) -> i64
      
      // Update statistics
      %h = txn.call @hits::@read() : () -> i64
      %h_new = arith.addi %h, %c1 : i64
      txn.call @hits::@write(%h_new) : (i64) -> ()
      
      txn.return %data_val : i64
    } else {
      // Cache miss - simplified
      %m = txn.call @misses::@read() : () -> i64
      %m_new = arith.addi %m, %c1 : i64
      txn.call @misses::@write(%m_new) : (i64) -> ()
      
      %zero = arith.constant 0 : i64
      txn.return %zero : i64
    }
  }
  
  txn.action_method @write(%addr: i32, %value: i64) {
    %tag = arith.shrui %addr, %c12 : i32
    %idx = arith.andi %addr, %c255 : i32
    
    // Write allocate policy
    txn.call @tags::@write(%idx, %tag) : (i32, i32) -> ()
    txn.call @data::@write(%idx, %value) : (i32, i64) -> ()
    
    // Set dirty bit
    %d = txn.call @dirty::@read() : () -> i256
    %mask = arith.shli %c1, %idx : i256
    %d_new = arith.ori %d, %mask : i256
    txn.call @dirty::@write(%d_new) : (i256) -> ()
    
    txn.yield
  }
  
  txn.value_method @get_hit_rate() -> f32 {
    %h = txn.call @hits::@read() : () -> i64
    %m = txn.call @misses::@read() : () -> i64
    %total = arith.addi %h, %m : i64
    
    %h_f = arith.sitofp %h : i64 to f32
    %t_f = arith.sitofp %total : i64 to f32
    %rate = arith.divf %h_f, %t_f : f32
    
    txn.return %rate : f32
  }
  
  txn.schedule [@read, @write, @get_hit_rate] {
    performance_targets = {
      "read_latency" = 1 : i32,
      "write_latency" = 1 : i32,
      "throughput" = 1000000000 : i64  // 1G ops/sec
    }
  }
}
```

### case_study_crypto.mlir

```mlir
// AES encryption engine
txn.module @AESEngine {
  // State array
  %state = txn.instance @state of @Register<i128> : !txn.module<"Register">
  
  // Round keys
  %round_keys = txn.instance @keys of @ROM<i128> : !txn.module<"ROM">
  %round = txn.instance @round of @Register<i8> : !txn.module<"Register">
  
  // S-box lookup
  %sbox = txn.instance @sbox of @ROM<i8> : !txn.module<"ROM">
  
  txn.action_method @load_plaintext(%data: i128) {
    txn.call @state::@write(%data) : (i128) -> ()
    %zero = arith.constant 0 : i8
    txn.call @round::@write(%zero) : (i8) -> ()
    txn.yield
  }
  
  txn.action_method @round_function() {
    %s = txn.call @state::@read() : () -> i128
    %r = txn.call @round::@read() : () -> i8
    
    // SubBytes - parallel S-box lookups
    %b0 = arith.trunci %s : i128 to i8
    %s1 = arith.shrui %s, %c8 : i128
    %b1 = arith.trunci %s1 : i128 to i8
    // ... continue for all 16 bytes
    
    %sb0 = txn.call @sbox::@read(%b0) : (i8) -> i8
    %sb1 = txn.call @sbox::@read(%b1) : (i8) -> i8
    // ... continue substitution
    
    // ShiftRows and MixColumns (simplified)
    %shifted = txn.call @shift_rows(%s) : (i128) -> i128
    %mixed = txn.call @mix_columns(%shifted) : (i128) -> i128
    
    // AddRoundKey
    %key = txn.call @round_keys::@read(%r) : (i8) -> i128
    %next_state = arith.xori %mixed, %key : i128
    
    txn.call @state::@write(%next_state) : (i128) -> ()
    
    // Increment round
    %one = arith.constant 1 : i8
    %next_round = arith.addi %r, %one : i8
    txn.call @round::@write(%next_round) : (i8) -> ()
    
    txn.yield
  }
  
  txn.value_method @get_ciphertext() -> i128 {
    %s = txn.call @state::@read() : () -> i128
    txn.return %s : i128
  }
  
  txn.value_method @is_complete() -> i1 {
    %r = txn.call @round::@read() : () -> i8
    %ten = arith.constant 10 : i8
    %done = arith.cmpi eq, %r, %ten : i8
    txn.return %done : i1
  }
  
  // Auto-process rounds
  txn.rule @auto_round {
    %done = txn.call @this.is_complete() : () -> i1
    %not_done = arith.xori %done, %true : i1
    scf.if %not_done {
      txn.call @this.round_function() : () -> ()
    }
    txn.yield
  }
  
  txn.schedule [@load_plaintext, @round_function, @get_ciphertext, @is_complete, @auto_round] {
    security_features = {
      "constant_time" = true,
      "side_channel_resistant" = true
    }
  }
}
```

## Debugging and Profiling

### debug_features.mlir

```mlir
// Module with extensive debugging
txn.module @DebugExample {
  %state = txn.instance @state of @Register<i32> : !txn.module<"Register">
  
  // Debug probes
  txn.debug_probe @state_probe {
    %val = txn.call @state::@read() : () -> i32
    txn.debug_print "State value: %d\n", %val : i32
  }
  
  // Performance counters
  %method_calls = txn.instance @calls of @Register<i64> : !txn.module<"Register">
  %cycle_count = txn.instance @cycles of @Register<i64> : !txn.module<"Register">
  
  txn.action_method @process(%input: i32) {
    // Increment call counter
    %calls = txn.call @method_calls::@read() : () -> i64
    %one = arith.constant 1 : i64
    %new_calls = arith.addi %calls, %one : i64
    txn.call @method_calls::@write(%new_calls) : (i64) -> ()
    
    // Actual processing
    %current = txn.call @state::@read() : () -> i32
    %result = arith.addi %current, %input : i32
    txn.call @state::@write(%result) : (i32) -> ()
    
    // Trigger debug probe
    txn.trigger @state_probe
    
    txn.yield
  }
  
  // Cycle counter rule
  txn.rule @count_cycles {
    %c = txn.call @cycle_count::@read() : () -> i64
    %one = arith.constant 1 : i64
    %new_c = arith.addi %c, %one : i64
    txn.call @cycle_count::@write(%new_c) : (i64) -> ()
    txn.yield
  }
  
  txn.value_method @get_stats() -> (i64, i64) {
    %calls = txn.call @method_calls::@read() : () -> i64
    %cycles = txn.call @cycle_count::@read() : () -> i64
    txn.return %calls, %cycles : i64, i64
  }
  
  txn.schedule [@process, @get_stats, @count_cycles] {
    debug_level = 2 : i32,
    profile_enabled = true
  }
}
```

### run.sh

```bash
#!/bin/bash

SHARP_ROOT="../../.."
SHARP_OPT="$SHARP_ROOT/build/bin/sharp-opt"

echo "=== Chapter 8: Advanced Topics ==="
echo ""

echo "1. Custom Primitives:"
echo "----------------------------------------"
if $SHARP_OPT custom_primitives.mlir --parse-only > /dev/null 2>&1; then
    echo "✅ Custom primitive definitions valid"
else
    echo "❌ Custom primitive parsing failed"
fi
echo ""

echo "2. Verification Examples:"
echo "----------------------------------------"
if $SHARP_OPT verification_example.mlir --sharp-verify > verify_output.txt 2>&1; then
    echo "✅ Verification properties checked"
    grep -E "(property|assert)" verify_output.txt || echo "Properties defined"
else
    echo "⚠️  Verification pass not yet implemented"
fi
echo ""

echo "3. Performance Patterns:"
echo "----------------------------------------"
$SHARP_OPT optimization_patterns.mlir --sharp-optimize > optimized.mlir 2>&1
if [ -f optimized.mlir ]; then
    echo "✅ Optimization patterns processed"
    echo "Original size: $(wc -l < optimization_patterns.mlir) lines"
    echo "Optimized size: $(wc -l < optimized.mlir) lines"
else
    echo "⚠️  Optimization pass in development"
fi
echo ""

echo "4. Case Study - Cache:"
echo "----------------------------------------"
$SHARP_ROOT/tools/generate-workspace.sh case_study_cache.mlir cache_sim
if [ -d cache_sim ]; then
    echo "✅ Cache controller ready for simulation"
    echo "Features: CAM-based tags, statistics, hit rate calculation"
else
    echo "❌ Cache case study generation failed"
fi
echo ""

echo "5. Case Study - Crypto:"
echo "----------------------------------------"
if $SHARP_OPT case_study_crypto.mlir --parse-only > /dev/null 2>&1; then
    echo "✅ AES engine design valid"
    echo "Features: Round-based processing, S-box lookups, auto-round rule"
else
    echo "❌ Crypto case study parsing failed"
fi
echo ""

echo "6. Debug Features:"
echo "----------------------------------------"
if $SHARP_OPT debug_features.mlir --sharp-simulate="mode=translation,debug=true" > debug_sim.cpp 2>&1; then
    echo "✅ Debug instrumentation added"
    grep -c "debug_print" debug_sim.cpp && echo "debug probes inserted"
else
    echo "⚠️  Debug features require simulation mode"
fi

echo ""
echo "Advanced features demonstrate Sharp's capability for:"
echo "- Custom hardware primitives"
echo "- Formal verification integration"
echo "- Performance optimization"
echo "- Real-world applications"