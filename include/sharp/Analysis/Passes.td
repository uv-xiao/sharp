//===- Passes.td - Sharp Analysis passes definition ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Sharp analysis passes.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_ANALYSIS_PASSES
#define SHARP_ANALYSIS_PASSES

include "mlir/Pass/PassBase.td"

def ConflictMatrixInference : Pass<"sharp-infer-conflict-matrix", "mlir::ModuleOp"> {
  let summary = "Infer and complete conflict matrix for txn modules";
  let description = [{
    This pass analyzes Sharp Txn modules and infers the conflict matrix (CM)
    relationships between actions (rules and action methods). The pass implements
    the following inference rules:

    1. Any action conflicts (C) with itself
    2. If two actions are both SA and SB, they conflict (C)
    3. If two actions call the same action method of the same instance, they conflict (C)
    4. Conflict propagation through method calls:
       - m0 SA m1 => a0 SA a1
       - m0 SB m1 => a0 SB a1
       - m0 C m1 => a0 C a1
    5. Default to conflict-free (CF) if relationship cannot be determined

    The pass operates on schedule operations within modules and completes any
    missing conflict relationships based on these rules.
  }];
  
  let constructor = "mlir::sharp::createConflictMatrixInferencePass()";
  
  let dependentDialects = ["::sharp::txn::TxnDialect"];
}

def PreSynthesisCheck : Pass<"sharp-pre-synthesis-check", "mlir::ModuleOp"> {
  let summary = "Check for non-synthesizable constructs before FIRRTL translation";
  let description = [{
    This pass analyzes Sharp Txn modules to identify non-synthesizable constructs
    that would prevent successful translation to FIRRTL/Verilog. The pass checks for:

    1. Non-synthesizable (spec) primitives
    2. Multi-cycle rules/methods (timing != "combinational")
    3. Modules that instantiate non-synthesizable submodules or primitives
    4. Operations from non-synthesizable dialects (only txn, firrtl, and builtin allowed)

    The pass marks modules as non-synthesizable by adding a "nonsynthesizable"
    attribute and emits clear error messages for unsupported constructs.
  }];
  
  let constructor = "mlir::sharp::createPreSynthesisCheckPass()";
  
  let dependentDialects = ["::sharp::txn::TxnDialect"];
}

def ReachabilityAnalysis : Pass<"sharp-reachability-analysis", "mlir::ModuleOp"> {
  let summary = "Compute reachability conditions for method calls in actions";
  let description = [{
    This pass analyzes Sharp Txn modules to compute reachability conditions for
    method calls within rules and action methods. The analysis tracks control flow
    through txn.if operations to determine under what conditions each method call
    can be reached.

    The pass adds a "reachability_condition" attribute to each txn.call operation,
    which represents the hardware condition under which the call can execute.
    This information is used by the txn-to-FIRRTL conversion to calculate
    conflict_inside for will-fire logic generation.

    Example:
    ```
    txn.rule @example {
      %cond = arith.cmpi eq, %x, %c0 : i32
      txn.if %cond {
        txn.call @inst::@method1() : () -> ()  // reach = %cond
      } else {
        txn.call @inst::@method2() : () -> ()  // reach = !%cond
      }
    }
    ```
  }];
  
  let constructor = "mlir::sharp::createReachabilityAnalysisPass()";
  
  let dependentDialects = ["::sharp::txn::TxnDialect", "::mlir::arith::ArithDialect"];
}

def CombinationalLoopDetection : Pass<"sharp-detect-combinational-loops", "mlir::ModuleOp"> {
  let summary = "Detect combinational loops in txn modules";
  let description = [{
    This pass analyzes Sharp Txn modules to detect combinational loops that would
    create invalid hardware. Combinational loops can occur through:

    1. Direct method call cycles (e.g., getValue -> compute -> getValue)
    2. Cycles through value method dependencies
    3. Cycles through combinational primitives (e.g., Wire)

    The pass builds a dependency graph of combinational paths and uses depth-first
    search to detect cycles. When a cycle is found, the pass emits an error with
    the complete cycle path for debugging.

    Example error: "Combinational loop detected: getValue -> compute -> process -> getValue"
  }];
  
  let constructor = "mlir::sharp::createCombinationalLoopDetectionPass()";
  
  let dependentDialects = ["::sharp::txn::TxnDialect"];
}

def MethodAttributeValidation : Pass<"sharp-validate-method-attributes", "mlir::ModuleOp"> {
  let summary = "Validate method attributes for FIRRTL translation";
  let description = [{
    This pass validates method attributes used in FIRRTL signal generation:

    1. Signal Name Conflicts:
       - Checks for name conflicts between modules and methods (considering prefix)
       - Ensures signal name uniqueness after applying prefixes/postfixes

    2. Protocol Attributes:
       - Validates `always_ready`: method must actually be always ready
       - Validates `always_enable`: method's callers must be always enabled

    3. Naming Attributes:
       - Validates `prefix`, `result`, `ready`, `enable` attributes
       - Ensures generated signal names don't conflict

    The pass emits errors for invalid attribute usage that would cause problems
    during FIRRTL translation.
  }];
  
  let constructor = "mlir::sharp::createMethodAttributeValidationPass()";
  
  let dependentDialects = ["::sharp::txn::TxnDialect"];
}

def ActionScheduling : Pass<"sharp-action-scheduling", "mlir::ModuleOp"> {
  let summary = "Complete partial schedules to minimize conflicts";
  let description = [{
    This pass automatically completes partial schedules to ensure all actions
    (rules and methods) are properly ordered while minimizing conflicts.
    
    The algorithm:
    1. Builds a dependency graph from SA relationships and partial schedule
    2. For small modules (≤10 actions), uses exact algorithm to find optimal schedule
    3. For larger modules, uses a heuristic based on Kahn's algorithm
    
    Optimization goals:
    - Minimize SB violations (where action A should be before B but is scheduled after)
    - Minimize SA violations (where action A should be after B but is scheduled before)
    - Minimize unavoidable conflicts (C relationships)
    - Preserve the original partial schedule order
    
    The pass runs after conflict matrix inference and before FIRRTL translation.
    If the pass fails (e.g., due to cyclic dependencies), FIRRTL translation
    will not proceed.
  }];
  
  let constructor = "mlir::sharp::createActionSchedulingPass()";
  
  let dependentDialects = ["::sharp::txn::TxnDialect"];
}

#endif // SHARP_ANALYSIS_PASSES