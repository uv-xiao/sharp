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

def GeneralCheck : Pass<"sharp-general-check", "mlir::ModuleOp"> {
  let summary = "General semantic validation for Sharp Txn modules";
  let description = [{
    This pass performs comprehensive semantic validation of Sharp Txn modules
    according to the core execution model. It combines multiple validation
    checks that apply to all Sharp code regardless of target:

    1. **Schedule Validation**: Ensures schedules only contain actions (rules
       and action methods), not value methods.

    2. **Value Method Conflict Check**: Validates that value methods are
       conflict-free with all scheduled actions, as required by the execution model.

    3. **Action Call Validation**: Ensures actions cannot call other actions
       within the same module, maintaining the one-rule-at-a-time semantics.

    This unified pass replaces the individual passes:
    - sharp-validate-schedule  
    - sharp-check-value-method-conflicts
    - sharp-validate-action-calls

    The pass should be run on any Sharp module to verify it follows the
    core execution model before further analysis or conversion.
  }];
  
  let constructor = "mlir::sharp::createGeneralCheckPass()";
  
  let dependentDialects = ["::sharp::txn::TxnDialect"];
}

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
  let summary = "Comprehensive synthesis validation for Sharp Txn modules";
  let description = [{
    This pass analyzes Sharp Txn modules to identify any constructs that would
    prevent successful translation to FIRRTL/Verilog hardware. The pass performs
    comprehensive validation including:

    1. **Primitive Synthesizability**: Checks for non-synthesizable (spec) primitives
       that are simulation-only and cannot be converted to hardware.

    2. **Operation Allowlist**: Validates that only synthesizable operations from
       approved dialects (txn, firrtl, builtin, test) and specific arith operations
       are used in synthesizable modules.

    3. **Multi-cycle Operations**: Checks for timing constraints that require
       multi-cycle implementation (currently disabled as all operations are single-cycle).

    4. **Method Attribute Validation**: Validates method attributes used in FIRRTL
       signal generation, including signal name uniqueness, prefix/postfix correctness,
       and synthesis-compatible attribute combinations.

    5. **Hierarchical Propagation**: Marks parent modules as non-synthesizable if
       they instantiate non-synthesizable child modules or primitives.

    This unified pass replaces the individual pass:
    - sharp-validate-method-attributes (now integrated)

    The pass marks non-synthesizable modules with a "nonsynthesizable" attribute
    and emits clear error messages for all violations. This validation must pass
    before modules can be converted to FIRRTL.
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


// MethodAttributeValidation is now integrated into PreSynthesisCheck
// def MethodAttributeValidation : Pass<"sharp-validate-method-attributes", "mlir::ModuleOp"> {
//   let summary = "Validate method attributes for FIRRTL translation";
//   ... (functionality moved to PreSynthesisCheck)
// }

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

// ScheduleValidation is now integrated into GeneralCheck
// def ScheduleValidation : Pass<"sharp-validate-schedule", "mlir::ModuleOp"> {
//   let summary = "Validate that schedules only contain actions (rules and action methods)";
//   ... (functionality moved to GeneralCheck)
// }

// ValueMethodConflictCheck is now integrated into GeneralCheck
// def ValueMethodConflictCheck : Pass<"sharp-check-value-method-conflicts", "mlir::ModuleOp"> {
//   let summary = "Check that value methods are conflict-free with all actions";
//   ... (functionality moved to GeneralCheck)
// }

// ActionCallValidation is now integrated into GeneralCheck
// def ActionCallValidation : Pass<"sharp-validate-action-calls", "mlir::ModuleOp"> {
//   let summary = "Validate that actions do not call other actions in the same module";
//   ... (functionality moved to GeneralCheck)
// }

def SharpAnalysisCollectPrimitiveActionsPass : Pass<"sharp-collect-primitive-actions", "mlir::ModuleOp"> {
  let summary = "Collect primitive action calls for each action";
  let description = [{
    This pass analyzes Sharp Txn modules to collect all primitive action calls
    made by each action (rule or action method). The pass adds a "primitive_calls"
    attribute to each action containing the list of primitive instance paths that
    the action calls methods on.
    
    This information is used by the dynamic mode of the TxnToFIRRTL conversion
    pass to track conflicts at the primitive action level rather than just at the
    module method level.
    
    The pass traces through method calls recursively to find all primitive calls:
    - Direct primitive calls (e.g., @reg::@write)
    - Indirect calls through module methods that eventually call primitives
    
    Example output:
    - Rule @incrementFirst gets attribute: primitive_calls = ["counter1::reg::write", "counter1::reg::read"]
    - Rule @resetFirst gets attribute: primitive_calls = ["counter1::reg::write"]
    - Rule @incrementSecond gets attribute: primitive_calls = ["counter2::reg::write", "counter2::reg::read"]
    
    This enables dynamic mode to detect that incrementFirst and resetFirst
    conflict because they both access counter1::reg, while incrementSecond does
    not conflict with them because it accesses counter2::reg.
  }];
  
  let constructor = "mlir::sharp::createCollectPrimitiveActionsPass()";
  
  let dependentDialects = ["::sharp::txn::TxnDialect"];
}

def InlineFunctions : Pass<"sharp-inline-functions", "mlir::ModuleOp"> {
  let summary = "Inline txn.func calls within txn modules";
  let description = [{
    This pass inlines all txn.func_call operations by replacing them with the
    body of the called function. Functions (txn.func) are syntax sugar for
    combinational logic that can take arguments, and they must be inlined
    before lowering to FIRRTL or Func dialects.
    
    The inlining process:
    1. Finds all txn.func_call operations
    2. Looks up the called function within the same module
    3. Creates a mapping from function arguments to call operands
    4. Clones the function body, replacing arguments with operands
    5. Replaces txn.return with the returned values
    6. Replaces the call with the inlined body
    7. Removes unused txn.func operations after inlining
    
    Example:
    ```mlir
    txn.func @add(%a: i32, %b: i32) -> i32 {
      %sum = arith.addi %a, %b : i32
      txn.return %sum : i32
    }
    
    txn.value_method @compute() -> i32 {
      %c5 = arith.constant 5 : i32
      %c3 = arith.constant 3 : i32
      %result = txn.func_call @add(%c5, %c3) : (i32, i32) -> i32
      txn.return %result : i32
    }
    ```
    
    After inlining:
    ```mlir
    txn.value_method @compute() -> i32 {
      %c5 = arith.constant 5 : i32
      %c3 = arith.constant 3 : i32
      %sum = arith.addi %c5, %c3 : i32
      txn.return %sum : i32
    }
    ```
    
    This pass must run before TxnToFIRRTL and TxnToFunc conversions since
    those passes do not handle txn.func operations.
  }];
  
  let constructor = "mlir::sharp::createInlineFunctionsPass()";
  
  let dependentDialects = ["::sharp::txn::TxnDialect"];
}

def PrimitiveGen : Pass<"sharp-primitive-gen", "mlir::ModuleOp"> {
  let summary = "Generate missing primitive definitions for referenced primitives";
  let description = [{
    This pass analyzes Sharp Txn modules to find all primitive instances and
    automatically generates the corresponding primitive definitions if they
    don't already exist. The generated primitives are created using the
    constructor functions in lib/Dialect/Txn/primitives/.
    
    Supported primitives:
    - Register<T>: Single-port register with read/write methods
    - Wire<T>: Combinational wire with read/write methods  
    - FIFO<T>: FIFO queue with enq/deq/first/notEmpty/notFull methods
    - Memory<T>: Memory with read/write methods
    - SpecFIFO<T>: Speculative FIFO for simulation
    - SpecMemory<T>: Speculative memory for simulation
    
    The pass:
    1. Scans all txn.instance operations to find primitive references
    2. Checks if the referenced primitive module already exists
    3. If not, generates it using the appropriate constructor function
    4. Adds the generated primitive to the module
    
    This ensures that all referenced primitives have valid definitions
    before conflict matrix inference runs, enabling proper conflict
    analysis based on primitive method conflict matrices.
    
    Example:
    ```mlir
    txn.module @MyModule {
      %reg = txn.instance @reg of @Register<i32> : !txn.module<"Register">
      // Register<i32> primitive will be auto-generated if missing
    }
    ```
  }];
  
  let constructor = "mlir::sharp::createPrimitiveGenPass()";
  
  let dependentDialects = ["::sharp::txn::TxnDialect"];
}

#endif // SHARP_ANALYSIS_PASSES