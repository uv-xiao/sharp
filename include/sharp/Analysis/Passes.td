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
    2. Multi-cycle operations (currently none since timing attributes removed)
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

def ScheduleValidation : Pass<"sharp-validate-schedule", "mlir::ModuleOp"> {
  let summary = "Validate that schedules only contain actions (rules and action methods)";
  let description = [{
    This pass validates that schedule operations only contain actions, not value methods.
    According to Sharp's execution model:
    
    1. Schedules must only list actions (rules and action methods)
    2. Value methods are NOT schedulable and must not appear in schedules
    3. Value methods must be conflict-free with all actions
    
    The pass checks each schedule operation and emits errors if:
    - A value method appears in the schedule
    - A scheduled item is not found in the module
    - A scheduled item is neither a rule nor an action method
    
    This validation ensures the execution model is correctly followed and prevents
    invalid hardware generation.
  }];
  
  let constructor = "mlir::sharp::createScheduleValidationPass()";
  
  let dependentDialects = ["::sharp::txn::TxnDialect"];
}

def ValueMethodConflictCheck : Pass<"sharp-check-value-method-conflicts", "mlir::ModuleOp"> {
  let summary = "Check that value methods are conflict-free with all actions";
  let description = [{
    This pass validates that value methods follow Sharp's execution model requirement
    of being conflict-free with all actions. According to the execution model:
    
    1. Value methods must be conflict-free (CF) with all actions
    2. Value methods can only have CF relationships in conflict matrices
    3. If a method has any non-CF conflict, it cannot be a value method
    
    The pass examines each value method and checks:
    - Its conflict relationships in the module's conflict matrix
    - Conflicts inherited from primitive methods it calls
    - Ensures no SA, SB, or C relationships exist
    
    This validation is critical because value methods are computed once per cycle
    in the Value Phase and must not have ordering dependencies with actions.
    
    Example violation: Wire's read method cannot be a value method because it has
    "read SA write" conflict, requiring it to be scheduled before write operations.
  }];
  
  let constructor = "mlir::sharp::createValueMethodConflictCheckPass()";
  
  let dependentDialects = ["::sharp::txn::TxnDialect"];
}

def ActionCallValidation : Pass<"sharp-validate-action-calls", "mlir::ModuleOp"> {
  let summary = "Validate that actions do not call other actions in the same module";
  let description = [{
    This pass validates Sharp's execution model requirement that actions (rules and
    action methods) cannot call other actions within the same module. According to
    the execution model:
    
    1. Actions cannot call other actions in the same module
    2. Actions can call value methods in the same module
    3. Actions can call methods (both value and action) of child module instances
    
    The pass examines each action (rule or action method) and checks all method
    calls within it to ensure they follow these restrictions.
    
    This restriction ensures that the schedule fully determines the execution order
    of actions within a module, preventing hidden dependencies and maintaining the
    one-rule-at-a-time semantics.
    
    Example violations:
    - A rule calling an action method in the same module
    - An action method calling another action method in the same module
    - An action method calling a rule (rules are not callable)
  }];
  
  let constructor = "mlir::sharp::createActionCallValidationPass()";
  
  let dependentDialects = ["::sharp::txn::TxnDialect"];
}

def SharpAnalysisCollectPrimitiveActionsPass : Pass<"sharp-collect-primitive-actions", "mlir::ModuleOp"> {
  let summary = "Collect primitive action calls for each action";
  let description = [{
    This pass analyzes Sharp Txn modules to collect all primitive action calls
    made by each action (rule or action method). The pass adds a "primitive_calls"
    attribute to each action containing the list of primitive instance paths that
    the action calls methods on.
    
    This information is used by the most-dynamic mode of the TxnToFIRRTL conversion
    pass to track conflicts at the primitive action level rather than just at the
    module method level.
    
    The pass traces through method calls recursively to find all primitive calls:
    - Direct primitive calls (e.g., @reg::@write)
    - Indirect calls through module methods that eventually call primitives
    
    Example output:
    - Rule @incrementFirst gets attribute: primitive_calls = ["counter1::reg::write", "counter1::reg::read"]
    - Rule @resetFirst gets attribute: primitive_calls = ["counter1::reg::write"]
    - Rule @incrementSecond gets attribute: primitive_calls = ["counter2::reg::write", "counter2::reg::read"]
    
    This enables most-dynamic mode to detect that incrementFirst and resetFirst
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

#endif // SHARP_ANALYSIS_PASSES