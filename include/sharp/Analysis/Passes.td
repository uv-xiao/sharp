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

#endif // SHARP_ANALYSIS_PASSES