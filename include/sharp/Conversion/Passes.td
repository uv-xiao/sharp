//===- Passes.td - Conversion pass definitions ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for Sharp conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_CONVERSION_PASSES_TD
#define SHARP_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// TxnToFIRRTL
//===----------------------------------------------------------------------===//

def TxnToFIRRTLConversion : Pass<"convert-txn-to-firrtl", "mlir::ModuleOp"> {
  let summary = "Convert Sharp Txn dialect to FIRRTL dialect";
  let description = [{
    This pass converts Sharp Txn modules to FIRRTL modules following the
    Koika-inspired translation approach. It generates:
    - FIRRTL module structure with appropriate ports
    - Will-fire logic for rules and methods
    - Conflict resolution based on conflict matrices
    - Reachability analysis for conflict_inside calculation
    
    The pass processes modules bottom-up to handle dependencies correctly.
  }];
  
  let dependentDialects = [
    "::sharp::txn::TxnDialect",
    "::circt::firrtl::FIRRTLDialect",
    "::mlir::arith::ArithDialect"
  ];
  
  let statistics = [
    Statistic<"numModulesConverted", "modules-converted",
              "Number of Txn modules converted to FIRRTL">,
    Statistic<"numRulesConverted", "rules-converted",
              "Number of rules converted">,
    Statistic<"numMethodsConverted", "methods-converted",
              "Number of methods converted">,
    Statistic<"numWillFireGenerated", "will-fire-generated",
              "Number of will-fire signals generated">
  ];
  
  let options = [
    Option<"enableOptimization", "enable-optimization", "bool",
           /*default=*/"true",
           "Enable optimization of generated will-fire logic">,
    Option<"debugWillFire", "debug-will-fire", "bool",
           /*default=*/"false",
           "Generate debug attributes for will-fire signals">
  ];
}

#endif // SHARP_CONVERSION_PASSES_TD