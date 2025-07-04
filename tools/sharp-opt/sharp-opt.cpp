//===- sharp-opt.cpp - Sharp optimizer driver -------------------*- C++ -*-===//
//
// Part of the Sharp Project
//
//===----------------------------------------------------------------------===//
//
// This file implements the Sharp optimizer driver with Verilog export support.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "circt/InitAllDialects.h"
#include "circt/Conversion/Passes.h"
#include "circt/Conversion/ExportVerilog.h"
#include "circt/Conversion/FIRRTLToHW.h"
#include "circt/Conversion/SeqToSV.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Dialect/Arc/ArcPasses.h"

#include "sharp/InitAllDialects.h"
#include "sharp/Analysis/Passes.h"
#include "sharp/Conversion/Passes.h"
#include "sharp/Simulation/Passes.h"

namespace {

// Register custom pipelines for Sharp
void registerSharpPipelines() {
  // Pipeline to lower Txn to HW
  mlir::PassPipelineRegistration<>(
    "lower-to-hw", "Lower Txn to HW dialect via FIRRTL",
    [](mlir::OpPassManager &pm) {
      pm.addPass(mlir::sharp::createTxnToFIRRTLConversion());
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(circt::createLowerFIRRTLToHWPass());
      pm.addPass(circt::sv::createHWCleanupPass());
      pm.addPass(circt::seq::createHWMemSimImplPass());
      pm.addPass(circt::createLowerSeqToSVPass());
      pm.addPass(circt::sv::createHWLegalizeModulesPass());
      pm.addPass(circt::sv::createHWCleanupPass());
    });
    
  // Pipeline to export Verilog
  mlir::PassPipelineRegistration<>(
    "txn-export-verilog", "Export to Verilog after lowering from Txn",
    [](mlir::OpPassManager &pm) {
      // First run the lowering pipeline
      pm.addPass(mlir::sharp::createTxnToFIRRTLConversion());
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(circt::createLowerFIRRTLToHWPass());
      pm.addPass(circt::sv::createHWCleanupPass());
      pm.addPass(circt::seq::createHWMemSimImplPass());
      pm.addPass(circt::createLowerSeqToSVPass());
      pm.addPass(circt::sv::createHWLegalizeModulesPass());
      pm.addPass(circt::sv::createHWCleanupPass());
      
      // Export to stdout
      pm.addPass(circt::createExportVerilogPass(llvm::outs()));
    });
}

} // namespace

int main(int argc, char **argv) {
  // Register all passes and dialects
  mlir::registerAllPasses();
  mlir::sharp::registerPasses();
  mlir::sharp::registerConversionPasses();
  sharp::registerSimulationPasses();
  
  // Register only the CIRCT passes we need
  circt::registerLowerFIRRTLToHWPass();
  circt::registerLowerSeqToSVPass();
  circt::registerExportSplitVerilogPass();
  
  // Register arcilator passes for RTL simulation
  circt::arc::registerPasses();
  
  // Register our custom pipelines
  registerSharpPipelines();

  mlir::DialectRegistry registry;
  ::sharp::registerAllDialects(registry);
  circt::registerAllDialects(registry);
  mlir::registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Sharp optimizer driver\n", registry));
}