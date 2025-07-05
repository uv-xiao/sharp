//===- TxnSimulatePass.cpp - Transaction-Level Simulation Pass ------------===//
//
// This file implements the pass to simulate txn modules at transaction level.
//
//===----------------------------------------------------------------------===//

#include "sharp/Simulation/Passes.h"
#include "sharp/Dialect/Txn/TxnOps.h"
#include "sharp/Dialect/Txn/TxnAttrs.h"
#include "sharp/Simulation/Simulator.h"
#include "sharp/Conversion/Passes.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <memory>

namespace sharp {

#define GEN_PASS_DEF_TXNSIMULATEPASS
#include "sharp/Simulation/Passes.h.inc"

namespace {

using namespace sharp::sim;

/// Helper class to generate C++ code from txn modules
class CppCodeGenerator {
public:
  CppCodeGenerator(llvm::raw_ostream &os) : os(os) {}

  void generateModule(txn::ModuleOp module) {
    // Generate header comment
    os << "// Generated Txn Module Simulation\n";
    
    // Generate includes matching test expectations
    os << "#include <iostream>\n";
    os << "#include <memory>\n";
    os << "#include <vector>\n";
    os << "#include <map>\n";
    os << "#include <string>\n";
    os << "#include <functional>\n";
    os << "#include <cassert>\n";
    os << "#include <chrono>\n";
    os << "#include <queue>\n\n";
    
    // Generate module class
    os << "class " << module.getName() << "Module : public SimModule {\n";
    os << "public:\n";
    
    // Constructor
    os << "  " << module.getName() << "Module() : SimModule(\"" 
       << module.getName() << "\") {\n";
    
    os << "    // Register methods\n";
    for (auto& op : module.getBodyBlock()->getOperations()) {
      if (auto method = mlir::dyn_cast<txn::ValueMethodOp>(op)) {
        os << "    registerValueMethod(\"" << method.getName() << "\", \n";
        os << "      [this](const std::vector<int64_t>& args) -> std::vector<int64_t> {\n";
        os << "        return " << method.getName() << "(";
        // Generate argument list
        auto funcType = method.getFunctionType();
        for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
          if (i > 0) os << ", ";
          os << "args[" << i << "]";
        }
        os << ");\n";
        os << "      });\n";
      } else if (auto method = mlir::dyn_cast<txn::ActionMethodOp>(op)) {
        os << "    registerActionMethod(\"" << method.getName() << "\", \n";
        os << "      [this](const std::vector<int64_t>& args) {\n";
        os << "        " << method.getName() << "(";
        // Generate argument list
        auto funcType = method.getFunctionType();
        for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
          if (i > 0) os << ", ";
          os << "args[" << i << "]";
        }
        os << ");\n";
        os << "      });\n";
      }
    }
    
    // Register rules
    for (auto& op : module.getBodyBlock()->getOperations()) {
      if (auto rule = mlir::dyn_cast<txn::RuleOp>(op)) {
        os << "    // Rule: " << rule.getSymName();
        if (auto timing = rule.getTiming()) {
          os << " (timing: " << timing << ")";
        }
        os << "\n";
        os << "    registerRule(\"" << rule.getSymName() << "\", "
           << "[this]() -> bool { return canFire_" << rule.getSymName() << "(); });\n";
      }
    }
    
    os << "  }\n";
    
    // Generate conflict matrix if present
    if (auto schedule = getScheduleOp(module)) {
      if (auto cm = schedule.getConflictMatrix()) {
        generateSimpleConflictMatrix(*cm);
      }
    }
    
    // Generate value methods
    for (auto& op : module.getBodyBlock()->getOperations()) {
      if (auto method = mlir::dyn_cast<txn::ValueMethodOp>(op)) {
        generateSimpleValueMethod(method);
      } else if (auto method = mlir::dyn_cast<txn::ActionMethodOp>(op)) {
        generateSimpleActionMethod(method);
      }
    }
    
    os << "};\n\n";
  }

private:
  llvm::raw_ostream &os;
  
  // Helper to find schedule operation
  txn::ScheduleOp getScheduleOp(txn::ModuleOp module) {
    for (auto& op : module.getBodyBlock()->getOperations()) {
      if (auto schedule = mlir::dyn_cast<txn::ScheduleOp>(op)) {
        return schedule;
      }
    }
    return nullptr;
  }
  
  std::string getCppType(mlir::Type type) {
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
      if (intType.getWidth() == 1) return "bool";
      if (intType.getWidth() <= 8) return "int8_t";
      if (intType.getWidth() <= 16) return "int16_t";
      if (intType.getWidth() <= 32) return "int32_t";
      if (intType.getWidth() <= 64) return "int64_t";
    }
    return "Value"; // Use generic Value type
  }
  
  void generateConflictMatrix(mlir::DictionaryAttr cmAttr) {
    os << "  // Conflict matrix\n";
    os << "  std::map<std::pair<std::string, std::string>, ConflictRelation> conflicts = {\n";
    
    // Generate conflict entries from flat dictionary with compound keys
    for (auto& entry : cmAttr) {
      // Parse compound key "method1,method2"
      std::string key = entry.getName().str();
      size_t commaPos = key.find(',');
      if (commaPos != std::string::npos) {
        std::string first = key.substr(0, commaPos);
        std::string second = key.substr(commaPos + 1);
        
        // Get conflict relation value
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(entry.getValue())) {
          int64_t conflictValue = intAttr.getInt();
          std::string conflictName;
          switch (conflictValue) {
            case 0: conflictName = "SequenceBefore"; break;
            case 1: conflictName = "SequenceAfter"; break;
            case 2: conflictName = "Conflict"; break;
            case 3: conflictName = "ConflictFree"; break;
            default: conflictName = "ConflictFree"; break;
          }
          os << "    {{\"" << first << "\", \"" << second << "\"}, ConflictRelation::" 
             << conflictName << "},\n";
        }
      }
    }
    os << "  };\n\n";
  }
  
  void generateSimpleConflictMatrix(mlir::DictionaryAttr cmAttr) {
    os << "\n  // Conflict matrix\n";
    os << "  std::map<std::pair<std::string, std::string>, ConflictRelation> conflicts = {\n";
    
    // Generate conflict entries from flat dictionary with compound keys
    for (auto& entry : cmAttr) {
      // Parse compound key "method1,method2"
      std::string key = entry.getName().str();
      size_t commaPos = key.find(',');
      if (commaPos != std::string::npos) {
        std::string first = key.substr(0, commaPos);
        std::string second = key.substr(commaPos + 1);
        
        // Get conflict relation value
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(entry.getValue())) {
          int64_t conflictValue = intAttr.getInt();
          std::string conflictName;
          switch (conflictValue) {
            case 0: conflictName = "SequenceBefore"; break;
            case 1: conflictName = "SequenceAfter"; break;
            case 2: conflictName = "Conflict"; break;
            case 3: conflictName = "ConflictFree"; break;
            default: conflictName = "ConflictFree"; break;
          }
          os << "    {{\"" << first << "\", \"" << second << "\"}, ConflictRelation::" 
             << conflictName << "},\n";
        }
      }
    }
    os << "  };\n";
  }
  
  void generateSimpleActionMethod(txn::ActionMethodOp method) {
    os << "\n  // Action method: " << method.getName() << "\n";
    os << "  void " << method.getName() << "(";
    
    // Generate parameter list
    auto funcType = method.getFunctionType();
    for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
      if (i > 0) os << ", ";
      os << "int64_t arg" << i;
    }
    
    os << ") {\n";
    os << "    // TODO: Implement action logic\n";
    os << "  }\n";
  }
  
  void generateSimpleValueMethod(txn::ValueMethodOp method) {
    os << "\n  // Value method: " << method.getName() << "\n";
    os << "  std::vector<int64_t> " << method.getName() << "(";
    
    // Generate parameter list
    auto funcType = method.getFunctionType();
    for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
      if (i > 0) os << ", ";
      os << "int64_t arg" << i;
    }
    
    os << ") {\n";
    
    // Generate method body - simple translation of operations
    if (!method.getBody().empty()) {
      auto& block = method.getBody().front();
      int varCount = 0;
      llvm::DenseMap<mlir::Value, std::string> valueNames;
      
      // Map block arguments to names
      for (unsigned i = 0; i < block.getNumArguments(); ++i) {
        valueNames[block.getArgument(i)] = "arg" + std::to_string(i);
      }
      
      for (auto& op : block.getOperations()) {
        if (auto constOp = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
          std::string varName = "_" + std::to_string(varCount++);
          valueNames[constOp.getResult()] = varName;
          if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
            os << "    int64_t " << varName << " = " << intAttr.getInt() << ";\n";
          }
        } else if (auto addOp = mlir::dyn_cast<mlir::arith::AddIOp>(op)) {
          std::string varName = "_" + std::to_string(varCount++);
          valueNames[addOp.getResult()] = varName;
          os << "    int64_t " << varName << " = ";
          os << valueNames[addOp.getLhs()] << " + " << valueNames[addOp.getRhs()] << ";\n";
        } else if (auto returnOp = mlir::dyn_cast<txn::ReturnOp>(op)) {
          os << "    return {";
          for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
            if (i > 0) os << ", ";
            os << valueNames[returnOp.getOperand(i)];
          }
          os << "};\n";
        }
      }
    }
    
    os << "  }\n";
  }
  
  // Removed unused generation methods - keeping only simple translation
};

/// JIT execution context for txn modules
class JitExecutionContext {
public:
  JitExecutionContext(mlir::ModuleOp module) : module(module) {}
  
  mlir::LogicalResult initialize() {
    // Create execution engine
    mlir::ExecutionEngineOptions options;
    // Note: transformer is a function_ref, so the lambda must outlive this scope
    // For now, we'll just not set a transformer
    
    auto maybeEngine = mlir::ExecutionEngine::create(module, options);
    if (!maybeEngine) {
      // Consume the error to avoid the assertion
      llvm::consumeError(maybeEngine.takeError());
      return mlir::failure();
    }
    
    engine = std::move(*maybeEngine);
    return mlir::success();
  }
  
  void run(unsigned maxCycles) {
    // Initialize Sharp simulator with config
    SimConfig simConfig;
    simConfig.maxCycles = maxCycles;
    Simulator sim(simConfig);
    
    // Convert txn modules to SimModule instances
    for (auto op : module.getBody()->getOps<txn::ModuleOp>()) {
      auto simModule = createSimModule(op);
      sim.addModule(op.getName().str(), std::move(simModule));
    }
    
    // Run simulation
    sim.run();
  }

private:
  mlir::ModuleOp module;
  std::unique_ptr<mlir::ExecutionEngine> engine;
  
  std::unique_ptr<SimModule> createSimModule(txn::ModuleOp txnModule) {
    // Create a dynamic SimModule that executes MLIR operations
    class MLIRSimModule : public SimModule {
    public:
      MLIRSimModule(txn::ModuleOp module, mlir::ExecutionEngine* engine) 
        : SimModule(module.getName().str()), module(module), engine(engine) {
        // Register methods
        for (auto& op : module.getBodyBlock()->getOperations()) {
          if (auto method = mlir::dyn_cast<txn::ValueMethodOp>(op)) {
            registerMethod(method.getName().str(), 
              [this, method](ArrayRef<Value> args) -> ExecutionResult {
                return executeMethod(method, args);
              });
          } else if (auto method = mlir::dyn_cast<txn::ActionMethodOp>(op)) {
            registerMethod(method.getName().str(), 
              [this, method](ArrayRef<Value> args) -> ExecutionResult {
                return executeMethod(method, args);
              });
          }
        }
      }
      
    private:
      txn::ModuleOp module;
      mlir::ExecutionEngine* engine;
      
      ExecutionResult executeMethod(mlir::Operation* method, ArrayRef<Value> args) {
        (void)engine; // TODO: Use engine for JIT execution
        // TODO: Execute MLIR operations through JIT
        // This would involve:
        // 1. Extracting method body
        // 2. Setting up arguments
        // 3. Invoking JIT-compiled code
        // 4. Returning results
        return ExecutionResult();
      }
    };
    
    return std::make_unique<MLIRSimModule>(txnModule, engine.get());
  }
};

/// The main TxnSimulate pass
struct TxnSimulatePass : public impl::TxnSimulatePassBase<TxnSimulatePass> {
  using Base = impl::TxnSimulatePassBase<TxnSimulatePass>;
  
  // Constructor needed for TableGen
  TxnSimulatePass() = default;
  TxnSimulatePass(const TxnSimulatePassOptions& options) {
    mode = options.mode;
    outputFile = options.outputFile;
    verbose = options.verbose;
    dumpStats = options.dumpStats;
    maxCycles = options.maxCycles;
  }

  void runOnOperation() override {
    auto module = getOperation();
    
    // Parse mode option
    bool isJitMode = (mode == "jit");
    
    if (verbose) {
      llvm::errs() << "Running TxnSimulate pass in " 
                   << (isJitMode ? "JIT" : "translation") << " mode\n";
    }
    
    if (isJitMode) {
      // JIT mode: compile and execute directly
      if (verbose) {
        llvm::errs() << "Running JIT compilation pipeline...\n";
      }
      
      // Create a pass manager for the lowering pipeline
      mlir::PassManager pm(module.getContext());
      pm.enableVerifier(true);
      
      // Add lowering passes
      pm.addPass(mlir::sharp::createConvertTxnToFuncPass());
      pm.addPass(mlir::createCanonicalizerPass());
      
      // Convert func to LLVM dialect
      pm.addPass(mlir::createConvertFuncToLLVMPass());
      pm.addPass(mlir::createReconcileUnrealizedCastsPass());
      
      // Clone the module for lowering (preserve original)
      auto moduleClone = module.clone();
      
      // Run the lowering pipeline
      if (failed(pm.run(moduleClone))) {
        module.emitError() << "Failed to lower txn dialect to LLVM";
        signalPassFailure();
        return;
      }
      
      // Initialize JIT execution context
      JitExecutionContext jitContext(moduleClone);
      if (failed(jitContext.initialize())) {
        module.emitError() << "Failed to initialize JIT execution engine";
        signalPassFailure();
        return;
      }
      
      // Run the simulation
      if (verbose) {
        llvm::errs() << "Starting JIT execution...\n";
      }
      jitContext.run(maxCycles);
      
      if (verbose) {
        llvm::errs() << "JIT execution completed\n";
      }
    } else {
      // Translation mode: generate C++ code
      std::unique_ptr<llvm::ToolOutputFile> output;
      llvm::raw_ostream* os = nullptr;
      
      if (outputFile.empty() || outputFile == "-") {
        // Write to stdout
        os = &llvm::outs();
      } else {
        // Write to file
        std::string errorMessage;
        output = mlir::openOutputFile(outputFile, &errorMessage);
        if (!output) {
          module.emitError() << "failed to open output file: " << errorMessage;
          signalPassFailure();
          return;
        }
        os = &output->os();
      }
      
      CppCodeGenerator generator(*os);
      
      // Find the top module
      txn::ModuleOp topModule;
      for (auto op : module.getBody()->getOps<txn::ModuleOp>()) {
        if (!topModule) {
          topModule = op;
        }
      }
      
      if (!topModule) {
        module.emitError() << "no txn.module found";
        signalPassFailure();
        return;
      }
      
      generator.generateModule(topModule);
      
      if (output) {
        output->keep();
      }
      
      if (verbose) {
        llvm::errs() << "Generated C++ code to " 
                     << (outputFile.empty() ? "stdout" : outputFile.getValue()) << "\n";
      }
    }
  }
};

} // namespace

// Note: The createTxnSimulatePass functions are generated by TableGen
// in the impl namespace and exposed in the outer namespace

// Custom createTxnSimulatePass that takes our TxnSimulateOptions
std::unique_ptr<mlir::Pass> createTxnSimulatePass(const TxnSimulateOptions &options) {
  TxnSimulatePassOptions passOptions;
  passOptions.mode = (options.mode == TxnSimulateOptions::Mode::JIT) ? "jit" : "translation";
  passOptions.outputFile = options.outputFile;
  passOptions.verbose = options.verbose;
  passOptions.dumpStats = options.dumpStats;
  passOptions.maxCycles = options.maxCycles;
  return createTxnSimulatePass(passOptions);
}

} // namespace sharp