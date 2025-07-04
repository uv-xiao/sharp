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
    // Generate includes
    os << "#include \"sharp/Simulation/Core/Simulator.h\"\n";
    os << "#include \"sharp/Simulation/Core/SimModule.h\"\n";
    os << "#include \"sharp/Simulation/Event.h\"\n";
    os << "#include <iostream>\n";
    os << "#include <map>\n";
    os << "#include <string>\n";
    os << "#include <utility>\n\n";
    
    os << "using namespace sharp::sim;\n\n";
    
    // Generate module class
    os << "class " << module.getName() << "Module : public sharp::sim::SimModule {\n";
    os << "private:\n";
    
    // Generate state variables (would need state tracking)
    os << "  // Module state variables\n";
    os << "  // TODO: Extract from txn.state operations\n\n";
    
    // Generate conflict matrix if present
    if (auto schedule = getScheduleOp(module)) {
      if (auto cm = schedule.getConflictMatrix()) {
        generateConflictMatrix(*cm);
      }
    }
    
    os << "public:\n";
    
    // Constructor
    os << "  " << module.getName() << "Module() : SimModule(\"" 
       << module.getName() << "\") {\n";
    
    // Register methods with proper signatures
    for (auto& op : module.getBodyBlock()->getOperations()) {
      if (auto method = mlir::dyn_cast<txn::ValueMethodOp>(op)) {
        os << "    // Value method: " << method.getName();
        if (auto timing = method.getTiming()) {
          os << " (timing: " << timing << ")";
        }
        os << "\n";
        os << "    registerMethod(\"" << method.getName() << "\", "
           << "[this](ArrayRef<Value> args) -> ExecutionResult {\n"
           << "      return execute_" << method.getName() << "(args);\n"
           << "    });\n";
      } else if (auto method = mlir::dyn_cast<txn::ActionMethodOp>(op)) {
        os << "    // Action method: " << method.getName();
        if (auto timing = method.getTiming()) {
          os << " (timing: " << timing << ")";
        }
        os << "\n";
        os << "    registerMethod(\"" << method.getName() << "\", "
           << "[this](ArrayRef<Value> args) -> ExecutionResult {\n"
           << "      return execute_" << method.getName() << "(args);\n"
           << "    });\n";
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
    
    os << "  }\n\n";
    
    // Generate execution methods
    for (auto& op : module.getBodyBlock()->getOperations()) {
      if (auto method = mlir::dyn_cast<txn::ValueMethodOp>(op)) {
        generateValueMethodExecution(method);
      } else if (auto method = mlir::dyn_cast<txn::ActionMethodOp>(op)) {
        generateActionMethodExecution(method);
      } else if (auto rule = mlir::dyn_cast<txn::RuleOp>(op)) {
        generateRuleExecution(rule);
      }
    }
    
    // Generate 1RaaT execution cycle
    generateExecutionCycle(module);
    
    os << "};\n\n";
    
    // Generate main function
    generateMain(module);
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
  
  void generateValueMethodExecution(txn::ValueMethodOp method) {
    os << "  ExecutionResult execute_" << method.getName() 
       << "(ArrayRef<Value> args) {\n";
    os << "    ExecutionResult result;\n";
    
    // Check timing attribute
    if (auto timing = method.getTiming()) {
      if (timing->str().find("static(") == 0) {
        // Extract latency
        auto latencyStr = timing->str().substr(7, timing->str().size() - 8);
        os << "    // Static latency: " << latencyStr << " cycles\n";
        os << "    result.isContinuation = true;\n";
        os << "    result.nextCycle = getCurrentTime() + " << latencyStr << ";\n";
      }
    }
    
    os << "    // TODO: Execute method body\n";
    os << "    return result;\n";
    os << "  }\n\n";
  }
  
  void generateActionMethodExecution(txn::ActionMethodOp method) {
    os << "  ExecutionResult execute_" << method.getName() 
       << "(ArrayRef<Value> args) {\n";
    os << "    ExecutionResult result;\n";
    
    // Check timing attribute
    if (auto timing = method.getTiming()) {
      if (timing->str() == "combinational") {
        os << "    // Combinational execution\n";
      } else if (timing->str().find("static(") == 0) {
        auto latencyStr = timing->str().substr(7, timing->str().size() - 8);
        os << "    // Static latency: " << latencyStr << " cycles\n";
        os << "    result.isContinuation = true;\n";
        os << "    result.nextCycle = getCurrentTime() + " << latencyStr << ";\n";
      }
    }
    
    os << "    // TODO: Execute method body\n";
    os << "    return result;\n";
    os << "  }\n\n";
  }
  
  void generateRuleExecution(txn::RuleOp rule) {
    // Generate guard check
    os << "  bool canFire_" << rule.getSymName() << "() {\n";
    os << "    // TODO: Evaluate rule guard\n";
    os << "    return true;\n";
    os << "  }\n\n";
    
    // Generate rule execution
    os << "  ExecutionResult execute_" << rule.getSymName() << "() {\n";
    os << "    ExecutionResult result;\n";
    os << "    // TODO: Execute rule body\n";
    os << "    return result;\n";
    os << "  }\n\n";
  }
  
  void generateExecutionCycle(txn::ModuleOp module) {
    os << "  // 1RaaT execution cycle\n";
    os << "  void executeCycle() override {\n";
    os << "    // Phase 1: Scheduling - evaluate guards and determine execution order\n";
    os << "    std::vector<std::string> enabledRules;\n";
    
    // Check all rules
    for (auto& op : module.getBodyBlock()->getOperations()) {
      if (auto rule = mlir::dyn_cast<txn::RuleOp>(op)) {
        os << "    if (canFire_" << rule.getSymName() << "()) {\n";
        os << "      enabledRules.push_back(\"" << rule.getSymName() << "\");\n";
        os << "    }\n";
      }
    }
    
    os << "\n    // Phase 2: Execution - run rules atomically in order\n";
    os << "    for (const auto& ruleName : enabledRules) {\n";
    
    // Generate dispatch for each rule
    bool first = true;
    for (auto& op : module.getBodyBlock()->getOperations()) {
      if (auto rule = mlir::dyn_cast<txn::RuleOp>(op)) {
        os << "      ";
        if (!first) os << "else ";
        os << "if (ruleName == \"" << rule.getSymName() << "\") {\n";
        os << "        auto result = execute_" << rule.getSymName() << "();\n";
        os << "        handleExecutionResult(result);\n";
        os << "      }\n";
        first = false;
      }
    }
    
    os << "    }\n";
    os << "\n    // Phase 3: Commit - state updates happen atomically\n";
    os << "    commitStateUpdates();\n";
    os << "  }\n\n";
  }
  
  void generateMain(txn::ModuleOp module) {
    os << "int main() {\n";
    os << "  sharp::sim::SimConfig config;\n";
    os << "  config.maxCycles = 1000;\n";
    os << "  sharp::sim::Simulator sim(config);\n";
    os << "  auto module = std::make_unique<" << module.getName() << "Module>();\n";
    os << "  sim.addModule(\"" << module.getName() << "\", std::move(module));\n";
    os << "  sim.run();\n";
    os << "  return 0;\n";
    os << "}\n";
  }
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
      std::string errorMessage;
      auto output = mlir::openOutputFile(outputFile, &errorMessage);
      if (!output) {
        module.emitError() << "failed to open output file: " << errorMessage;
        signalPassFailure();
        return;
      }
      
      CppCodeGenerator generator(output->os());
      
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
      output->keep();
      
      if (verbose) {
        llvm::errs() << "Generated C++ code to " << outputFile << "\n";
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