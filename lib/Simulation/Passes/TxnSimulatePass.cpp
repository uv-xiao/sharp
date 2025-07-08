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
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
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
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Path.h"
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
    os << "// Module: " << module.getName() << "\n\n";
    
    // Generate includes
    os << "#include \"SimulationBase.h\"\n";
    os << "#include <iostream>\n";
    os << "#include <memory>\n";
    os << "#include <vector>\n";
    os << "#include <map>\n";
    os << "#include <string>\n";
    os << "#include <functional>\n";
    os << "#include <cassert>\n";
    os << "#include <chrono>\n";
    os << "#include <queue>\n\n";
    
    // Check if module has multi-cycle actions
    bool hasMultiCycle = false;
    for (auto& op : module.getBodyBlock()->getOperations()) {
      if (auto method = mlir::dyn_cast<txn::ActionMethodOp>(op)) {
        // Check if method contains a future block
        for (auto& innerOp : method.getBody().front()) {
          if (mlir::isa<txn::FutureOp>(innerOp)) {
            hasMultiCycle = true;
            break;
          }
        }
      } else if (auto rule = mlir::dyn_cast<txn::RuleOp>(op)) {
        // Check if rule contains a future block
        for (auto& innerOp : rule.getBody().front()) {
          if (mlir::isa<txn::FutureOp>(innerOp)) {
            hasMultiCycle = true;
            break;
          }
        }
      }
      if (hasMultiCycle) break;
    }
    
    // Generate module class
    std::string baseClass = hasMultiCycle ? "MultiCycleSimModule" : "SimModule";
    os << "class " << module.getName() << "Module : public " << baseClass << " {\n";
    os << "public:\n";
    
    // Constructor
    os << "  " << module.getName() << "Module() : " << baseClass << "(\"" 
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
        // Check if this is a multi-cycle action
        bool isMultiCycle = false;
        for (auto& innerOp : method.getBody().front()) {
          if (mlir::isa<txn::FutureOp>(innerOp)) {
            isMultiCycle = true;
            break;
          }
        }
        
        if (isMultiCycle) {
          os << "    registerMultiCycleAction(\"" << method.getName() << "\", \n";
          os << "      [this](const std::vector<int64_t>& args) {\n";
          os << "        return " << method.getName() << "_multicycle(";
          // Generate argument list
          auto funcType = method.getFunctionType();
          for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
            if (i > 0) os << ", ";
            os << "args[" << i << "]";
          }
          os << ");\n";
          os << "      });\n";
        } else {
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
    }
    
    // Register rules
    for (auto& op : module.getBodyBlock()->getOperations()) {
      if (auto rule = mlir::dyn_cast<txn::RuleOp>(op)) {
        // Check if this is a multi-cycle rule
        bool isMultiCycle = false;
        for (auto& innerOp : rule.getBody().front()) {
          if (mlir::isa<txn::FutureOp>(innerOp)) {
            isMultiCycle = true;
            break;
          }
        }
        
        os << "    // Rule: " << rule.getSymName() << "\n";
        os << "    registerRule(\"" << rule.getSymName() << "\", "
           << "[this]() -> bool { return canFire_" << rule.getSymName() << "(); });\n";
        
        if (isMultiCycle) {
          os << "    registerMultiCycleAction(\"" << rule.getSymName() << "\", \n";
          os << "      [this](const std::vector<int64_t>& args) {\n";
          os << "        return " << rule.getSymName() << "_multicycle();\n";
          os << "      });\n";
        }
      }
    }
    
    // Set schedule and conflict matrix
    if (auto schedule = getScheduleOp(module)) {
      os << "\n    // Set schedule\n";
      os << "    setSchedule({";
      auto actions = schedule.getActions();
      for (unsigned i = 0; i < actions.size(); ++i) {
        if (i > 0) os << ", ";
        os << "\"" << mlir::cast<mlir::FlatSymbolRefAttr>(actions[i]).getValue() << "\"";
      }
      os << "});\n";
      
      if (auto cm = schedule.getConflictMatrix()) {
        os << "\n    // Set conflict matrix\n";
        os << "    setConflictMatrix({\n";
        generateInlineConflictMatrix(*cm, os);
        os << "    });\n";
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
        // Check if this is a multi-cycle action
        bool isMultiCycle = false;
        for (auto& innerOp : method.getBody().front()) {
          if (mlir::isa<txn::FutureOp>(innerOp)) {
            isMultiCycle = true;
            break;
          }
        }
        
        if (isMultiCycle) {
          generateMultiCycleActionMethod(method);
        } else {
          generateSimpleActionMethod(method);
        }
      }
    }
    
    // Generate rules
    for (auto& op : module.getBodyBlock()->getOperations()) {
      if (auto rule = mlir::dyn_cast<txn::RuleOp>(op)) {
        // Check if this is a multi-cycle rule
        bool isMultiCycle = false;
        for (auto& innerOp : rule.getBody().front()) {
          if (mlir::isa<txn::FutureOp>(innerOp)) {
            isMultiCycle = true;
            break;
          }
        }
        
        if (isMultiCycle) {
          generateMultiCycleRule(rule);
        } else {
          generateRule(rule);
        }
      }
    }
    
    // Add primitive state
    generatePrimitiveState(module);
    
    os << "};\n\n";
    
    // Generate main function
    generateMainFunction(module);
  }
  
  void generateMainFunction(txn::ModuleOp module) {
    os << "// Main function\n";
    os << "int main(int argc, char* argv[]) {\n";
    os << "  // Parse command line arguments\n";
    os << "  int maxCycles = 100;\n";
    os << "  bool verbose = false;\n";
    os << "  bool dumpStats = false;\n\n";
    
    os << "  for (int i = 1; i < argc; ++i) {\n";
    os << "    std::string arg = argv[i];\n";
    os << "    if (arg == \"--cycles\" && i + 1 < argc) {\n";
    os << "      maxCycles = std::stoi(argv[++i]);\n";
    os << "    } else if (arg == \"--verbose\") {\n";
    os << "      verbose = true;\n";
    os << "    } else if (arg == \"--stats\") {\n";
    os << "      dumpStats = true;\n";
    os << "    } else if (arg == \"--help\") {\n";
    os << "      std::cout << \"Usage: \" << argv[0] << \" [options]\\n\";\n";
    os << "      std::cout << \"Options:\\n\";\n";
    os << "      std::cout << \"  --cycles <n>  Run for n cycles (default: 100)\\n\";\n";
    os << "      std::cout << \"  --verbose     Enable verbose output\\n\";\n";
    os << "      std::cout << \"  --stats       Dump performance statistics\\n\";\n";
    os << "      std::cout << \"  --help        Show this help message\\n\";\n";
    os << "      return 0;\n";
    os << "    }\n";
    os << "  }\n\n";
    
    os << "  // Create simulator\n";
    os << "  Simulator sim;\n";
    os << "  sim.setVerbose(verbose);\n";
    os << "  sim.setDumpStats(dumpStats);\n\n";
    
    os << "  // Create and add module\n";
    os << "  auto module = std::make_unique<" << module.getName() << "Module>();\n";
    os << "  sim.addModule(std::move(module));\n\n";
    
    os << "  // Run simulation\n";
    os << "  sim.run(maxCycles);\n\n";
    
    os << "  return 0;\n";
    os << "}\n";
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
  
  void generateInlineConflictMatrix(mlir::DictionaryAttr cmAttr, llvm::raw_ostream& os) {
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
          os << "      {{\"" << first << "\", \"" << second << "\"}, ConflictRelation::" 
             << conflictName << "},\n";
        }
      }
    }
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
  
  void generateMultiCycleActionMethod(txn::ActionMethodOp method) {
    os << "\n  // Multi-cycle action method: " << method.getName() << "\n";
    os << "  std::unique_ptr<MultiCycleExecution> " << method.getName() << "_multicycle(";
    
    // Generate parameter list
    auto funcType = method.getFunctionType();
    for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
      if (i > 0) os << ", ";
      os << "int64_t arg" << i;
    }
    
    os << ") {\n";
    os << "    auto exec = std::make_unique<MultiCycleExecution>();\n";
    os << "    exec->actionName = \"" << method.getName() << "\";\n";
    
    // Generate per-cycle actions first
    if (!method.getBody().empty()) {
      auto& block = method.getBody().front();
      int varCount = 0;
      llvm::DenseMap<mlir::Value, std::string> valueNames;
      
      // Map block arguments to names
      for (unsigned i = 0; i < block.getNumArguments(); ++i) {
        valueNames[block.getArgument(i)] = "arg" + std::to_string(i);
      }
      
      os << "\n    // Per-cycle actions execute immediately\n";
      
      // First pass: generate per-cycle actions (before future block)
      for (auto& op : block.getOperations()) {
        if (mlir::isa<txn::FutureOp>(op)) {
          break; // Stop at future block
        }
        
        // Generate the per-cycle action code similar to simple action method
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
        } else if (auto callOp = mlir::dyn_cast<txn::CallOp>(op)) {
          // Handle calls to primitive methods
          auto callee = callOp.getCallee();
          auto leafRef = callee.getLeafReference().getValue();
          if (leafRef == "read") {
            // Reading from a primitive
            std::string instanceName = callee.getRootReference().getValue().str();
            std::string varName = "_" + std::to_string(varCount++);
            valueNames[callOp.getResult(0)] = varName;
            os << "    int64_t " << varName << " = " << instanceName << "_data;\n";
          } else if (leafRef == "write") {
            // Writing to a primitive
            std::string instanceName = callee.getRootReference().getValue().str();
            os << "    " << instanceName << "_data = " << valueNames[callOp.getOperand(0)] << ";\n";
          }
        }
      }
      
      os << "\n    // Future block launches\n";
      // Second pass: process future block
      for (auto& op : block.getOperations()) {
        if (auto futureOp = mlir::dyn_cast<txn::FutureOp>(op)) {
          // Process launches within the future block
          for (auto& innerOp : futureOp.getBody().front()) {
            if (auto launchOp = mlir::dyn_cast<txn::LaunchOp>(innerOp)) {
              os << "\n    // Launch\n";
              os << "    {\n";
              os << "      auto launch = std::make_unique<LaunchState>();\n";
              
              if (launchOp.getLatency()) {
                os << "      launch->hasStaticLatency = true;\n";
                os << "      launch->latency = " << *launchOp.getLatency() << ";\n";
              }
              
              if (launchOp.getCondition()) {
                // Track dependency based on launch results
                // For now, we'll use a simple naming scheme
                int depIndex = 0;
                for (auto& innerOp2 : futureOp.getBody().front()) {
                  if (auto prevLaunch = mlir::dyn_cast<txn::LaunchOp>(innerOp2)) {
                    if (prevLaunch.getResult() == launchOp.getCondition()) {
                      os << "      launch->conditionName = \"" << method.getName() << "_launch_" << depIndex << "\";\n";
                      break;
                    }
                    depIndex++;
                  }
                }
              }
              
              os << "      launch->body = [=]() {\n";
              // Generate launch body
              generateLaunchBody(launchOp.getBody(), "        ");
              os << "      };\n";
              
              os << "      exec->launches.push_back(std::move(launch));\n";
              os << "    }\n";
            }
          }
        }
      }
    }
    
    os << "    return exec;\n";
    os << "  }\n";
  }
  
  void generateLaunchBody(mlir::Region& region, const std::string& indent) {
    if (!region.empty()) {
      auto& block = region.front();
      int varCount = 0;
      llvm::DenseMap<mlir::Value, std::string> valueNames;
      
      for (auto& op : block.getOperations()) {
        if (auto constOp = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
          std::string varName = "_launch_" + std::to_string(varCount++);
          valueNames[constOp.getResult()] = varName;
          if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
            os << indent << "int64_t " << varName << " = " << intAttr.getInt() << ";\n";
          }
        } else if (auto addOp = mlir::dyn_cast<mlir::arith::AddIOp>(op)) {
          std::string varName = "_launch_" + std::to_string(varCount++);
          valueNames[addOp.getResult()] = varName;
          os << indent << "int64_t " << varName << " = ";
          os << valueNames[addOp.getLhs()] << " + " << valueNames[addOp.getRhs()] << ";\n";
        } else if (auto callOp = mlir::dyn_cast<txn::CallOp>(op)) {
          auto callee = callOp.getCallee();
          auto leafRef = callee.getLeafReference().getValue();
          
          if (leafRef == "read") {
            // Reading from a primitive
            std::string instanceName = callee.getRootReference().getValue().str();
            std::string varName = "_launch_" + std::to_string(varCount++);
            valueNames[callOp.getResult(0)] = varName;
            os << indent << "int64_t " << varName << " = " << instanceName << "_data;\n";
          } else if (leafRef == "write") {
            // Writing to a primitive
            std::string instanceName = callee.getRootReference().getValue().str();
            if (callOp.getNumOperands() > 0) {
              os << indent << instanceName << "_data = " << valueNames[callOp.getOperand(0)] << ";\n";
            }
          } else if (leafRef == "enqueue") {
            // Enqueue to FIFO
            std::string instanceName = callee.getRootReference().getValue().str();
            if (callOp.getNumOperands() > 0) {
              os << indent << "if (" << instanceName << "_queue.size() >= " << instanceName << "_depth) {\n";
              os << indent << "  throw std::runtime_error(\"FIFO full\");\n";
              os << indent << "}\n";
              os << indent << instanceName << "_queue.push(" << valueNames[callOp.getOperand(0)] << ");\n";
            }
          }
        } else if (mlir::isa<txn::YieldOp>(op)) {
          // Yield marks the end of the launch body
          break;
        }
      }
    }
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
    
    // Generate method body
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
        } else if (auto xorOp = mlir::dyn_cast<mlir::arith::XOrIOp>(op)) {
          std::string varName = "_" + std::to_string(varCount++);
          valueNames[xorOp.getResult()] = varName;
          os << "    int64_t " << varName << " = ";
          os << valueNames[xorOp.getLhs()] << " ^ " << valueNames[xorOp.getRhs()] << ";\n";
        } else if (auto subOp = mlir::dyn_cast<mlir::arith::SubIOp>(op)) {
          std::string varName = "_" + std::to_string(varCount++);
          valueNames[subOp.getResult()] = varName;
          os << "    int64_t " << varName << " = ";
          os << valueNames[subOp.getLhs()] << " - " << valueNames[subOp.getRhs()] << ";\n";
        } else if (auto callOp = mlir::dyn_cast<txn::CallOp>(op)) {
          // Handle calls to primitive methods
          auto callee = callOp.getCallee();
          auto leafRef = callee.getLeafReference().getValue();
          if (leafRef.contains(".read") || leafRef == "read") {
            // Reading from a primitive
            std::string instanceName = callee.getRootReference().getValue().str();
            std::string varName = "_" + std::to_string(varCount++);
            valueNames[callOp.getResult(0)] = varName;
            os << "    int64_t " << varName << " = " << instanceName << "_data;\n";
          } else if (leafRef.contains(".write") || leafRef == "write") {
            // Writing to a primitive
            std::string instanceName = callee.getRootReference().getValue().str();
            if (callOp.getNumOperands() > 0) {
              os << "    " << instanceName << "_data = " << valueNames[callOp.getOperand(0)] << ";\n";
            }
          } else if (leafRef.contains(".enqueue") || leafRef == "enqueue") {
            // Enqueue to FIFO
            std::string instanceName = callee.getRootReference().getValue().str();
            if (callOp.getNumOperands() > 0) {
              os << "    if (" << instanceName << "_queue.size() < " << instanceName << "_depth) {\n";
              os << "      " << instanceName << "_queue.push(" << valueNames[callOp.getOperand(0)] << ");\n";
              os << "    }\n";
            }
          } else if (leafRef.contains(".dequeue") || leafRef == "dequeue") {
            // Dequeue from FIFO
            std::string instanceName = callee.getRootReference().getValue().str();
            std::string varName = "_" + std::to_string(varCount++);
            valueNames[callOp.getResult(0)] = varName;
            os << "    int64_t " << varName << " = 0;\n";
            os << "    if (!" << instanceName << "_queue.empty()) {\n";
            os << "      " << varName << " = " << instanceName << "_queue.front();\n";
            os << "      " << instanceName << "_queue.pop();\n";
            os << "    }\n";
          } else if (leafRef.contains(".isEmpty") || leafRef == "isEmpty") {
            // Check if FIFO is empty
            std::string instanceName = callee.getRootReference().getValue().str();
            std::string varName = "_" + std::to_string(varCount++);
            valueNames[callOp.getResult(0)] = varName;
            os << "    int64_t " << varName << " = " << instanceName << "_queue.empty() ? 1 : 0;\n";
          } else if (leafRef.contains(".isFull") || leafRef == "isFull") {
            // Check if FIFO is full
            std::string instanceName = callee.getRootReference().getValue().str();
            std::string varName = "_" + std::to_string(varCount++);
            valueNames[callOp.getResult(0)] = varName;
            os << "    int64_t " << varName << " = (" << instanceName << "_queue.size() >= " 
               << instanceName << "_depth) ? 1 : 0;\n";
          }
        } else if (auto futureOp = mlir::dyn_cast<txn::FutureOp>(op)) {
          // Future operations encapsulate multi-cycle execution
          os << "    // Future block - multi-cycle execution not yet supported in simulation\n";
          os << "    // TODO: Implement multi-cycle simulation support\n";
        } else if (auto launchOp = mlir::dyn_cast<txn::LaunchOp>(op)) {
          // Launch operations represent deferred execution
          std::string varName = "_done_" + std::to_string(varCount++);
          valueNames[launchOp.getDone()] = varName;
          os << "    // Launch operation - deferred execution not yet supported\n";
          os << "    int64_t " << varName << " = 1; // Assume immediate completion\n";
          if (launchOp.getLatency()) {
            os << "    // Would wait " << *launchOp.getLatency() << " cycles\n";
          }
          if (launchOp.getCondition()) {
            os << "    // Would wait for condition " << valueNames[launchOp.getCondition()] << "\n";
          }
        }
      }
    } else {
      os << "    // TODO: Implement action logic\n";
    }
    
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
      
      bool hasReturn = false;
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
        } else if (auto xorOp = mlir::dyn_cast<mlir::arith::XOrIOp>(op)) {
          std::string varName = "_" + std::to_string(varCount++);
          valueNames[xorOp.getResult()] = varName;
          os << "    int64_t " << varName << " = ";
          os << valueNames[xorOp.getLhs()] << " ^ " << valueNames[xorOp.getRhs()] << ";\n";
        } else if (auto subOp = mlir::dyn_cast<mlir::arith::SubIOp>(op)) {
          std::string varName = "_" + std::to_string(varCount++);
          valueNames[subOp.getResult()] = varName;
          os << "    int64_t " << varName << " = ";
          os << valueNames[subOp.getLhs()] << " - " << valueNames[subOp.getRhs()] << ";\n";
        } else if (auto callOp = mlir::dyn_cast<txn::CallOp>(op)) {
          // Handle calls to primitive methods
          auto callee = callOp.getCallee();
          auto leafRef = callee.getLeafReference().getValue();
          if (leafRef.contains(".read") || leafRef == "read") {
            // Reading from a primitive
            std::string instanceName = callee.getRootReference().getValue().str();
            std::string varName = "_" + std::to_string(varCount++);
            valueNames[callOp.getResult(0)] = varName;
            os << "    int64_t " << varName << " = " << instanceName << "_data;\n";
          }
        } else if (auto returnOp = mlir::dyn_cast<txn::ReturnOp>(op)) {
          hasReturn = true;
          os << "    return {";
          for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
            if (i > 0) os << ", ";
            os << valueNames[returnOp.getOperand(i)];
          }
          os << "};\n";
        }
      }
      
      if (!hasReturn) {
        os << "    return {};\n";
      }
    } else {
      os << "    return {};\n";
    }
    
    os << "  }\n";
  }
  
  void generateRule(txn::RuleOp rule) {
    os << "\n  // Rule: " << rule.getSymName() << "\n";
    os << "  bool canFire_" << rule.getSymName() << "() {\n";
    os << "    // TODO: Implement rule guard logic\n";
    os << "    return true;\n";
    os << "  }\n";
  }
  
  void generateMultiCycleRule(txn::RuleOp rule) {
    os << "\n  // Multi-cycle Rule: " << rule.getSymName() << "\n";
    os << "  bool canFire_" << rule.getSymName() << "() {\n";
    os << "    // TODO: Implement rule guard logic\n";
    os << "    return true;\n";
    os << "  }\n";
    
    os << "\n  std::unique_ptr<MultiCycleExecution> " << rule.getSymName() << "_multicycle() {\n";
    os << "    auto exec = std::make_unique<MultiCycleExecution>();\n";
    os << "    exec->actionName = \"" << rule.getSymName() << "\";\n";
    os << "    exec->startCycle = currentCycle;\n";
    
    // Generate per-cycle actions and launches
    os << "    // Per-cycle actions execute immediately\n";
    os << "    // TODO: Add per-cycle action logic\n";
    
    // Look for FutureOp and generate launches
    for (auto& op : rule.getBody().front()) {
      if (auto futureOp = mlir::dyn_cast<txn::FutureOp>(op)) {
        os << "    // Future block launches\n";
        for (auto& innerOp : futureOp.getBody().front()) {
          if (auto launchOp = mlir::dyn_cast<txn::LaunchOp>(innerOp)) {
            os << "    {\n";
            os << "      auto launch = std::make_unique<LaunchState>();\n";
            
            // Handle latency
            if (launchOp.getLatency().has_value()) {
              os << "      launch->hasStaticLatency = true;\n";
              os << "      launch->latency = " << launchOp.getLatency().value() << ";\n";
              os << "      launch->targetCycle = currentCycle + " << launchOp.getLatency().value() << ";\n";
            }
            
            // Handle condition dependency
            if (launchOp.getCondition()) {
              // Track dependency based on launch results
              int depIndex = 0;
              for (auto& innerOp2 : futureOp.getBody().front()) {
                if (auto prevLaunch = mlir::dyn_cast<txn::LaunchOp>(innerOp2)) {
                  if (prevLaunch.getResult() == launchOp.getCondition()) {
                    os << "      launch->conditionName = \"" << rule.getSymName() << "_launch_" << depIndex << "\";\n";
                    break;
                  }
                  depIndex++;
                }
              }
            }
            
            os << "      launch->body = [=]() {\n";
            // Generate launch body
            generateLaunchBody(launchOp.getBody(), "        ");
            os << "      };\n";
            
            os << "      exec->launches.push_back(std::move(launch));\n";
            os << "    }\n";
          }
        }
      }
    }
    
    os << "    return exec;\n";
    os << "  }\n";
  }
  
  void generatePrimitiveState(txn::ModuleOp module) {
    os << "\nprivate:\n";
    os << "  // Primitive state\n";
    
    // Generate state for primitive instances
    for (auto& op : module.getBodyBlock()->getOperations()) {
      if (auto inst = mlir::dyn_cast<txn::InstanceOp>(op)) {
        std::string typeName = inst.getModuleName().str();
        std::string instName = inst.getSymName().str();
        
        if (typeName.find("Register") != std::string::npos) {
          os << "  int32_t " << instName << "_data = 0;\n";
        } else if (typeName.find("Wire") != std::string::npos) {
          os << "  int32_t " << instName << "_data = 0;\n";
        } else if (typeName.find("FIFO") != std::string::npos) {
          // FIFO state: queue and status flags
          os << "  std::queue<int32_t> " << instName << "_queue;\n";
          os << "  static constexpr size_t " << instName << "_depth = 16;\n";
        }
      }
    }
  }
  
  // Removed unused generation methods - keeping only simple translation
};

/// Helper class to generate simulation workspace
class SimulationWorkspaceGenerator {
public:
  SimulationWorkspaceGenerator(llvm::StringRef outputDir) : outputDir(outputDir) {}
  
  mlir::LogicalResult generateWorkspace(txn::ModuleOp module) {
    // Create workspace directory
    if (std::error_code EC = llvm::sys::fs::create_directories(outputDir)) {
      llvm::errs() << "Failed to create directory " << outputDir << ": " << EC.message() << "\n";
      return mlir::failure();
    }
    
    // Generate all workspace files
    if (failed(generateCppFile(module))) return mlir::failure();
    if (failed(generateCMakeFile(module))) return mlir::failure();
    if (failed(generateREADME(module))) return mlir::failure();
    if (failed(generateSimulationHeaders())) return mlir::failure();
    
    return mlir::success();
  }
  
private:
  llvm::StringRef outputDir;
  
  mlir::LogicalResult generateCppFile(txn::ModuleOp module) {
    std::string filename = (outputDir + "/" + module.getName().str() + "_sim.cpp").str();
    std::string errorMessage;
    auto output = mlir::openOutputFile(filename, &errorMessage);
    if (!output) {
      llvm::errs() << "Failed to open " << filename << ": " << errorMessage << "\n";
      return mlir::failure();
    }
    
    CppCodeGenerator generator(output->os());
    generator.generateModule(module);
    output->keep();
    return mlir::success();
  }
  
  mlir::LogicalResult generateCMakeFile(txn::ModuleOp module) {
    std::string filename = (outputDir + "/CMakeLists.txt").str();
    std::string errorMessage;
    auto output = mlir::openOutputFile(filename, &errorMessage);
    if (!output) {
      llvm::errs() << "Failed to open " << filename << ": " << errorMessage << "\n";
      return mlir::failure();
    }
    
    auto& os = output->os();
    os << "cmake_minimum_required(VERSION 3.16)\n";
    os << "project(" << module.getName() << "_simulation)\n\n";
    os << "set(CMAKE_CXX_STANDARD 17)\n";
    os << "set(CMAKE_CXX_STANDARD_REQUIRED ON)\n\n";
    os << "# Add executable\n";
    os << "add_executable(" << module.getName() << "_sim\n";
    os << "  " << module.getName() << "_sim.cpp\n";
    os << "  SimulationBase.h\n";
    os << "  SimulationBase.cpp\n";
    os << ")\n\n";
    os << "# Include directories\n";
    os << "target_include_directories(" << module.getName() << "_sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})\n\n";
    os << "# Link libraries if needed\n";
    os << "# target_link_libraries(" << module.getName() << "_sim PRIVATE ...)\n";
    
    output->keep();
    return mlir::success();
  }
  
  mlir::LogicalResult generateREADME(txn::ModuleOp module) {
    std::string filename = (outputDir + "/README.md").str();
    std::string errorMessage;
    auto output = mlir::openOutputFile(filename, &errorMessage);
    if (!output) {
      llvm::errs() << "Failed to open " << filename << ": " << errorMessage << "\n";
      return mlir::failure();
    }
    
    auto& os = output->os();
    os << "# " << module.getName() << " Simulation\n\n";
    os << "This is a generated transaction-level simulation of the `" << module.getName() << "` module.\n\n";
    os << "## Building\n\n";
    os << "```bash\n";
    os << "mkdir build\n";
    os << "cd build\n";
    os << "cmake ..\n";
    os << "make\n";
    os << "```\n\n";
    os << "## Running\n\n";
    os << "```bash\n";
    os << "./build/" << module.getName() << "_sim [options]\n";
    os << "```\n\n";
    os << "### Options\n\n";
    os << "- `--cycles <n>`: Run simulation for n cycles (default: 100)\n";
    os << "- `--verbose`: Enable verbose output\n";
    os << "- `--stats`: Print performance statistics\n";
    os << "- `--help`: Show help message\n\n";
    os << "## Module Description\n\n";
    os << "The module contains the following methods:\n\n";
    
    // List methods
    for (auto& op : module.getBodyBlock()->getOperations()) {
      if (auto method = mlir::dyn_cast<txn::ValueMethodOp>(op)) {
        os << "- **" << method.getName() << "** (value method): ";
        os << "Returns " << method.getFunctionType().getNumResults() << " value(s)\n";
      } else if (auto method = mlir::dyn_cast<txn::ActionMethodOp>(op)) {
        os << "- **" << method.getName() << "** (action method): ";
        os << "Modifies state\n";
      } else if (auto rule = mlir::dyn_cast<txn::RuleOp>(op)) {
        os << "- **" << rule.getSymName() << "** (rule): ";
        os << "Executes automatically when enabled\n";
      }
    }
    
    os << "\n## Generated from Sharp\n\n";
    os << "This simulation was generated using the Sharp framework's `--sharp-simulate` pass.\n";
    
    output->keep();
    return mlir::success();
  }
  
  mlir::LogicalResult generateSimulationHeaders() {
    // Generate SimulationBase.h
    {
      std::string filename = (outputDir + "/SimulationBase.h").str();
      std::string errorMessage;
      auto output = mlir::openOutputFile(filename, &errorMessage);
      if (!output) {
        llvm::errs() << "Failed to open " << filename << ": " << errorMessage << "\n";
        return mlir::failure();
      }
      
      generateSimulationBaseHeader(output->os());
      output->keep();
    }
    
    // Generate SimulationBase.cpp
    {
      std::string filename = (outputDir + "/SimulationBase.cpp").str();
      std::string errorMessage;
      auto output = mlir::openOutputFile(filename, &errorMessage);
      if (!output) {
        llvm::errs() << "Failed to open " << filename << ": " << errorMessage << "\n";
        return mlir::failure();
      }
      
      generateSimulationBaseImpl(output->os());
      output->keep();
    }
    
    return mlir::success();
  }
  
  void generateSimulationBaseHeader(llvm::raw_ostream& os) {
    os << "#pragma once\n\n";
    os << "#include <string>\n";
    os << "#include <vector>\n";
    os << "#include <map>\n";
    os << "#include <functional>\n";
    os << "#include <memory>\n";
    os << "#include <cstdint>\n";
    os << "#include <unordered_map>\n\n";
    
    os << "// Conflict relations between actions\n";
    os << "enum class ConflictRelation {\n";
    os << "  SequenceBefore = 0,  // SB: First must execute before second\n";
    os << "  SequenceAfter = 1,   // SA: First must execute after second\n";
    os << "  Conflict = 2,        // C: Cannot execute in same cycle\n";
    os << "  ConflictFree = 3     // CF: Can execute in any order\n";
    os << "};\n\n";
    
    os << "// Base class for simulated modules\n";
    os << "class SimModule {\n";
    os << "public:\n";
    os << "  SimModule(const std::string& name) : moduleName(name) {}\n";
    os << "  virtual ~SimModule() = default;\n\n";
    
    os << "  // Register methods\n";
    os << "  void registerValueMethod(const std::string& name,\n";
    os << "                          std::function<std::vector<int64_t>(const std::vector<int64_t>&)> impl) {\n";
    os << "    valueMethods[name] = impl;\n";
    os << "  }\n\n";
    
    os << "  void registerActionMethod(const std::string& name,\n";
    os << "                           std::function<void(const std::vector<int64_t>&)> impl) {\n";
    os << "    actionMethods[name] = impl;\n";
    os << "  }\n\n";
    
    os << "  void registerRule(const std::string& name, std::function<bool()> impl) {\n";
    os << "    rules[name] = impl;\n";
    os << "  }\n\n";
    
    os << "  // Set schedule (action execution order)\n";
    os << "  void setSchedule(const std::vector<std::string>& sched) {\n";
    os << "    schedule = sched;\n";
    os << "  }\n\n";
    
    os << "  // Set conflict matrix\n";
    os << "  void setConflictMatrix(const std::map<std::pair<std::string, std::string>, ConflictRelation>& cm) {\n";
    os << "    conflictMatrix = cm;\n";
    os << "  }\n\n";
    
    os << "  // Execute value method (cached)\n";
    os << "  std::vector<int64_t> callValueMethod(const std::string& name, const std::vector<int64_t>& args) {\n";
    os << "    // Check cache first\n";
    os << "    auto cacheKey = std::make_pair(name, args);\n";
    os << "    auto cached = valueMethodCache.find(cacheKey);\n";
    os << "    if (cached != valueMethodCache.end()) {\n";
    os << "      return cached->second;\n";
    os << "    }\n";
    os << "    \n";
    os << "    // Compute and cache result\n";
    os << "    auto it = valueMethods.find(name);\n";
    os << "    if (it != valueMethods.end()) {\n";
    os << "      auto result = it->second(args);\n";
    os << "      valueMethodCache[cacheKey] = result;\n";
    os << "      return result;\n";
    os << "    }\n";
    os << "    return {};\n";
    os << "  }\n\n";
    
    os << "  // Execute action method\n";
    os << "  void callActionMethod(const std::string& name, const std::vector<int64_t>& args) {\n";
    os << "    auto it = actionMethods.find(name);\n";
    os << "    if (it != actionMethods.end()) {\n";
    os << "      it->second(args);\n";
    os << "    }\n";
    os << "  }\n\n";
    
    os << "  // Check if rule can fire\n";
    os << "  bool canFireRule(const std::string& name) {\n";
    os << "    auto it = rules.find(name);\n";
    os << "    if (it != rules.end()) {\n";
    os << "      return it->second();\n";
    os << "    }\n";
    os << "    return false;\n";
    os << "  }\n\n";
    
    os << "  // Clear value method cache (called at end of cycle)\n";
    os << "  void clearValueCache() {\n";
    os << "    valueMethodCache.clear();\n";
    os << "  }\n\n";
    
    os << "  const std::string& getName() const { return moduleName; }\n";
    os << "  const std::vector<std::string>& getSchedule() const { return schedule; }\n";
    os << "  const std::map<std::pair<std::string, std::string>, ConflictRelation>& getConflictMatrix() const { return conflictMatrix; }\n\n";
    
    os << "protected:\n";
    os << "  std::string moduleName;\n";
    os << "  std::map<std::string, std::function<std::vector<int64_t>(const std::vector<int64_t>&)>> valueMethods;\n";
    os << "  std::map<std::string, std::function<void(const std::vector<int64_t>&)>> actionMethods;\n";
    os << "  std::map<std::string, std::function<bool()>> rules;\n";
    os << "  std::vector<std::string> schedule;\n";
    os << "  std::map<std::pair<std::string, std::string>, ConflictRelation> conflictMatrix;\n";
    os << "  std::map<std::pair<std::string, std::vector<int64_t>>, std::vector<int64_t>> valueMethodCache;\n";
    os << "};\n\n";
    
    os << "// Launch execution state\n";
    os << "struct LaunchState {\n";
    os << "  enum Status { Pending, Running, Completed };\n";
    os << "  Status status = Pending;\n";
    os << "  uint64_t startCycle = 0;\n";
    os << "  uint64_t targetCycle = 0;  // For static latency\n";
    os << "  std::string conditionName; // For dynamic dependency\n";
    os << "  std::function<void()> body;\n";
    os << "  bool hasStaticLatency = false;\n";
    os << "  uint32_t latency = 0;\n";
    os << "};\n\n";
    
    os << "// Multi-cycle execution state for an action\n";
    os << "struct MultiCycleExecution {\n";
    os << "  std::string actionName;\n";
    os << "  uint64_t startCycle;\n";
    os << "  std::vector<std::unique_ptr<LaunchState>> launches;\n";
    os << "  std::map<std::string, bool> launchCompletions; // launch ID -> completion status\n";
    os << "  bool completed = false;\n";
    os << "};\n\n";
    
    os << "// Extended module class with multi-cycle support\n";
    os << "class MultiCycleSimModule : public SimModule {\n";
    os << "public:\n";
    os << "  MultiCycleSimModule(const std::string& name) : SimModule(name) {}\n\n";
    
    os << "  // Track active multi-cycle executions\n";
    os << "  std::vector<std::unique_ptr<MultiCycleExecution>> activeExecutions;\n\n";
    
    os << "  // Register multi-cycle action method\n";
    os << "  void registerMultiCycleAction(const std::string& name,\n";
    os << "                               std::function<std::unique_ptr<MultiCycleExecution>(const std::vector<int64_t>&)> impl) {\n";
    os << "    multiCycleActions[name] = impl;\n";
    os << "  }\n\n";
    
    os << "  // Execute multi-cycle action\n";
    os << "  void startMultiCycleAction(const std::string& name, const std::vector<int64_t>& args, uint64_t currentCycle) {\n";
    os << "    auto it = multiCycleActions.find(name);\n";
    os << "    if (it != multiCycleActions.end()) {\n";
    os << "      auto execution = it->second(args);\n";
    os << "      execution->startCycle = currentCycle;\n";
    os << "      activeExecutions.push_back(std::move(execution));\n";
    os << "    }\n";
    os << "  }\n\n";
    
    os << "  // Update multi-cycle executions\n";
    os << "  void updateMultiCycleExecutions(uint64_t currentCycle) {\n";
    os << "    std::vector<std::unique_ptr<MultiCycleExecution>> stillActive;\n";
    os << "    \n";
    os << "    for (auto& exec : activeExecutions) {\n";
    os << "      if (exec->completed) continue;\n";
    os << "      \n";
    os << "      bool allLaunchesComplete = true;\n";
    os << "      \n";
    os << "      // Update each launch\n";
    os << "      for (size_t i = 0; i < exec->launches.size(); ++i) {\n";
    os << "        auto& launch = exec->launches[i];\n";
    os << "        std::string launchId = exec->actionName + \"_launch_\" + std::to_string(i);\n";
    os << "        \n";
    os << "        if (launch->status == LaunchState::Completed) {\n";
    os << "          exec->launchCompletions[launchId] = true;\n";
    os << "          continue;\n";
    os << "        }\n";
    os << "        \n";
    os << "        allLaunchesComplete = false;\n";
    os << "        \n";
    os << "        // Check if launch can start\n";
    os << "        if (launch->status == LaunchState::Pending) {\n";
    os << "          bool canStart = true;\n";
    os << "          \n";
    os << "          // Check dynamic dependency\n";
    os << "          if (!launch->conditionName.empty()) {\n";
    os << "            canStart = exec->launchCompletions[launch->conditionName];\n";
    os << "          }\n";
    os << "          \n";
    os << "          if (canStart) {\n";
    os << "            launch->status = LaunchState::Running;\n";
    os << "            launch->startCycle = currentCycle;\n";
    os << "            if (launch->hasStaticLatency) {\n";
    os << "              launch->targetCycle = currentCycle + launch->latency;\n";
    os << "            }\n";
    os << "          }\n";
    os << "        }\n";
    os << "        \n";
    os << "        // Check if launch can complete\n";
    os << "        if (launch->status == LaunchState::Running) {\n";
    os << "          bool canComplete = true;\n";
    os << "          \n";
    os << "          if (launch->hasStaticLatency && currentCycle < launch->targetCycle) {\n";
    os << "            canComplete = false;\n";
    os << "          }\n";
    os << "          \n";
    os << "          if (canComplete) {\n";
    os << "            // Try to execute the launch body\n";
    os << "            try {\n";
    os << "              launch->body();\n";
    os << "              launch->status = LaunchState::Completed;\n";
    os << "              exec->launchCompletions[launchId] = true;\n";
    os << "            } catch (...) {\n";
    os << "              // For dynamic launches, retry next cycle\n";
    os << "              if (!launch->hasStaticLatency) {\n";
    os << "                // Keep trying\n";
    os << "              } else {\n";
    os << "                // Static launch failed - this would be a panic\n";
    os << "                std::cerr << \"PANIC: Static launch failed after \" << launch->latency << \" cycles\\n\";\n";
    os << "                throw;\n";
    os << "              }\n";
    os << "            }\n";
    os << "          }\n";
    os << "        }\n";
    os << "      }\n";
    os << "      \n";
    os << "      exec->completed = allLaunchesComplete;\n";
    os << "      if (!exec->completed) {\n";
    os << "        stillActive.push_back(std::move(exec));\n";
    os << "      }\n";
    os << "    }\n";
    os << "    \n";
    os << "    activeExecutions = std::move(stillActive);\n";
    os << "  }\n\n";
    
    os << "private:\n";
    os << "  std::map<std::string, std::function<std::unique_ptr<MultiCycleExecution>(const std::vector<int64_t>&)>> multiCycleActions;\n";
    
    os << "};\n\n";
    
    os << "// Main simulation driver\n";
    os << "class Simulator {\n";
    os << "public:\n";
    os << "  Simulator() : cycles(0), verbose(false), dumpStats(false) {}\n\n";
    
    os << "  void addModule(std::unique_ptr<SimModule> module) {\n";
    os << "    modules.push_back(std::move(module));\n";
    os << "  }\n\n";
    
    os << "  void run(int maxCycles);\n";
    os << "  void setVerbose(bool v) { verbose = v; }\n";
    os << "  void setDumpStats(bool d) { dumpStats = d; }\n\n";
    
    os << "private:\n";
    os << "  std::vector<std::unique_ptr<SimModule>> modules;\n";
    os << "  uint64_t cycles;\n";
    os << "  bool verbose;\n";
    os << "  bool dumpStats;\n";
    os << "};\n";
  }
  
  void generateSimulationBaseImpl(llvm::raw_ostream& os) {
    os << "#include \"SimulationBase.h\"\n";
    os << "#include <iostream>\n";
    os << "#include <chrono>\n";
    os << "#include <set>\n\n";
    
    os << "void Simulator::run(int maxCycles) {\n";
    os << "  if (verbose) {\n";
    os << "    std::cout << \"Starting simulation for \" << maxCycles << \" cycles\\n\";\n";
    os << "  }\n\n";
    
    os << "  auto startTime = std::chrono::high_resolution_clock::now();\n\n";
    
    os << "  for (cycles = 0; cycles < maxCycles; ++cycles) {\n";
    os << "    if (verbose && cycles % 100 == 0) {\n";
    os << "      std::cout << \"Cycle \" << cycles << \"\\n\";\n";
    os << "    }\n\n";
    
    os << "    // Execute each module following the three-phase execution model\n";
    os << "    for (auto& module : modules) {\n";
    os << "      // Check if this is a multi-cycle module\n";
    os << "      auto* mcModule = dynamic_cast<MultiCycleSimModule*>(module.get());\n";
    os << "      \n";
    os << "      // Phase 1: Value Phase - Compute all value methods (done lazily via caching)\n";
    os << "      // Value methods are computed on-demand and cached in callValueMethod()\n\n";
    
    os << "      // Update multi-cycle executions if this is a multi-cycle module\n";
    os << "      if (mcModule) {\n";
    os << "        mcModule->updateMultiCycleExecutions(cycles);\n";
    os << "      }\n\n";
    
    os << "      // Phase 2: Execution Phase - Execute actions in schedule order\n";
    os << "      const auto& schedule = module->getSchedule();\n";
    os << "      const auto& conflictMatrix = module->getConflictMatrix();\n";
    os << "      std::set<std::string> executedActions;\n\n";
    
    os << "      for (const auto& actionName : schedule) {\n";
    os << "        // Check if this is a rule that can fire\n";
    os << "        bool canExecute = false;\n";
    os << "        bool isRule = module->canFireRule(actionName);\n";
    os << "        \n";
    os << "        if (isRule) {\n";
    os << "          canExecute = true; // Rule guard already checked\n";
    os << "        } else {\n";
    os << "          // For action methods, we assume they're always enabled in simulation\n";
    os << "          // In real hardware, parent module would control enabling\n";
    os << "          canExecute = true;\n";
    os << "        }\n\n";
    
    os << "        // Check conflicts with already executed actions\n";
    os << "        bool hasConflict = false;\n";
    os << "        for (const auto& executed : executedActions) {\n";
    os << "          auto conflictKey = std::make_pair(executed, actionName);\n";
    os << "          auto it = conflictMatrix.find(conflictKey);\n";
    os << "          if (it != conflictMatrix.end() && it->second == ConflictRelation::Conflict) {\n";
    os << "            hasConflict = true;\n";
    os << "            break;\n";
    os << "          }\n";
    os << "        }\n\n";
    
    os << "        if (canExecute && !hasConflict) {\n";
    os << "          // Check if this is a multi-cycle action\n";
    os << "          bool isMultiCycle = false;\n";
    os << "          if (mcModule) {\n";
    os << "            // Check if action has active multi-cycle execution\n";
    os << "            for (const auto& exec : mcModule->activeExecutions) {\n";
    os << "              if (exec->actionName == actionName && !exec->completed) {\n";
    os << "                isMultiCycle = true;\n";
    os << "                break;\n";
    os << "              }\n";
    os << "            }\n";
    os << "          }\n";
    os << "          \n";
    os << "          if (!isMultiCycle) {\n";
    os << "            // Execute the action\n";
    os << "            if (isRule) {\n";
    os << "              // For rules, call the associated action method or inline logic\n";
    os << "              module->callActionMethod(actionName, {});\n";
    os << "            } else {\n";
    os << "              // For action methods, execute them\n";
    os << "              module->callActionMethod(actionName, {});\n";
    os << "            }\n";
    os << "          }\n";
    os << "          executedActions.insert(actionName);\n";
    os << "        }\n";
    os << "      }\n\n";
    
    os << "      // Phase 3: Commit Phase - Clear caches for next cycle\n";
    os << "      module->clearValueCache();\n";
    os << "    }\n";
    os << "  }\n\n";
    
    os << "  auto endTime = std::chrono::high_resolution_clock::now();\n";
    os << "  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);\n\n";
    
    os << "  if (dumpStats) {\n";
    os << "    std::cout << \"\\nSimulation Statistics:\\n\";\n";
    os << "    std::cout << \"  Total cycles: \" << cycles << \"\\n\";\n";
    os << "    std::cout << \"  Execution time: \" << duration.count() << \" ms\\n\";\n";
    os << "    std::cout << \"  Cycles per second: \" << (cycles * 1000.0 / duration.count()) << \"\\n\";\n";
    os << "  }\n";
    os << "}\n";
  }
};

/// JIT execution context for txn modules
class JitExecutionContext {
public:
  JitExecutionContext(mlir::ModuleOp module) : module(module) {}
  
  mlir::LogicalResult initialize() {
    // Create execution engine options
    mlir::ExecutionEngineOptions options;
    // We don't need a transformer since translation interfaces are registered globally
    
    auto maybeEngine = mlir::ExecutionEngine::create(module, options);
    if (!maybeEngine) {
      // Log the error before consuming it
      auto err = maybeEngine.takeError();
      llvm::handleAllErrors(std::move(err), [&](const llvm::ErrorInfoBase &E) {
        module.emitError() << "cannot be converted to LLVM IR: missing `LLVMTranslationDialectInterface` for dialect: " << E.message();
      });
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
  
  std::string workspaceDir;  // Store workspace directory
  
  // Constructor needed for TableGen
  TxnSimulatePass() = default;
  TxnSimulatePass(const TxnSimulatePassOptions& options) {
    mode = options.mode;
    outputFile = options.outputFile;
    verbose = options.verbose;
    dumpStats = options.dumpStats;
    maxCycles = options.maxCycles;
    
    // Override: always generate workspace for translation mode
    if (mode == "translation" && !outputFile.empty() && outputFile != "-") {
      workspaceDir = outputFile;
      outputFile = "";  // Clear to avoid file output
    }
  }

  void runOnOperation() override {
    auto module = getOperation();
    
    // Parse mode option
    bool isJitMode = (mode == "jit");
    
    // Remove debug output
    // llvm::errs() << "TxnSimulatePass: mode=" << mode << ", outputFile=" << outputFile << ", verbose=" << verbose << "\n";
    
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
      
      // Convert SCF to control flow
      pm.addPass(mlir::createSCFToControlFlowPass());
      
      // Convert arith to LLVM
      pm.addPass(mlir::createArithToLLVMConversionPass());
      
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
      
      // Check if any txn operations remain after lowering
      bool hasTxnOps = false;
      moduleClone.walk([&](mlir::Operation *op) {
        if (op->getDialect() && op->getDialect()->getNamespace() == "txn") {
          hasTxnOps = true;
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });
      
      if (hasTxnOps) {
        module.emitError() << "cannot be converted to LLVM IR: missing `LLVMTranslationDialectInterface`";
        signalPassFailure();
        return;
      }
      
      // For JIT mode, replace the original module with the lowered LLVM module
      // This allows users to see the lowered LLVM IR
      auto moduleOp = cast<mlir::ModuleOp>(module);
      auto clonedModuleOp = cast<mlir::ModuleOp>(moduleClone);
      
      // Clear the original module and copy all operations from the cloned module
      moduleOp.getBody()->clear();
      for (auto &op : clonedModuleOp.getBody()->getOperations()) {
        auto clonedOp = op.clone();
        moduleOp.getBody()->push_back(clonedOp);
      }
      
      if (verbose) {
        llvm::errs() << "JIT lowering completed successfully\n";
      }
    } else {
      // Translation mode: generate workspace or C++ code
      
      // Check if we should generate workspace
      bool isWorkspaceMode = !workspaceDir.empty();
      
      if (verbose) {
        llvm::errs() << "outputFile: " << outputFile << "\n";
        llvm::errs() << "workspaceDir: " << workspaceDir << "\n";
        llvm::errs() << "isWorkspaceMode: " << isWorkspaceMode << "\n";
      }
      
      if (isWorkspaceMode) {
        // Workspace mode: generate complete simulation workspace
        if (verbose) {
          llvm::errs() << "Generating simulation workspace in " << workspaceDir << "\n";
        }
        
        SimulationWorkspaceGenerator workspaceGen(workspaceDir);
        
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
        
        if (failed(workspaceGen.generateWorkspace(topModule))) {
          module.emitError() << "failed to generate simulation workspace";
          signalPassFailure();
          return;
        }
        
        if (verbose) {
          llvm::errs() << "Successfully generated simulation workspace\n";
          llvm::errs() << "To build and run:\n";
          llvm::errs() << "  cd " << workspaceDir << "\n";
          llvm::errs() << "  mkdir build && cd build\n";
          llvm::errs() << "  cmake .. && make\n";
          llvm::errs() << "  ./" << topModule.getName() << "_sim\n";
        }
      } else {
        // Single file mode: generate C++ code only
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
  }
}; // End of TxnSimulatePass struct

} // namespace

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