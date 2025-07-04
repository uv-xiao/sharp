//===- HybridSimulationPass.cpp - Hybrid TL-RTL Simulation Pass ----------===//
//
// This pass generates hybrid simulation infrastructure that bridges
// transaction-level and RTL simulation domains.
//
//===----------------------------------------------------------------------===//

#include "sharp/Simulation/Passes.h"
#include "sharp/Dialect/Txn/TxnOps.h"
#include "sharp/Conversion/Passes.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include <sstream>

namespace sharp {

#define GEN_PASS_DEF_HYBRIDSIMULATIONPASS  
#include "sharp/Simulation/Passes.h.inc"

namespace {

/// Code generator for hybrid simulation bridge
class HybridCodeGenerator {
public:
  HybridCodeGenerator(llvm::raw_ostream& os) : os(os), indentLevel(0) {}
  
  void generate(mlir::ModuleOp module) {
    generateIncludes();
    generateBridgeConfig(module);
    generateMainFunction(module);
  }

private:
  llvm::raw_ostream& os;
  int indentLevel;
  
  void indent() { 
    for (int i = 0; i < indentLevel; i++) os << "  "; 
  }
  
  void generateIncludes() {
    os << "// Generated Hybrid TL-RTL Simulation Code\n";
    os << "#include \"sharp/Simulation/Hybrid/HybridBridge.h\"\n";
    os << "#include \"sharp/Simulation/Core/Simulator.h\"\n";
    os << "#include \"sharp/Simulation/Core/SimModule.h\"\n";
    os << "#include <iostream>\n";
    os << "#include <memory>\n";
    os << "#include <string>\n";
    os << "#include <thread>\n\n";
    os << "using namespace sharp::simulation;\n\n";
  }
  
  void generateBridgeConfig(mlir::ModuleOp module) {
    os << "// Bridge configuration\n";
    os << "const char* BRIDGE_CONFIG = R\"({\n";
    os << "  \"sync_mode\": \"lockstep\",\n";
    os << "  \"max_time_divergence\": 1000,\n";
    os << "  \"module_mappings\": [\n";
    
    bool first = true;
    module.walk([&](::sharp::txn::ModuleOp txnModule) {
      if (!first) os << ",\n";
      first = false;
      
      os << "    {\n";
      os << "      \"tl_module\": \"" << txnModule.getName() << "\",\n";
      os << "      \"rtl_module\": \"" << txnModule.getName() << "_rtl\"\n";
      os << "    }";
    });
    
    os << "\n  ],\n";
    os << "  \"method_mappings\": [\n";
    
    // Generate method mappings
    first = true;
    module.walk([&](::sharp::txn::ModuleOp txnModule) {
      txnModule.walk([&](::sharp::txn::ValueMethodOp method) {
        if (!first) os << ",\n";
        first = false;
        
        os << "    {\n";
        os << "      \"method_name\": \"" << method.getName() << "\",\n";
        os << "      \"input_signals\": [";
        
        // Generate input signal names
        for (size_t i = 0; i < method.getNumArguments(); i++) {
          if (i > 0) os << ", ";
          os << "\"" << method.getName() << "_arg" << i << "\"";
        }
        
        os << "],\n";
        os << "      \"output_signals\": [\"" << method.getName() << "_result\"],\n";
        os << "      \"enable_signal\": \"" << method.getName() << "_en\",\n";
        os << "      \"ready_signal\": \"" << method.getName() << "_rdy\"\n";
        os << "    }";
      });
      
      txnModule.walk([&](::sharp::txn::ActionMethodOp method) {
        if (!first) os << ",\n";
        first = false;
        
        os << "    {\n";
        os << "      \"method_name\": \"" << method.getName() << "\",\n";
        os << "      \"input_signals\": [";
        
        // Generate input signal names
        for (size_t i = 0; i < method.getNumArguments(); i++) {
          if (i > 0) os << ", ";
          os << "\"" << method.getName() << "_arg" << i << "\"";
        }
        
        os << "],\n";
        os << "      \"output_signals\": [],\n";
        os << "      \"enable_signal\": \"" << method.getName() << "_en\",\n";
        os << "      \"ready_signal\": \"" << method.getName() << "_rdy\"\n";
        os << "    }";
      });
    });
    
    os << "\n  ]\n";
    os << "})\";\n\n";
  }
  
  void generateTLModules(mlir::ModuleOp module) {
    module.walk([&](::sharp::txn::ModuleOp txnModule) {
      std::string className = txnModule.getName().str() + "TLSim";
      
      os << "// Transaction-level simulation for " << txnModule.getName() << "\n";
      os << "class " << className << " : public SimModule {\n";
      os << "public:\n";
      indentLevel++;
      
      indent(); os << className << "() : SimModule(\"" << txnModule.getName() << "\") {\n";
      indentLevel++;
      
      // Register methods
      txnModule.walk([&](::sharp::txn::ValueMethodOp method) {
        indent(); os << "registerMethod(\"" << method.getName() << "\", "
                     << "[this](const std::vector<uint64_t>& args) -> std::vector<uint64_t> {\n";
        indentLevel++;
        indent(); os << "// TL implementation of " << method.getName() << "\n";
        indent(); os << "return {0}; // Placeholder\n";
        indentLevel--;
        indent(); os << "});\n";
      });
      
      txnModule.walk([&](::sharp::txn::ActionMethodOp method) {
        indent(); os << "registerMethod(\"" << method.getName() << "\", "
                     << "[this](const std::vector<uint64_t>& args) -> std::vector<uint64_t> {\n";
        indentLevel++;
        indent(); os << "// TL implementation of " << method.getName() << "\n";
        indent(); os << "return {};\n";
        indentLevel--;
        indent(); os << "});\n";
      });
      
      indentLevel--;
      indent(); os << "}\n";
      
      indentLevel--;
      os << "};\n\n";
    });
  }
  
  void generateMainFunction(mlir::ModuleOp module) {
    generateTLModules(module);
    
    os << "int main(int argc, char** argv) {\n";
    indentLevel++;
    
    indent(); os << "std::cout << \"Starting Hybrid TL-RTL Simulation\" << std::endl;\n\n";
    
    // Create simulator and bridge
    indent(); os << "// Create TL simulator\n";
    indent(); os << "auto tlSimulator = std::make_unique<Simulator>();\n\n";
    
    indent(); os << "// Create hybrid bridge\n";
    indent(); os << "auto bridge = HybridBridgeFactory::create(\"\", SyncMode::Lockstep);\n";
    indent(); os << "bridge->configure(BRIDGE_CONFIG);\n\n";
    
    indent(); os << "// Create RTL simulator backend\n";
    indent(); os << "auto rtlSim = HybridBridgeFactory::createRTLSimulator(\"arcilator\");\n";
    indent(); os << "if (!rtlSim) {\n";
    indentLevel++;
    indent(); os << "std::cerr << \"Failed to create RTL simulator\" << std::endl;\n";
    indent(); os << "return 1;\n";
    indentLevel--;
    indent(); os << "}\n\n";
    
    // Initialize RTL with converted module
    indent(); os << "// Initialize RTL simulator with Arc module\n";
    indent(); os << "// Note: This assumes the Arc module was generated by --sharp-arcilator\n";
    indent(); os << "rtlSim->initialize(\"output.arc.mlir\");\n";
    indent(); os << "bridge->setRTLSimulator(std::move(rtlSim));\n\n";
    
    // Create and connect TL modules
    indent(); os << "// Create and connect TL modules\n";
    module.walk([&](::sharp::txn::ModuleOp txnModule) {
      std::string varName = txnModule.getName().str() + "_tl";
      std::string className = txnModule.getName().str() + "TLSim";
      
      indent(); os << "auto " << varName << " = std::make_shared<" << className << ">();\n";
      indent(); os << "tlSimulator->addModule(\"" << txnModule.getName() << "\", " 
                   << varName << ");\n";
      indent(); os << "bridge->connectModule(\"" << txnModule.getName() << "\", \""
                   << txnModule.getName() << "_rtl\", " << varName << ");\n\n";
    });
    
    // Start bridge
    indent(); os << "// Start hybrid bridge\n";
    indent(); os << "bridge->start();\n\n";
    
    // Create test stimulus
    indent(); os << "// Run test stimulus\n";
    indent(); os << "std::cout << \"Running hybrid simulation...\" << std::endl;\n\n";
    
    indent(); os << "// Example: Schedule some TL events\n";
    indent(); os << "for (int cycle = 0; cycle < 10; cycle++) {\n";
    indentLevel++;
    indent(); os << "tlSimulator->advanceTime(cycle);\n";
    indent(); os << "bridge->synchronizeTime(cycle);\n";
    indent(); os << "std::this_thread::sleep_for(std::chrono::milliseconds(10));\n";
    indentLevel--;
    indent(); os << "}\n\n";
    
    // Print statistics
    indent(); os << "// Print statistics\n";
    indent(); os << "auto stats = bridge->getStatistics();\n";
    indent(); os << "std::cout << \"\\nHybrid Simulation Statistics:\" << std::endl;\n";
    indent(); os << "std::cout << \"  Method calls: \" << stats.methodCalls << std::endl;\n";
    indent(); os << "std::cout << \"  State updates: \" << stats.stateUpdates << std::endl;\n";
    indent(); os << "std::cout << \"  Time syncs: \" << stats.timeSyncs << std::endl;\n\n";
    
    // Cleanup
    indent(); os << "// Stop bridge\n";
    indent(); os << "bridge->stop();\n\n";
    
    indent(); os << "std::cout << \"Hybrid simulation completed\" << std::endl;\n";
    indent(); os << "return 0;\n";
    
    indentLevel--;
    os << "}\n";
  }
};

/// The hybrid simulation pass
struct HybridSimulationPass : public impl::HybridSimulationPassBase<HybridSimulationPass> {
  using Base = impl::HybridSimulationPassBase<HybridSimulationPass>;
  
  HybridSimulationPass() = default;
  HybridSimulationPass(const HybridSimulationPassOptions& options) {
    outputFile = options.outputFile;
    syncMode = options.syncMode;
    configFile = options.configFile;
  }

  void runOnOperation() override {
    auto module = getOperation();
    
    // Check if we have both TL and RTL representations
    bool hasTxn = false;
    bool hasArc = false;
    
    module.walk([&](mlir::Operation* op) {
      if (isa<::sharp::txn::ModuleOp>(op)) hasTxn = true;
      if (op->getDialect() && 
          op->getDialect()->getNamespace() == "arc") hasArc = true;
    });
    
    if (!hasTxn) {
      module.emitError() << "No transaction-level modules found for hybrid simulation";
      signalPassFailure();
      return;
    }
    
    // Generate hybrid simulation code
    std::string output;
    llvm::raw_string_ostream os(output);
    
    HybridCodeGenerator generator(os);
    generator.generate(module);
    
    // Write to file if specified
    if (!outputFile.empty()) {
      std::error_code ec;
      llvm::raw_fd_ostream fileOs(outputFile, ec);
      if (ec) {
        module.emitError() << "Failed to open output file: " << ec.message();
        signalPassFailure();
        return;
      }
      fileOs << os.str();
      module.emitRemark() << "Generated hybrid simulation code to: " << outputFile;
    } else {
      // Print to stdout
      llvm::outs() << os.str();
    }
    
    // Provide instructions
    module.emitRemark() << "Hybrid Simulation Setup:";
    module.emitRemark() << "1. Run --sharp-arcilator to generate RTL representation";
    module.emitRemark() << "2. Compile the generated C++ code with simulation libraries";
    module.emitRemark() << "3. Run the executable for hybrid TL-RTL simulation";
    
    if (!configFile.empty()) {
      module.emitRemark() << "Using bridge configuration from: " << configFile;
    }
  }
};

} // namespace

// Note: The createHybridSimulationPass function is generated by TableGen

} // namespace sharp