//===- Passes.td - Sharp Simulation Passes Definition ------*- tablegen -*-===//
//
// This file defines simulation-related passes for Sharp.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_SIMULATION_PASSES_TD
#define SHARP_SIMULATION_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// TxnSimulate
//===----------------------------------------------------------------------===//

def TxnSimulatePass : Pass<"sharp-simulate", "mlir::ModuleOp"> {
  let summary = "Simulate transaction-level modules";
  let description = [{
    This pass simulates txn.module operations at the transaction level using
    Sharp's event-driven simulation framework. It supports two modes:
    
    1. Translation mode: Generates C++ code that uses the simulation API
    2. JIT mode: Compiles MLIR to executable code on-the-fly
    
    The pass converts txn.module operations into SimModule instances,
    handles method implementations, extracts scheduling information,
    and sets up the event-driven simulation infrastructure.
  }];

  let options = [
    Option<"mode", "mode", "std::string", /*default=*/"\"jit\"",
           "Simulation mode: 'translation' or 'jit'">,
    Option<"outputFile", "output", "std::string", /*default=*/"\"\"",
           "Output file for translation mode">,
    Option<"verbose", "verbose", "bool", /*default=*/"false",
           "Enable verbose simulation output">,
    Option<"dumpStats", "dump-stats", "bool", /*default=*/"false",
           "Dump simulation statistics">,
    Option<"maxCycles", "max-cycles", "unsigned", /*default=*/"1000000",
           "Maximum number of simulation cycles">
  ];

  let dependentDialects = [
    "sharp::txn::TxnDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::func::FuncDialect",
    "mlir::ub::UBDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ConcurrentSimulation
//===----------------------------------------------------------------------===//

def ConcurrentSimulationPass : Pass<"sharp-concurrent-sim", "mlir::ModuleOp"> {
  let summary = "Enable concurrent simulation using DAM methodology";
  let description = [{
    This pass transforms the simulation to support concurrent execution
    following the DAM (Discrete-event simulation with Adaptive Multiprocessing)
    methodology. It analyzes module dependencies and schedules concurrent
    execution of non-conflicting events.
    
    The pass:
    - Identifies parallelizable events
    - Constructs dependency graphs
    - Implements work-stealing schedulers
    - Manages thread-safe state updates
  }];

  let options = [
    Option<"numThreads", "num-threads", "unsigned", /*default=*/"0",
           "Number of simulation threads (0 = auto)">,
    Option<"granularity", "granularity", "std::string", /*default=*/"\"adaptive\"",
           "Parallelization granularity: 'fine', 'coarse', or 'adaptive'">
  ];

  let dependentDialects = [
    "sharp::txn::TxnDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ArcilatorIntegration
//===----------------------------------------------------------------------===//

def ArcilatorIntegrationPass : Pass<"sharp-arcilator", "mlir::ModuleOp"> {
  let summary = "Integrate with CIRCT's arcilator for RTL simulation";
  let description = [{
    This pass enables RTL-level simulation by integrating with CIRCT's
    arcilator infrastructure. It converts suitable modules to arc dialect
    and sets up the arcilator simulation environment.
    
    The pass handles:
    - Conversion from txn/hw to arc dialect
    - Setting up arcilator state management
    - Clock and reset handling
    - VCD trace generation setup
  }];

  let options = [
    Option<"enableTracing", "enable-tracing", "bool", /*default=*/"false",
           "Enable VCD trace generation">,
    Option<"traceFile", "trace-file", "std::string", /*default=*/"\"trace.vcd\"",
           "VCD trace output file">
  ];

  let dependentDialects = [
    "sharp::txn::TxnDialect",
    "circt::hw::HWDialect",
    "circt::arc::ArcDialect",
    "circt::firrtl::FIRRTLDialect",
    "circt::seq::SeqDialect",
    "circt::comb::CombDialect",
    "circt::emit::EmitDialect",
    "circt::sv::SVDialect",
    "circt::sim::SimDialect",
    "circt::verif::VerifDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::ub::UBDialect"
  ];
}

//===----------------------------------------------------------------------===//
// HybridSimulation
//===----------------------------------------------------------------------===//

def HybridSimulationPass : Pass<"sharp-hybrid-sim", "mlir::ModuleOp"> {
  let summary = "Generate hybrid TL-RTL simulation infrastructure";
  let description = [{
    This pass generates hybrid simulation infrastructure that bridges
    transaction-level and RTL simulation domains. It creates synchronization
    mechanisms between TL simulation and RTL simulation (via arcilator).
    
    The pass:
    - Generates bridge configuration for module/method mappings
    - Creates TL simulation stubs that interface with RTL
    - Sets up time synchronization between domains
    - Implements method call translation between TL and RTL
    - Provides different synchronization modes (lockstep/decoupled/adaptive)
    
    Prerequisites:
    - Run --sharp-arcilator first to generate RTL representation
    - Ensure both TL and RTL representations are available
  }];

  let options = [
    Option<"outputFile", "output", "std::string", /*default=*/"\"\"",
           "Output file for generated hybrid simulation code">,
    Option<"syncMode", "sync-mode", "std::string", /*default=*/"\"lockstep\"",
           "Synchronization mode: 'lockstep', 'decoupled', or 'adaptive'">,
    Option<"configFile", "config", "std::string", /*default=*/"\"\"",
           "External JSON configuration file for bridge setup">
  ];

  let dependentDialects = [
    "sharp::txn::TxnDialect",
    "circt::arc::ArcDialect",
    "circt::hw::HWDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect"
  ];
}

#endif // SHARP_SIMULATION_PASSES_TD