//===- SimulationOps.td - Sharp Simulation Operations ---------------------===//
//
// Defines operations for Sharp's simulation framework.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_SIMULATION_OPS
#define SHARP_SIMULATION_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "sharp/Dialect/Txn/TxnOps.td"

def Sharp_SimulationDialect : Dialect {
  let name = "sharp.sim";
  let summary = "Sharp simulation dialect";
  let description = [{
    This dialect provides operations for configuring and controlling
    Sharp's multi-level simulation framework.
  }];
  
  let cppNamespace = "::sharp::sim";
}

class Sim_Op<string mnemonic, list<Trait> traits = []> :
    Op<Sharp_SimulationDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Simulation Configuration
//===----------------------------------------------------------------------===//

def SimConfigOp : Sim_Op<"config", [Symbol]> {
  let summary = "Simulation configuration";
  let description = [{
    Defines a simulation configuration including the module to simulate,
    simulation parameters, and performance tracking options.
    
    Example:
    ```mlir
    sharp.sim @main {
      module = @TopModule,
      max_cycles = 1000000 : i64,
      debug = true,
      profile = true
    }
    ```
  }];
  
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$module,
    OptionalAttr<I64Attr>:$max_cycles,
    OptionalAttr<BoolAttr>:$debug,
    OptionalAttr<BoolAttr>:$profile,
    OptionalAttr<ArrayAttr>:$track_methods
  );
  
  let assemblyFormat = [{
    $sym_name `{` 
      `module` `=` $module 
      (`,` `max_cycles` `=` $max_cycles^)?
      (`,` `debug` `=` $debug^)?
      (`,` `profile` `=` $profile^)?
      (`,` `track_methods` `=` $track_methods^)?
    `}`
  }];
}

//===----------------------------------------------------------------------===//
// Bridge Configuration
//===----------------------------------------------------------------------===//

def BridgeConfigOp : Sim_Op<"bridge", [Symbol]> {
  let summary = "TL-to-RTL bridge configuration";
  let description = [{
    Configures the bridge between transaction-level and RTL simulation domains.
    Specifies how TL method calls map to RTL signal interfaces.
  }];
  
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    DictionaryAttr:$method_mapping
  );
  
  let assemblyFormat = [{
    $sym_name `{` `method_mapping` `=` $method_mapping `}`
  }];
}

//===----------------------------------------------------------------------===//
// Spec Primitives
//===----------------------------------------------------------------------===//

def SpecMultiCycleOp : Sim_Op<"spec.multi_cycle",
    [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Multi-cycle specification primitive";
  let description = [{
    Represents a multi-cycle operation in the specification.
    The operation takes a specified number of cycles to complete.
    
    Example:
    ```mlir
    %result = txn.spec.multi_cycle<5> {
      %processed = arith.muli %input, %input : i32
      txn.yield %processed : i32
    } : i32
    ```
  }];
  
  let arguments = (ins I32Attr:$cycles);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  
  let assemblyFormat = [{
    `<` $cycles `>` $body attr-dict `:` type($results)
  }];
}

def SpecAssertOp : Sim_Op<"spec.assert"> {
  let summary = "Specification assertion";
  let description = [{
    Assert a condition during specification execution.
    Used for verification in transaction-level simulation.
  }];
  
  let arguments = (ins
    I1:$condition,
    OptionalAttr<StrAttr>:$message
  );
  
  let assemblyFormat = [{
    $condition (`message` $message^)? attr-dict
  }];
}

def SpecRandomOp : Sim_Op<"spec.random"> {
  let summary = "Generate random value for testing";
  let description = [{
    Generates a random value within specified bounds.
    Used in testbenches for stimulus generation.
  }];
  
  let arguments = (ins
    AnyInteger:$min,
    AnyInteger:$max
  );
  let results = (outs AnyInteger:$result);
  
  let assemblyFormat = [{
    `(` $min `,` $max `)` attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Performance Tracking
//===----------------------------------------------------------------------===//

def PerfStartOp : Sim_Op<"perf.start"> {
  let summary = "Start performance measurement";
  let description = [{
    Marks the beginning of a performance measurement region.
  }];
  
  let arguments = (ins StrAttr:$region_name);
  let assemblyFormat = [{ $region_name attr-dict }];
}

def PerfEndOp : Sim_Op<"perf.end"> {
  let summary = "End performance measurement";
  let description = [{
    Marks the end of a performance measurement region.
  }];
  
  let arguments = (ins StrAttr:$region_name);
  let assemblyFormat = [{ $region_name attr-dict }];
}

#endif // SHARP_SIMULATION_OPS