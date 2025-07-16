//===- TxnOps.td - Txn dialect operation definitions ----------------------===//
//
// Part of the Sharp Project.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_DIALECT_TXN_OPS_TD
#define SHARP_DIALECT_TXN_OPS_TD

include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "sharp/Dialect/Txn/TxnTypes.td"

//===----------------------------------------------------------------------===//
// Module and Primitive Operations
//===----------------------------------------------------------------------===//

def ModuleOp : TxnOp<"module", [
    IsolatedFromAbove, Symbol, SymbolTable, NoTerminator
  ]> {
  let summary = "Transaction module definition";
  let description = [{
    Defines a module containing instances, primitives, methods, rules,
    and a schedule that defines the module's behavior.
  }];

  let arguments = (ins 
    SymbolNameAttr:$sym_name,
    OptionalAttr<ArrayAttr>:$type_parameters  // Array of TypeAttr for parametric modules
  );
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    /// Returns the entry block of the module.
    ::mlir::Block *getBodyBlock() { return &getBody().front(); }
    
    /// Returns the full name including type parameters (e.g., "Register<!firrtl.uint<32>>")
    std::string getFullName() {
      return ::sharp::txn::module_name_with_type_args(getName(), getTypeParameters().value_or(::mlir::ArrayAttr()));
    }
  }];

  let assemblyFormat = [{
    $sym_name attr-dict-with-keyword $body
  }];
  let hasVerifier = 1;
}

def PrimitiveOp : TxnOp<"primitive", [
    IsolatedFromAbove, Symbol, NoTerminator, SymbolTable
  ]> {
  let summary = "Primitive module definition";
  let description = [{
    Defines a primitive module that can be either synthesizable (using CIRCT
    dialects) or non-synthesizable (using general MLIR operations).
    
    Primitives can be parameterized with type parameters, allowing for
    generic definitions like Register<T> or Wire<T>.
  }];

  let arguments = (ins 
    SymbolNameAttr:$sym_name,
    StrAttr:$type,  // "hw" for synthesizable, "spec" for specification
    TypeAttr:$interface_type,
    OptionalAttr<ArrayAttr>:$type_parameters  // Array of TypeAttr for parametric primitives
  );
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    /// Returns the full name including type parameters (e.g., "Register<!firrtl.uint<32>>")
    std::string getFullName() {
      return ::sharp::txn::module_name_with_type_args(getName(), getTypeParameters().value_or(::mlir::ArrayAttr()));
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def InstanceOp : TxnOp<"instance", [Symbol]> {
  let summary = "Module instance declaration";
  let description = [{
    Declares an instance of another module within the current module.
    
    For parametric primitives, type arguments can be provided to instantiate
    the primitive with specific types. For example:
      %reg = txn.instance @reg of @Register<i32> : !txn.module<"Register">
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$module_name,
    OptionalAttr<ArrayAttr>:$type_arguments  // Array of TypeAttr for instantiating parametric modules
  );
  let results = (outs Txn_ModuleType:$result);

  let extraClassDeclaration = [{
    /// Returns the full module name including type arguments (e.g., "Register<!firrtl.uint<32>>")
    std::string getFullModuleName() {
      return ::sharp::txn::module_name_with_type_args(getModuleName(), getTypeArguments().value_or(::mlir::ArrayAttr()));
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Method and Rule Operations
//===----------------------------------------------------------------------===//

def ValueMethodOp : TxnOp<"value_method", [
    IsolatedFromAbove, FunctionOpInterface, AutomaticAllocationScope
  ]> {
  let summary = "Pure value method definition";
  let description = [{
    Defines a pure, read-only method that returns a value without side effects.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    OptionalAttr<StrAttr>:$result,          // Result signal postfix (default="OUT")
    OptionalAttr<StrAttr>:$prefix           // Method name prefix for translation
  );
  let regions = (region SizedRegion<1>:$body);


  let extraClassDeclaration = [{
    /// Returns the argument types of this method.
    ::mlir::ArrayRef<::mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this method.
    ::mlir::ArrayRef<::mlir::Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Returns the region that is callable.
    ::mlir::Region *getCallableRegion() { return &getBody(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

def ActionMethodOp : TxnOp<"action_method", [
    IsolatedFromAbove, FunctionOpInterface, AutomaticAllocationScope
  ]> {
  let summary = "Action method definition";
  let description = [{
    Defines an action method that may modify state or abort.
    Can optionally include a guard region that returns a condition.
    If the guard region returns false, the method will abort.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    OptionalAttr<StrAttr>:$ready,          // Ready signal postfix (default="RDY")
    OptionalAttr<StrAttr>:$enable,         // Enable signal postfix (default="EN")
    OptionalAttr<StrAttr>:$result,         // Result signal postfix (default="OUT")
    OptionalAttr<StrAttr>:$prefix,         // Method name prefix for translation
    OptionalAttr<UnitAttr>:$always_ready,  // Method is always ready (removes ready signal)
    OptionalAttr<UnitAttr>:$always_enable  // Method is always enabled (removes enable signal)
  );
  let regions = (region SizedRegion<1>:$body, VariadicRegion<AnyRegion>:$guard);


  let extraClassDeclaration = [{
    /// Returns the argument types of this method.
    ::mlir::ArrayRef<::mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this method.
    ::mlir::ArrayRef<::mlir::Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Returns the region that is callable.
    ::mlir::Region *getCallableRegion() { return &getBody(); }
    
    /// Returns true if this method has a guard region.
    bool hasGuard() { return !getGuard().empty(); }
    
    /// Returns the guard region if it exists.
    ::mlir::Region &getGuardRegion() { return getGuard()[0]; }
  }];

  let hasCustomAssemblyFormat = 1;
}

def RuleOp : TxnOp<"rule", [
    IsolatedFromAbove, NoTerminator, Symbol
  ]> {
  let summary = "Spontaneous transition rule";
  let description = [{
    Defines a rule that executes spontaneously.
    Can optionally include a guard region that returns a condition.
    If the guard region returns false, the rule will abort.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$prefix
  );
  let regions = (region SizedRegion<1>:$body, VariadicRegion<AnyRegion>:$guard);

  let extraClassDeclaration = [{
    /// Returns true if this rule has a guard region.
    bool hasGuard() { return !getGuard().empty(); }
    
    /// Returns the guard region if it exists.
    ::mlir::Region &getGuardRegion() { return getGuard()[0]; }
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Transaction Control Operations
//===----------------------------------------------------------------------===//

def ReturnOp : TxnOp<"return", [
    Pure, ReturnLike, Terminator
  ]> {
  let summary = "Return from a TXN method";
  let description = [{
    Returns values from TXN methods (value_method, action_method).
    This is the standard terminator for TXN methods.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, /*operands=*/{});
    }]>
  ];
}

def YieldOp : TxnOp<"yield", [
    Pure, ReturnLike, Terminator
  ]> {
  let summary = "Yield values from a region";
  let description = [{
    Yields values from regions (e.g., in txn.if).
    The operation takes a variable number of operands that are yielded.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, /*operands=*/{});
    }]>
  ];
}

def ScheduleOp : TxnOp<"schedule", [
    Terminator
  ]> {
  let summary = "Module schedule - defines module behavior";
  let description = [{
    Defines the schedule of a module - a list of methods and rules to execute.
    This is the terminator of a module and specifies which methods/rules
    can be executed in each cycle.
  }];

  let arguments = (ins 
    SymbolRefArrayAttr:$actions,
    OptionalAttr<TxnConflictMatrixAttr>:$conflict_matrix
  );

  let assemblyFormat = "$actions attr-dict";
}

def IfOp : TxnOp<"if", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface,
      ["getSuccessorRegions", "getRegionInvocationBounds", "getEntrySuccessorRegions"]>,
    NoRegionArguments
  ]> {
  let summary = "Conditional execution";
  let description = [{
    Conditionally executes one of two regions based on a boolean condition.
    Both regions can be empty. If the operation has results, both regions
    must yield values using txn.yield.
  }];

  let arguments = (ins I1:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$thenRegion,
                        AnyRegion:$elseRegion);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Action Operations
//===----------------------------------------------------------------------===//

def AbortOp : TxnOp<"abort", [Terminator]> {
  let summary = "Abort the current transaction";
  let description = [{
    Causes the current transaction to fail and rollback any effects.
    
    An optional reachability condition can be provided as the first operand.
    If present, it must be of type i1 and represents the condition under which
    this abort can be reached during execution.
  }];

  let arguments = (ins Optional<AnyType>:$condition);

  let hasCustomAssemblyFormat = 1;
  
  let builders = [
    OpBuilder<(ins), [{
      $_state.addOperands(::mlir::ValueRange{});
    }]>,
    OpBuilder<(ins "::mlir::Value":$condition), [{
      $_state.addOperands(condition ? ::mlir::ValueRange{condition} : ::mlir::ValueRange{});
    }]>
  ];
}

def CallOp : TxnOp<"call", [AttrSizedOperandSegments]> {
  let summary = "Call a method on a module";
  let description = [{
    Invokes a value or action method on a module instance.
    The callee can be either a method in the current module (@method)
    or a method on an instance (@instance.method).
    
    An optional reachability condition can be provided as the first operand.
    If present, it must be of type i1 and represents the condition under which
    this call can be reached during execution.
  }];

  let arguments = (ins 
    SymbolRefAttr:$callee,  // Changed to SymbolRefAttr to support nested references
    Optional<AnyType>:$condition,  // Optional reachability condition (any type)
    Variadic<AnyType>:$args  // Renamed from operands to avoid overlap
  );
  let results = (outs Variadic<AnyType>:$results);


  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  
  let builders = [
    OpBuilder<(ins "::mlir::SymbolRefAttr":$callee, "::mlir::ValueRange":$args, 
                   "::mlir::TypeRange":$results), [{
      build($_builder, $_state, results, callee, ::mlir::Value(), args);
    }]>,
    OpBuilder<(ins "::mlir::SymbolRefAttr":$callee, "::mlir::Value":$condition,
                   "::mlir::ValueRange":$args, "::mlir::TypeRange":$results), [{
      build($_builder, $_state, results, callee, condition, args);
    }]>
  ];
  
  let extraClassDeclaration = [{
    /// Get the operands to the called method (alias for args)
    ::mlir::OperandRange getOperands() {
      return getArgs();
    }
    
    /// Get the types of the operands to the called method
    ::mlir::TypeRange getOperandTypes() {
      return getArgs().getTypes();
    }
  }];
}

//===----------------------------------------------------------------------===//
// FIRRTL Interface Operations  
//===----------------------------------------------------------------------===//

def FirValueMethodOp : TxnOp<"fir_value_method", [
    HasParent<"PrimitiveOp">,
    Symbol
  ]> {
  let summary = "FIRRTL value method declaration for primitives";
  let description = [{
    Declares a value method interface for hardware primitives.
    This is used to specify the FIRRTL interface that will be generated
    for the primitive's value methods.
  }];
  
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<StrAttr>:$result,         // Result signal postfix (default="OUT")
    OptionalAttr<StrAttr>:$prefix          // Method name prefix for translation
  );
  
  let assemblyFormat = [{
    $sym_name `(` `)` attr-dict `:` $function_type
  }];
}

def FirActionMethodOp : TxnOp<"fir_action_method", [
    HasParent<"PrimitiveOp">,
    Symbol
  ]> {
  let summary = "FIRRTL action method declaration for primitives";
  let description = [{
    Declares an action method interface for hardware primitives.
    This is used to specify the FIRRTL interface that will be generated
    for the primitive's action methods.
  }];
  
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<StrAttr>:$ready,          // Ready signal postfix (default="RDY")
    OptionalAttr<StrAttr>:$enable,         // Enable signal postfix (default="EN")
    OptionalAttr<StrAttr>:$result,         // Result signal postfix (default="OUT")
    OptionalAttr<StrAttr>:$prefix,         // Method name prefix for translation
    OptionalAttr<UnitAttr>:$always_ready,  // Method is always ready (removes ready signal)
    OptionalAttr<UnitAttr>:$always_enable  // Method is always enabled (removes enable signal)
  );
  
  let assemblyFormat = [{
    $sym_name `(` `)` attr-dict `:` $function_type
  }];
}

def ClockByOp : TxnOp<"clock_by", [
    HasParent<"PrimitiveOp">
  ]> {
  let summary = "Specify default clock for primitive";
  let description = [{
    Specifies which clock signal the primitive uses.
    This corresponds to the "Default clock" in BSV embedding.
  }];
  
  let arguments = (ins SymbolRefAttr:$clock);
  
  let assemblyFormat = [{
    $clock attr-dict
  }];
}

def ResetByOp : TxnOp<"reset_by", [
    HasParent<"PrimitiveOp">
  ]> {
  let summary = "Specify default reset for primitive";
  let description = [{
    Specifies which reset signal the primitive uses.
    This corresponds to the "Default reset" in BSV embedding.
  }];
  
  let arguments = (ins SymbolRefAttr:$reset);
  
  let assemblyFormat = [{
    $reset attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Multi-cycle Operations  
//===----------------------------------------------------------------------===//

def FutureOp : TxnOp<"future", [
    SingleBlock, NoTerminator
  ]> {
  let summary = "Encloses multi-cycle actions with launches";
  let description = [{
    The future operation marks a region containing launch operations for
    multi-cycle execution. All operations within a future block are deferred
    and executed according to their launch constraints.
    
    Example:
    ```mlir
    txn.future {
      %done1 = txn.launch {latency=3} {
        // Executes 3 cycles later
      }
      %done2 = txn.launch until %done1 {
        // Executes when %done1 completes
      }
    }
    ```
  }];
  
  let regions = (region SizedRegion<1>:$body);
  
  let assemblyFormat = [{
    $body attr-dict
  }];
}

def LaunchOp : TxnOp<"launch", [
    SingleBlock
  ]> {
  let summary = "Deferred execution with optional dependencies and latency";
  let description = [{
    Launch operations enable deferred execution within multi-cycle actions.
    They can have:
    - Static latency: {latency=n} - Must succeed after n cycles or panic
    - Dynamic dependency: until %cond - Retries until condition is true
    - Both: until %cond {latency=n} - Waits for condition then n more cycles
    
    The operation returns an i1 value indicating completion.
    
    Example:
    ```mlir
    // Static latency
    %done1 = txn.launch {latency=3} {
      txn.call @reg.write(%data) : (i32) -> ()
      txn.yield
    }
    
    // Dynamic dependency
    %done2 = txn.launch until %done1 {
      txn.call @fifo.enqueue(%data) : (i32) -> ()
      txn.yield
    }
    ```
  }];
  
  let arguments = (ins
    Optional<I1>:$condition,      // Optional dependency condition
    OptionalAttr<I32Attr>:$latency  // Optional static latency
  );
  
  let results = (outs I1:$done);  // Completion signal
  
  let regions = (region SizedRegion<1>:$body);
  
  let assemblyFormat = [{
    (`until` $condition^)? 
    (`after` $latency^)?
    $body attr-dict
  }];
  
  let hasVerifier = 1;
}



//===----------------------------------------------------------------------===//
// Function Operations
//===----------------------------------------------------------------------===//

def FuncOp : TxnOp<"func",
    [Symbol, FunctionOpInterface, IsolatedFromAbove]> {
  let summary = "Function within a txn module";
  let description = [{
    Represents a pure function within a transaction module. Unlike value methods,
    functions are syntax sugar that get inlined before lowering. They can take
    arguments and are used for combinational logic that depends on inputs.
    
    Functions are inlined before TxnToFIRRTL and TxnToFunc passes, so they don't
    appear in the final hardware. They're purely for code organization and reuse.
    
    Example:
    ```mlir
    txn.module @ALU {
      // Instead of: txn.value_method @add(%a: i32, %b: i32) -> i32
      txn.func @add(%a: i32, %b: i32) -> i32 {
        %sum = arith.addi %a, %b : i32
        txn.return %sum : i32
      }
      
      // Value method that uses the function
      txn.value_method @compute() -> i32 {
        %a = arith.constant 5 : i32
        %b = arith.constant 3 : i32
        %result = txn.func_call @add(%a, %b) : (i32, i32) -> i32
        txn.return %result : i32
      }
    }
    ```
  }];
  
  let arguments = (ins 
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  
  let regions = (region AnyRegion:$body);
  
  let extraClassDeclaration = [{
    /// Returns the argument types of this function.
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() { 
      return getFunctionType().getInputs(); 
    }

    /// Returns the result types of this function.  
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() { 
      return getFunctionType().getResults(); 
    }
    
    // CallableOpInterface methods
    ::mlir::Region *getCallableRegion();
    ::llvm::ArrayRef<::mlir::Type> getCallableResults();
  }];
  
  let hasCustomAssemblyFormat = 1;
}

def FuncCallOp : TxnOp<"func_call"> {
  let summary = "Call a txn.func";
  let description = [{
    Calls a function defined with txn.func. This operation will be inlined
    during the function inlining pass before any lowering.
    
    Example:
    ```mlir
    %result = txn.func_call @add(%a, %b) : (i32, i32) -> i32
    ```
  }];
  
  let arguments = (ins 
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$operands
  );
  
  let results = (outs Variadic<AnyType>:$results);
  
  let assemblyFormat = [{
    $callee `(` $operands `)` `:` functional-type($operands, $results) attr-dict
  }];
}

#endif // SHARP_DIALECT_TXN_OPS_TD