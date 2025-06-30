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
include "TxnTypes.td"

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

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    /// Returns the entry block of the module.
    Block *getBodyBlock() { return &getBody().front(); }
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
  }];

  let arguments = (ins 
    SymbolNameAttr:$sym_name,
    StrAttr:$type,  // "hw" for synthesizable, "spec" for specification
    TypeAttr:$interface_type
  );
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def InstanceOp : TxnOp<"instance", [Symbol]> {
  let summary = "Module instance declaration";
  let description = [{
    Declares an instance of another module within the current module.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$module_name
  );
  let results = (outs Txn_ModuleType:$result);

  let assemblyFormat = "$sym_name `of` $module_name attr-dict `:` type($result)";
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
    OptionalAttr<StrAttr>:$timing,
    OptionalAttr<StrAttr>:$result,          // Result signal postfix (default="OUT")
    OptionalAttr<StrAttr>:$prefix           // Method name prefix for translation
  );
  let regions = (region AnyRegion:$body);


  let extraClassDeclaration = [{
    /// Returns the argument types of this method.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this method.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Returns the region that is callable.
    Region *getCallableRegion() { return &getBody(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

def ActionMethodOp : TxnOp<"action_method", [
    IsolatedFromAbove, FunctionOpInterface, AutomaticAllocationScope
  ]> {
  let summary = "Action method definition";
  let description = [{
    Defines an action method that may modify state or abort.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    OptionalAttr<StrAttr>:$timing,
    OptionalAttr<StrAttr>:$ready,          // Ready signal postfix (default="RDY")
    OptionalAttr<StrAttr>:$enable,         // Enable signal postfix (default="EN")
    OptionalAttr<StrAttr>:$result,         // Result signal postfix (default="OUT")
    OptionalAttr<StrAttr>:$prefix,         // Method name prefix for translation
    OptionalAttr<UnitAttr>:$always_ready,  // Method is always ready (removes ready signal)
    OptionalAttr<UnitAttr>:$always_enable  // Method is always enabled (removes enable signal)
  );
  let regions = (region AnyRegion:$body);


  let extraClassDeclaration = [{
    /// Returns the argument types of this method.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this method.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Returns the region that is callable.
    Region *getCallableRegion() { return &getBody(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

def RuleOp : TxnOp<"rule", [
    IsolatedFromAbove, NoTerminator
  ]> {
  let summary = "Spontaneous transition rule";
  let description = [{
    Defines a rule that executes spontaneously.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$timing,
    OptionalAttr<StrAttr>:$prefix
  );
  let regions = (region SizedRegion<1>:$body);

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
  }];

  let assemblyFormat = "attr-dict";
}

def CallOp : TxnOp<"call", []> {
  let summary = "Call a method on a module";
  let description = [{
    Invokes a value or action method on a module instance.
    The callee can be either a method in the current module (@method)
    or a method on an instance (@instance.method).
  }];

  let arguments = (ins 
    SymbolRefAttr:$callee,  // Changed to SymbolRefAttr to support nested references
    Variadic<AnyType>:$operands
  );
  let results = (outs Variadic<AnyType>:$results);


  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
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



#endif // SHARP_DIALECT_TXN_OPS_TD