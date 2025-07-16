# Txn to FIRRTL Conversion

## Overview

This document describes the algorithm for converting Sharp Txn modules to synthesizable FIRRTL, extending Koika's approach with methods, explicit conflict matrices, and parametric primitives.

## Core Algorithm

**Implementation**: `lib/Conversion/TxnToFIRRTL/TxnToFIRRTLPass.cpp`

### 1. Module Structure Translation

**Code Position**: Lines 130-143 (`getOrCreatePort` function)

Each `txn.module` becomes a FIRRTL module with:
- Clock and reset ports
- Method interface ports (enable/ready for actions, args/results for values)
- Instance ports for submodules
- Will-fire signals for scheduling

### 2. Will-Fire Logic Generation

The will-fire (WF) logic determines which actions can execute in the current cycle. It strictly follows the [execution model](execution_model.md).

The schedule and the conflict matrix of a module are prepared before the conversion.

For current module, the schedule is [`action0`, `action1`, ..., `actionN`]. For `actionk`, it contains action method calls `action_calls[actionk] = [call0, call1, ..., callM]`. Every action method call corresponds to a action method of an instance, `method[calli] = InstMethod = (instance, method)`. `aborts[actionk]` is the list of abort operations inside the actionk.

`reach_abort[actionk]` is a hardware signal representing the condition under which the actionk is aborted.
```
reach_abort[actionk] = OR(reach(ai, actionk) && reach_abort(method[ai]) for every ai in action_calls[actionk])
                      && OR(reach(aborti, actionk) for every aborti in aborts[actionk])
// reach_abort on a action method also relates to the arguments passed to the method, which is omitted in the expression above but must be considered in the implementation
```

#### Static Mode (Conservative)

**Code Position**: Lines 258-322 (`generateStaticWillFire` function)

```
wf[actionk] = enabled[actionk] && !reach_abort[actionk] && !conflicts_with_earlier[actionk] && !conflict_inside[actionk]
```

**✅ IMPLEMENTED**: Lines 638-700 correctly implement reach_abort calculation and conflict_inside logic for static mode.

**Conflict with earlier actions**:
```
conflicts_with_earlier[actionk] = OR(wf[actioni] && conflict(actioni, actionk) for all actioni before actionk in the schedule)
conflict(actioni, actionk) = C(method[actioni], method[actionk]) || SA(method[actioni], method[actionk]) || SB(method[actionk], method[actioni])
```

**Code Position**: Lines 276-308 (conflict checking loop)

**Conflict inside an action**:
```
conflict_inside[actionk] = OR(conflict(method[ai], method[aj]) && reach(ai, actionk) && reach(aj, actionk) for every ai, aj in action_calls[actionk] and ai is before aj)
// where reach(ai, actionk) is a hardware signal representing the condition under which action method call ai 
// can be reached from the root of the actionk (considering all txn.if conditions along the path)
// reach(ai, actionk) is computed by the reachability analysis (ReachabilityAnalysis) of the current module; it relates to arguments for an action method call, which must be considered in the implementation
```

**NOTE**: The `static` mode uses the conflict matrix of current module to determine "Conflict with earlier actions" and uses the conflict matrices of the called modules to determine "Conflict inside an action".

#### Dynamic Mode (Precise) 

**Code Position**: Lines 324-430 (`generateDynamicWillFire` function)

```
wf[actionk] = enabled[actionk] && !reach_abort[actionk] && 
             AND{for every method call ai in action_calls[actionk]: 
                 NOT(reach(ai, actionk) && conflict_with_earlier(method[ai]))}
```

**✅ IMPLEMENTED**: Lines 702-751 correctly implement reach_abort calculation and method-level conflict tracking for dynamic mode.

This mode generates more complex hardware but allows more concurrency by considering actual execution paths.

**Method Call Tracking**: Track which action methods have been called
```
method_called[methodk] = OR{wf[actioni] && OR(reach(ai, actioni) for ai in action_calls[actioni] and method[ai] == methodk) for every actioni that has been scheduled} 
                  || OR(reach(ai, current_action) for every ai in methodk and ai has been processed in the current action) 
```

**Code Position**: Lines 343-384 (method tracking loop)

**Conflict with earlier actions**:
```
conflict_with_earlier(methodk) = OR(method_called[methodi] && conflict(methodi, methodk) for every methodi in method_called)
```

**Code Position**: Lines 386-417 (conflict checking per method call)

**NOTE**: The `dynamic` mode uses the conflict matrix of the called modules to determine "Conflict with earlier actions", which unifies all the conflict scenarios for the current action ("inside an action" and "with earlier actions" in `static` mode). The conflict matrix of the current module is not used for the current module's will-fire logic.

### 3. Conflict Resolution

Conflicts from the schedule's conflict_matrix:
- **SB/SA**: Enforce ordering constraints
- **C**: Prevent concurrent execution
- **CF**: Allow parallel execution

**Action-level conflict inference**(`ConflictMatrixInference`): If action A calls method M1 and action B calls method M2, and M1 SB/SA/C M2, then A SB/SA/C B

### 4. Method Translation

**Value Methods** → Combinational logic:
```mlir
// txn.value_method @getValue() -> i32
wire [31:0] getValue_OUT = register_value;
```

**Action Methods** → Stateful with handshaking:
```mlir
// txn.action_method @setValue(%arg: i32)
input setValue_EN;
output setValue_RDY;
input [31:0] setValue_arg;
// Body executes when EN & RDY
```


#### Method Attributes for FIRRTL Generation

Sharp provides fine-grained control over FIRRTL signal generation through method attributes:

**Signal Naming Attributes**

```mlir
txn.action_method @doReset() attributes {
  prefix = "rst",      // Signal prefix (default: method name)
  result = "_data",    // Data signal postfix (default: "OUT")
  enable = "_en",      // Enable signal postfix (default: "EN")
  ready = "_rdy"       // Ready signal postfix (default: "RDY")
}
```

Generated signals: `rst_en` (input), `rst_rdy` (output)

**Protocol Optimization Attributes**

```mlir
txn.action_method @alwaysReady() attributes {
  always_ready        // No ready signal needed
}

txn.value_method @pureCombo() attributes {
  always_enable       // No enable signal needed
}
```

These attributes optimize away handshake signals when methods have constant availability.

### 5. Primitive Instantiation

Parametric primitives are instantiated on-demand:
```mlir
// %reg = txn.instance @r of @Register<i32>
// Generates: Register_i32 module with appropriate ports
```


## Multi-Cycle Operations

This is a TODO.

## Implementation Notes

1. **Use CIRCT APIs**: All FIRRTL generation uses builder APIs, not text
2. **Preserve Hierarchy**: Submodule structure maintained in FIRRTL
3. **Type Safety**: Parametric types ensure correct bit widths
4. **Optimization**: Constant propagation and dead code elimination

## Example Pipeline

```bash
# Txn → FIRRTL → Verilog
sharp-opt input.mlir \
  --sharp-infer-conflict-matrix \
  --convert-txn-to-firrtl \
  --lower-firrtl-to-hw \
  --export-verilog -o output.v
```

## Documentation vs Implementation Status

### Issues Resolved

1. **✅ `reach_abort` Logic Implemented**
   - **Location**: Lines 405-520 (`calculateReachAbort` function)
   - **Status**: Fully implemented with path condition tracking and abort propagation
   - **Implementation**: All will-fire modes (static, dynamic) now include reach_abort calculation

2. **✅ `conflict_inside` Implementation**
   - **Location**: Lines 638-700 in static mode will-fire generation
   - **Status**: Implemented with reachability condition checking for static mode
   - **Implementation**: Checks conflicts between method calls within an action using reach conditions

3. **✅ Reachability Conditions Integration**
   - **Location**: Lines 405-520 in `calculateReachAbort`
   - **Status**: Properly integrated with ReachabilityAnalysis pass results
   - **Implementation**: Uses both analysis-provided conditions and fallback to call operands

4. **✅ Most-Dynamic Mode Complete**
   - **Location**: Lines 752-833 in TxnToFIRRTLPass.cpp
   - **Status**: Fully implemented with recursive call tracking via collectAllReachableCalls
   - **Implementation**: Complete direct/indirect method conflict analysis with reach_abort logic

### Remaining Issues

1. **Multi-Cycle Support**
   - **Location**: Line 186 notes "This is a TODO"
   - **Issue**: Launch operations for multi-cycle execution not implemented
   - **Impact**: Cannot generate FIRRTL for designs with launch operations

2. **Primitive Conflict Matrix**
   - **Issue**: Primitive methods assumed to have hardcoded conflict behavior
   - **Impact**: May not accurately model all primitive conflicts

### Implementation Notes

- The implementation uses a helper function `walkWithPathConditions` to track control flow paths
- Abort conditions from both explicit `txn.abort` ops and method calls are considered
- Test suite needs updating to match new output format with temporary variables