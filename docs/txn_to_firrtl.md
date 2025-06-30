# Sharp Txn to FIRRTL Conversion Algorithm

## Overview

This document describes the algorithm for converting Sharp Txn modules to FIRRTL, extending the Koika approach to support transaction-based hardware description with methods and conflict matrices.

**Important Note**: Throughout this document, FIRRTL code examples are shown in plain text for illustration purposes. However, the actual implementation MUST use CIRCT's FIRRTL dialect APIs (e.g., `builder.create<FModuleOp>`, `builder.create<ConnectOp>`, etc.) rather than generating text. The plain FIRRTL examples help understand the intended structure, but all generation should be done programmatically through the CIRCT APIs.

## Background

### Koika Translation Model
Koika translates rule-based hardware descriptions to RTL by:
1. Generating combinational logic for each rule
2. Creating will-fire (wf) logic that determines when rules can execute
3. Managing register read/write conflicts
4. Ensuring single-cycle atomic execution

### Sharp Txn Extensions
Sharp extends Koika with:
- **Methods**: Reusable action/value methods that can be called by rules
- **Conflict Matrix**: Explicit specification of action relationships (SB, SA, C, CF)
- **Module Hierarchy**: Support for instantiating submodules
- **Primitive Interface**: Bridging between txn methods and FIRRTL ports

## Method Attributes for FIRRTL Generation

Sharp supports several attributes on methods to control FIRRTL signal naming and protocol generation:

### Signal Naming Attributes

1. **`prefix`**: Custom prefix for method name in FIRRTL (default: method name)
2. **`result`**: Postfix for result/data signals (default: "OUT")
3. **`ready`**: Postfix for ready signals (default: "RDY") - action methods only
4. **`enable`**: Postfix for enable signals (default: "EN") - action methods only

### Protocol Optimization Attributes

1. **`always_ready`**: Removes ready signal generation when method is always ready
2. **`always_enable`**: Removes enable signal generation when method is always enabled

### Example Usage

```mlir
txn.module @Example {
  // Value method with custom signal names
  txn.value_method @getValue() -> i32 
    attributes {prefix = "get", result = "_data", always_ready} {
    %val = txn.call @reg::@read() : () -> i32
    txn.return %val : i32
  }
  
  // Action method with protocol optimization
  txn.action_method @reset() -> () 
    attributes {prefix = "do", ready = "_ready", enable = "_en", always_ready} {
    txn.call @reg::@write(%c0) : (i32) -> ()
    txn.return
  }
}
```

Generated FIRRTL ports:
- `get_data` (instead of `getValue_OUT`)
- `do_en` (instead of `reset_EN`)
- No `do_ready` signal due to `always_ready`

## Translation Strategy

### 1. Bottom-Up Module Processing

The conversion must process modules in dependency order:
```
1. Identify module dependency graph
2. Topologically sort modules (leaves first)
3. Process each module:
   a. Primitives: Link to pre-defined FIRRTL implementations
   b. Regular modules: Generate FIRRTL from txn description
```

### 2. Module Structure Translation

For each txn module, generate a FIRRTL module with:

#### Ports
Using CIRCT FIRRTL dialect APIs:
```cpp
// Create FIRRTL module
SmallVector<PortInfo> ports;
// Clock and reset
ports.push_back({builder.getStringAttr("clock"), 
                 ClockType::get(ctx), Direction::In});
ports.push_back({builder.getStringAttr("reset"), 
                 UIntType::get(ctx, 1), Direction::In});

// For each value method
auto prefix = method.getPrefix().value_or(method.getSymName());
auto resultPostfix = method.getResult().value_or("OUT");
ports.push_back({builder.getStringAttr(prefix + resultPostfix), 
                 convertType(returnType), Direction::Out});

// For each action method
auto prefix = method.getPrefix().value_or(method.getSymName());
auto resultPostfix = method.getResult().value_or("OUT");
auto enablePostfix = method.getEnable().value_or("EN");
auto readyPostfix = method.getReady().value_or("RDY");

// Data port if method has arguments
if (!argTypes.empty()) {
  ports.push_back({builder.getStringAttr(prefix + resultPostfix), 
                   convertType(argType), Direction::In});
}

// Enable port (unless always_enable)
if (!method.getAlwaysEnable()) {
  ports.push_back({builder.getStringAttr(prefix + enablePostfix), 
                   UIntType::get(ctx, 1), Direction::In});
}

// Ready port (unless always_ready)
if (!method.getAlwaysReady()) {
  ports.push_back({builder.getStringAttr(prefix + readyPostfix), 
                   UIntType::get(ctx, 1), Direction::Out});
}

auto firrtlModule = builder.create<FModuleOp>(loc, moduleName, ports);
```

#### Internal Structure
1. **Submodule Instances**: Direct mapping of txn instances
2. **Rule Logic**: Combinational logic for each rule
3. **Method Logic**: Combinational logic for each method
4. **Scheduling Logic**: Will-fire calculation and conflict resolution

### 3. Will-Fire (WF) Logic Generation

The will-fire logic determines whether an action (rule or method) can execute in the current cycle.

#### Basic WF Calculation
```
wf[action] = enabled[action] && !conflicts_with_earlier[action] && !conflict_inside[action]
```

#### Conflict Detection

For each action pair (a1, a2) where a1 is scheduled before a2:
```
conflicts(a1, a2) = 
  (CM[a1,a2] == C) ||                     // Explicit conflict
  (CM[a1,a2] == SA && wf[a1])             // a1 must happen after a2, but a1 has already happened
```

**Method Call Tracking**: Track which action methods have been called
```
method_called[m] = OR(wf[a] for all a that calls m)
```

**Conflict with earlier actions**:
```
conflicts_with_earlier[a] = OR(wf[a1] && conflict(a1, a) for all a1 that is scheduled before a)
```

**Conflict inside an action**:
```
conflict_inside[a] = OR(conflict(m1, m2) && reach(m1) && reach(m2) for every m1, m2 in a, m1 is before m2)
// where reach(m) is a hardware signal representing the condition under which method call m 
// can be reached from the root of the action (considering all `txn.if` conditions along the path)
// Note: reach(m1) && reach(m2) evaluates to true in hardware when both methods can be reached 
// under some execution path
```


### 4. Method Implementation


#### Value Methods
Value methods are purely combinational:

**FIRRTL representation (for illustration):**
```firrtl
// For txn value method returning data
node <method>_result = <combinational_logic>
connect <method>_data, <method>_result
```

**Implementation using CIRCT APIs:**
```cpp
// Get method attributes
auto prefix = method.getPrefix().value_or(method.getSymName());
auto resultPostfix = method.getResult().value_or("OUT");

// Inside FIRRTL module body
auto methodResult = builder.create<NodeOp>(loc, 
    resultType, builder.getStringAttr(prefix + "_result"), 
    combinationalLogic);

// Connect to output port
auto methodDataPort = getPort(prefix + resultPostfix);
builder.create<ConnectOp>(loc, methodDataPort, methodResult);
```

#### Action Methods
Action methods have side effects and conflict resolution:

**FIRRTL representation (for illustration):**
```firrtl
// Will-fire for action method
node <method>_wf = <method>_enable and <conflict_free_logic>

// Execute side effects when wf is true
when <method>_wf :
  <side_effect_logic>
  
// Ready signal indicates method can be called
connect <method>_ready, <conflict_free_logic>
```

**Implementation using CIRCT APIs:**
```cpp
// Get method attributes
auto prefix = method.getPrefix().value_or(method.getSymName());
auto enablePostfix = method.getEnable().value_or("EN");
auto readyPostfix = method.getReady().value_or("RDY");

// Create will-fire logic
Value methodEnable;
if (method.getAlwaysEnable()) {
  // Method is always enabled - use constant true
  methodEnable = builder.create<ConstantOp>(loc, 
      UIntType::get(ctx, 1), APInt(1, 1));
} else {
  // Use the enable input port
  methodEnable = getPort(prefix + enablePostfix);
}

auto methodWF = builder.create<NodeOp>(loc,
    UIntType::get(ctx, 1), builder.getStringAttr(prefix + "_wf"),
    builder.create<AndPrimOp>(loc, methodEnable, conflictFreeLogic));

// Execute side effects conditionally
builder.create<WhenOp>(loc, methodWF, /*hasElse=*/false, [&]() {
    // Build side effect logic inside when block
    generateSideEffects();
});

// Connect ready signal (if not always_ready)
if (!method.getAlwaysReady()) {
  builder.create<ConnectOp>(loc, getPort(prefix + readyPostfix), 
                           conflictFreeLogic);
}
```

### 5. Rule Implementation

Each rule generates:

**FIRRTL representation (for illustration):**
```firrtl
// Rule condition evaluation
node <rule>_condition = <guard_logic>

// Will-fire calculation
node <rule>_wf = <rule>_condition and <conflict_free_logic>

// Rule body execution
when <rule>_wf :
  <rule_body_logic>
```

**Implementation using CIRCT APIs:**
```cpp
// Evaluate rule condition
auto ruleCondition = builder.create<NodeOp>(loc,
    UIntType::get(ctx, 1), builder.getStringAttr(ruleName + "_condition"),
    generateGuardLogic(rule));

// Calculate will-fire
auto ruleWF = builder.create<NodeOp>(loc,
    UIntType::get(ctx, 1), builder.getStringAttr(ruleName + "_wf"),
    builder.create<AndPrimOp>(loc, ruleCondition, conflictFreeLogic));

// Execute rule body
builder.create<WhenOp>(loc, ruleWF, /*hasElse=*/false, [&]() {
    generateRuleBody(rule);
});
```

### 6. Scheduling and Conflict Resolution

The scheduler enforces the conflict matrix relationships:

#### Sequential Ordering (SB/SA)
**FIRRTL representation:**
```firrtl
// For CM[a1,a2] = SB (a1 before a2)
node a2_blocked_by_a1 = a1_wf
node a2_wf = a2_enabled and not(a2_blocked_by_a1)
```

**Using CIRCT APIs:**
```cpp
auto a2Blocked = builder.create<NodeOp>(loc, UIntType::get(ctx, 1),
    "a2_blocked_by_a1", a1WF);
auto a2WF = builder.create<AndPrimOp>(loc, a2Enabled,
    builder.create<NotPrimOp>(loc, a2Blocked));
```

#### Mutual Exclusion (C)
**FIRRTL representation:**
```firrtl
// For CM[a1,a2] = C (conflict)
node conflict_12 = a1_wf
node a2_wf = a2_enabled and not(conflict_12)
```

**Using CIRCT APIs:**
```cpp
auto conflict12 = builder.create<NodeOp>(loc, UIntType::get(ctx, 1),
    "conflict_12", a1WF);
auto a2WF = builder.create<AndPrimOp>(loc, a2Enabled,
    builder.create<NotPrimOp>(loc, conflict12));
```

#### Conflict-Free (CF)
```firrtl
// For CM[a1,a2] = CF
// No additional constraints
```

### 7. Primitive Bridging

Primitives bridge txn methods to FIRRTL ports. While primitives are pre-defined, the instantiation and connection use CIRCT APIs:

**FIRRTL representation of a Register primitive:**
```firrtl
// Register primitive example
module Register_impl :
  input clock : Clock
  input reset : UInt<1>
  output read_data : UInt<32>
  input read_enable : UInt<1>
  input write_data : UInt<32>
  input write_enable : UInt<1>
  
  reg register : UInt<32>, clock with :
    reset => (reset, UInt<32>(0))
    
  // Read is always available
  connect read_data, register
  
  // Write updates register when enabled
  when write_enable :
    connect register, write_data
```

**Instantiation using CIRCT APIs:**
```cpp
// Create instance of primitive
auto primInst = builder.create<InstanceOp>(loc, 
    builder.getStringAttr(instanceName),
    builder.getStringAttr("Register_impl"),
    /*portDirections=*/..., /*portTypes=*/...);

// Connect clock and reset
builder.create<ConnectOp>(loc, primInst.getResult(0), clockSignal);
builder.create<ConnectOp>(loc, primInst.getResult(1), resetSignal);
```

## Implementation Details

### Pass Structure

1. **Analysis Phase**
   - Build module dependency graph
   - Complete conflict matrices (using inference pass)
   - Verify synthesizability (using pre-synthesis check)
   - Validate method attributes (using method attribute validation)

2. **Conversion Phase**
   - Process modules bottom-up
   - Generate FIRRTL operations
   - Build will-fire logic
   - Connect submodules

3. **Optimization Phase**
   - Simplify will-fire logic
   - Remove redundant conflict checks
   - Optimize method call tracking

### Data Structures

```cpp
struct ConversionContext {
  // Track generated FIRRTL ops
  DenseMap<Value, Value> txnToFirrtl;
  
  // Track will-fire signals
  DenseMap<StringRef, Value> willFireSignals;
  
  // Track method calls
  DenseMap<StringRef, SmallVector<StringRef>> methodCallers;
  
  // Conflict matrix for current module
  ConflictMatrix conflictMatrix;
  
  // Reachability conditions for method calls
  DenseMap<MethodCallOp, Value> reachabilityConditions;
};
```

### Key Algorithms

#### Reachability Analysis

Before calculating conflict_inside, we need to compute reachability conditions for all method calls within actions. This analysis determines under what hardware conditions each method call can be reached.

```cpp
// Analysis pass to compute reachability conditions
class ReachabilityAnalysis {
  // Compute reachability condition for each method call in an action
  void analyzeAction(StringRef action, ConversionContext &ctx) {
    auto actionOp = getActionOp(action);
    
    // Track current path condition
    Value pathCondition = builder.create<ConstantOp>(loc, 
        UIntType::get(ctx, 1), APInt(1, 1)); // Start with true
    
    // Walk the action body tracking path conditions
    actionOp.walk([&](Operation *op) {
      if (auto ifOp = dyn_cast<txn::IfOp>(op)) {
        // Save current path condition
        auto savedCondition = pathCondition;
        
        // Process then branch
        Value thenCondition = builder.create<AndPrimOp>(loc, 
            pathCondition, ifOp.getCondition());
        processRegion(ifOp.getThenRegion(), thenCondition, ctx);
        
        // Process else branch if exists
        if (!ifOp.getElseRegion().empty()) {
          Value elseCondition = builder.create<AndPrimOp>(loc,
              pathCondition, 
              builder.create<NotPrimOp>(loc, ifOp.getCondition()));
          processRegion(ifOp.getElseRegion(), elseCondition, ctx);
        }
        
        // Don't process nested operations here
        return WalkResult::skip();
      }
      
      if (auto callOp = dyn_cast<txn::CallOp>(op)) {
        // Record reachability condition for this method call
        ctx.reachabilityConditions[callOp] = pathCondition;
      }
      
      return WalkResult::advance();
    });
  }
  
  void processRegion(Region &region, Value pathCondition, 
                     ConversionContext &ctx) {
    // Process operations in region with given path condition
    for (auto &op : region.getOps()) {
      if (auto ifOp = dyn_cast<txn::IfOp>(&op)) {
        // Nested if - update path condition
        Value thenCondition = builder.create<AndPrimOp>(loc,
            pathCondition, ifOp.getCondition());
        processRegion(ifOp.getThenRegion(), thenCondition, ctx);
        
        if (!ifOp.getElseRegion().empty()) {
          Value elseCondition = builder.create<AndPrimOp>(loc,
              pathCondition,
              builder.create<NotPrimOp>(loc, ifOp.getCondition()));
          processRegion(ifOp.getElseRegion(), elseCondition, ctx);
        }
      } else if (auto callOp = dyn_cast<txn::CallOp>(&op)) {
        ctx.reachabilityConditions[callOp] = pathCondition;
      }
    }
  }
};
```

#### Example: Reachability Analysis

Consider this action with conditional method calls:
```mlir
txn.rule @example {
  %cond1 = arith.cmpi eq, %x, %c0 : i32
  txn.if %cond1 {
    txn.call @inst1::@method1() : () -> ()  // reach(m1) = %cond1
  } else {
    %cond2 = arith.cmpi gt, %y, %c10 : i32
    txn.if %cond2 {
      txn.call @inst1::@method2() : () -> ()  // reach(m2) = !%cond1 && %cond2
      txn.call @inst2::@method3() : () -> ()  // reach(m3) = !%cond1 && %cond2
    }
  }
  txn.return
}
```

The reachability analysis would compute:
- `reach(method1) = %cond1`
- `reach(method2) = and(not(%cond1), %cond2)`
- `reach(method3) = and(not(%cond1), %cond2)`

If method2 and method3 conflict, then:
- `conflict_inside = and(reach(method2), reach(method3))`
- `conflict_inside = and(not(%cond1), %cond2)`

This creates a hardware signal that is true when both conflicting methods would execute.

#### Will-Fire Calculation
```cpp
Value calculateWillFire(StringRef action, ConversionContext &ctx) {
  // Base condition: action is enabled
  Value wf = ctx.enabled[action];
  
  // Check conflicts with earlier actions
  for (auto earlier : ctx.schedule) {
    if (earlier == action) break;
    
    auto conflict = checkConflict(earlier, action, ctx);
    if (conflict) {
      wf = builder.create<AndPrimOp>(loc, wf, 
        builder.create<NotPrimOp>(loc, ctx.willFireSignals[earlier]));
    }
  }

  // Check for conflicts inside the action
  auto conflictInside = calculateConflictInside(action, ctx);
  if (conflictInside) {
    wf = builder.create<AndPrimOp>(loc, wf,
        builder.create<NotPrimOp>(loc, conflictInside));
  }
  
  return wf;
}

// Calculate if there are conflicts within an action
Value calculateConflictInside(StringRef action, ConversionContext &ctx) {
  // First run reachability analysis for this action
  ReachabilityAnalysis reachAnalysis;
  reachAnalysis.analyzeAction(action, ctx);
  
  // Get all method calls in this action
  auto methodCalls = collectMethodCalls(action);
  
  Value conflictInside = builder.create<ConstantOp>(loc, 
      UIntType::get(ctx, 1), APInt(1, 0)); // Start with false
  
  // Check each pair of method calls
  for (size_t i = 0; i < methodCalls.size(); ++i) {
    for (size_t j = i + 1; j < methodCalls.size(); ++j) {
      auto m1 = methodCalls[i];
      auto m2 = methodCalls[j];
      
      // Get reachability conditions for both method calls
      Value reach1 = ctx.reachabilityConditions[m1];
      Value reach2 = ctx.reachabilityConditions[m2];
      
      // Check if the methods conflict
      if (checkMethodConflict(m1.getInstance(), m1.getMethod(),
                             m2.getInstance(), m2.getMethod(), ctx)) {
        // Both methods can be reached simultaneously if their
        // reachability conditions can both be true
        Value bothReachable = builder.create<AndPrimOp>(loc, reach1, reach2);
        
        // Add to conflict_inside calculation
        conflictInside = builder.create<OrPrimOp>(loc, 
            conflictInside, bothReachable);
      }
    }
  }
  
  return conflictInside;
}
```

#### Conflict Checking
```cpp
bool checkConflict(StringRef a1, StringRef a2, ConversionContext &ctx) {
  // Direct conflict matrix lookup
  auto rel = ctx.conflictMatrix.get(a1, a2);
  
  switch (rel) {
    case ConflictRelation::C:  // Conflict
      return true;
      
    case ConflictRelation::SB: // a1 before a2
      return ctx.willFireSignals[a1] != nullptr;
      
    case ConflictRelation::SA: // a1 after a2 (shouldn't happen)
      assert(false && "Invalid schedule order");
      
    case ConflictRelation::CF: // Conflict-free
      break;
  }
  
  return false;
}
```

## Combinational Loop Detection

### Overview
The translation must detect and prevent combinational loops that would create invalid hardware. Combinational loops can occur through:
1. Direct method call cycles
2. Cycles through value method dependencies
3. Cycles through combinational primitives (e.g., Wire)

### Detection Algorithm
```cpp
class CombinationalLoopDetector {
  // Build dependency graph of combinational paths
  void buildDependencyGraph(ModuleOp module) {
    // For each value method, track what it calls
    // For each rule/action method, track combinational paths
    // For each instance, track input-output dependencies
  }
  
  // Detect cycles using DFS
  bool hasCycle() {
    enum State { Unvisited, Visiting, Visited };
    DenseMap<StringRef, State> states;
    
    for (auto node : graph.nodes()) {
      if (states[node] == Unvisited) {
        if (dfsHasCycle(node, states)) return true;
      }
    }
    return false;
  }
  
  // Report cycle path for diagnostics
  SmallVector<StringRef> getCyclePath() {
    // Return the path that forms a cycle
  }
};
```

### Integration with Conversion
- Run combinational loop detection as a pre-pass
- Emit clear error messages showing the cycle path
- Example error: "Combinational loop detected: getValue -> compute -> getValue"

## Pass Structure

### Overall Conversion Pipeline

```cpp
// Main conversion pipeline
void TxnToFIRRTLPipeline::runOnModule(ModuleOp module) {
  // 1. Pre-synthesis checking
  if (failed(runPreSynthesisCheck(module)))
    return signalPassFailure();
    
  // 2. Conflict matrix inference
  if (failed(runConflictMatrixInference(module)))
    return signalPassFailure();
    
  // 3. Method attribute validation
  if (failed(runMethodAttributeValidation(module)))
    return signalPassFailure();
    
  // 4. Combinational loop detection
  CombinationalLoopDetector detector;
  if (detector.detectLoops(module)) {
    module.emitError("Combinational loops detected");
    return signalPassFailure();
  }
  
  // 5. Module dependency analysis
  auto sortedModules = topologicalSort(module);
  
  // 6. Bottom-up conversion
  for (auto txnModule : sortedModules) {
    convertModule(txnModule);
  }
}
```

### Individual Passes

1. **TxnToFIRRTLConversionPass**
   - Main conversion pass
   - Depends on: PreSynthesisCheck, ConflictMatrixInference
   - Includes reachability analysis for conflict_inside calculation
   - Produces: FIRRTL modules

2. **CombinationalLoopDetectionPass** 
   - Analyzes txn modules for combinational cycles
   - Can run standalone or as part of conversion
   - Produces: Diagnostic information

3. **TxnPrimitiveLoadingPass**
   - Loads FIRRTL implementations of primitives
   - Caches primitive definitions
   - Validates primitive interfaces

4. **ReachabilityAnalysisPass** (optional standalone)
   - Computes reachability conditions for method calls
   - Analyzes control flow through txn.if operations
   - Can be integrated into conversion pass or run separately

5. **MethodAttributeValidationPass**
   - Validates method attributes for FIRRTL translation
   - Checks for name conflicts between modules and methods (considering prefix)
   - Validates `always_ready` usage (method must actually be always ready)
   - Validates `always_enable` usage (method's callers must be always enabled)
   - Ensures signal name uniqueness after applying prefixes/postfixes

### Method Attribute Validation Algorithm

```cpp
class MethodAttributeValidator {
  void validateModule(ModuleOp module) {
    // 1. Check name conflicts
    StringSet<> usedNames;
    module.walk([&](Operation *op) {
      if (auto mod = dyn_cast<txn::ModuleOp>(op)) {
        if (!usedNames.insert(mod.getSymName()).second) {
          op->emitError("Duplicate name: ") << mod.getSymName();
        }
      }
      if (auto method = dyn_cast<txn::ValueMethodOp>(op)) {
        auto name = method.getPrefix().value_or(method.getSymName());
        if (!usedNames.insert(name).second) {
          op->emitError("Method prefix conflicts with existing name: ") << name;
        }
      }
      // Similar for ActionMethodOp
    });
    
    // 2. Validate always_ready
    module.walk([&](Operation *op) {
      if (auto method = dyn_cast<txn::ActionMethodOp>(op)) {
        if (method.getAlwaysReady()) {
          // Check if method can actually be always ready
          if (!isAlwaysReady(method)) {
            op->emitError("Method marked always_ready but has conditional availability");
          }
        }
      }
    });
    
    // 3. Validate always_enable
    module.walk([&](Operation *op) {
      if (auto method = dyn_cast<txn::ActionMethodOp>(op)) {
        if (method.getAlwaysEnable()) {
          // Check if all callers are unconditionally enabled
          if (!areAllCallersAlwaysEnabled(method)) {
            op->emitError("Method marked always_enable but has conditional callers");
          }
        }
      }
    });
  }
  
  bool isAlwaysReady(ActionMethodOp method) {
    // A method is always ready if:
    // 1. It has no conflict with any other action, OR
    // 2. All conflicting actions are never enabled
    // This requires analyzing the conflict matrix and action conditions
  }
  
  bool areAllCallersAlwaysEnabled(ActionMethodOp method) {
    // Check all call sites to this method
    // Return true only if all calls are unconditional (not inside txn.if)
  }
};
```

### Pass Registration
```cpp
namespace mlir::sharp {
void registerTxnToFIRRTLPasses() {
  PassRegistration<TxnToFIRRTLConversionPass>();
  PassRegistration<CombinationalLoopDetectionPass>();
  PassRegistration<TxnPrimitiveLoadingPass>();
  PassRegistration<ReachabilityAnalysisPass>(); // If implemented as standalone
  PassRegistration<MethodAttributeValidationPass>();
}
}
```

## Example Translation

### Input Txn Module
```mlir
txn.module @Counter {
  %reg = txn.instance @count of @Register : !txn.module<"Register">
  
  txn.rule @increment {
    %val = txn.call @count::@read() : () -> i32
    %inc = arith.addi %val, %c1 : i32
    txn.call @count::@write(%inc) : (i32) -> ()
    txn.return
  }
  
  txn.value_method @getValue() -> i32 {
    %val = txn.call @count::@read() : () -> i32
    txn.return %val : i32
  }
  
  txn.action_method @reset() -> () {
    txn.call @count::@write(%c0) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@increment, @getValue, @reset] {
    conflict_matrix = {
      "increment,reset" = 2 : i32,  // C
      "getValue,reset" = 0 : i32    // SB
    }
  }
}
```

### Output FIRRTL Module

**Note**: The following FIRRTL code is shown in plain text format for illustration purposes only. The actual implementation MUST generate this structure using CIRCT FIRRTL dialect APIs (e.g., `builder.create<FModuleOp>`, `builder.create<InstanceOp>`, etc.) as shown in the earlier sections of this document.

```firrtl
module Counter :
  input clock : Clock
  input reset : UInt<1>
  
  ; Value method ports
  output getValue_data : UInt<32>
  input getValue_enable : UInt<1>
  
  ; Action method ports  
  input reset_enable : UInt<1>
  output reset_ready : UInt<1>
  
  ; Submodule instance
  inst count of Register_impl
  connect count.clock, clock
  connect count.reset, reset
  
  ; Rule: increment
  node increment_condition = UInt<1>(1)  ; always enabled
  
  ; Will-fire: increment (no conflicts with earlier actions)
  node increment_wf = increment_condition
  
  ; Rule body
  connect count.read_enable, increment_wf
  node inc_value = add(count.read_data, UInt<32>(1))
  connect count.write_enable, increment_wf
  connect count.write_data, inc_value
  
  ; Method: getValue  
  connect getValue_data, count.read_data
  
  ; Method: reset
  ; Will-fire: reset (check conflicts)
  node reset_conflicts_increment = and(CM_increment_reset_C, increment_wf)
  node reset_wf = and(reset_enable, not(reset_conflicts_increment))
  
  connect count.write_enable, reset_wf
  connect count.write_data, UInt<32>(0)
  connect reset_ready, not(reset_conflicts_increment)
```

## Error Handling

### Common Error Scenarios

1. **Cyclic Dependencies**: Modules with circular instantiation
   - Detection: During dependency analysis
   - Response: Emit error with cycle path

2. **Unresolved Method Calls**: Calls to non-existent methods
   - Detection: During method resolution
   - Response: Emit error with location

3. **Type Mismatches**: Incompatible types in method calls
   - Detection: During type checking
   - Response: Emit error with expected/actual types

4. **Scheduling Violations**: Invalid conflict matrix relationships
   - Detection: During schedule analysis
   - Response: Emit error with conflict details

## Testing Strategy

1. **Unit Tests**: Individual algorithm components
2. **Integration Tests**: Full module translations
3. **Regression Tests**: Known corner cases
4. **Validation**: Compare with reference implementations

## Future Extensions

1. **Multi-cycle Support**: Extend WF logic for timing attributes
2. **Optimization**: Minimize generated logic
3. **Debug Support**: Generate assertions and coverage points
4. **Performance**: Parallel module processing