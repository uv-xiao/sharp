# Sharp Txn to FIRRTL Conversion Algorithm

## Overview

This document provides a comprehensive description of the algorithm for converting Sharp Txn modules to FIRRTL. The conversion extends the Koika approach to support transaction-based hardware description with methods, conflict matrices, and parametric primitives.

The Sharp Txn dialect provides a high-level abstraction for hardware design with explicit conflict management, while FIRRTL serves as the low-level RTL representation. This conversion bridges the gap between these two levels, ensuring that the transaction semantics are correctly translated into synthesizable hardware.

**Important Note**: Throughout this document, FIRRTL code examples are shown in plain text for illustration purposes. However, the actual implementation MUST use CIRCT's FIRRTL dialect APIs (e.g., `builder.create<FModuleOp>`, `builder.create<ConnectOp>`, etc.) rather than generating text. The plain FIRRTL examples help understand the intended structure, but all generation should be done programmatically through the CIRCT APIs.

## Background and Motivation

### Koika Translation Model
Koika pioneered the approach of translating rule-based hardware descriptions to RTL with these key principles:
1. **Atomic Rule Execution**: Each rule executes atomically within a single cycle
2. **Will-Fire Logic**: Dynamic determination of which rules can execute
3. **Conflict Resolution**: Managing register read/write conflicts automatically
4. **One-Rule-At-A-Time Semantics**: Conceptually, rules execute sequentially

### Sharp Txn Extensions
Sharp extends Koika's model with several important features:
- **Methods**: Reusable action/value methods that encapsulate behavior
- **Explicit Conflict Matrix**: Declarative specification of action relationships (SB, SA, C, CF)
- **Module Hierarchy**: First-class support for instantiating submodules
- **Parametric Primitives**: Type-safe primitives like `Register<T>`, `Wire<T>`
- **Conditional Execution**: Support for control flow within actions
- **Method Attributes**: Fine-grained control over FIRRTL signal generation

These extensions provide more modularity and explicit control over hardware behavior while maintaining the benefits of atomic execution and automatic conflict resolution.

## Core Concepts

### Conflict Matrix Relationships
The conflict matrix defines relationships between actions:
- **SB (Sequential Before, value=0)**: Action A must complete before action B starts
- **SA (Sequential After, value=1)**: Action A must start after action B completes  
- **C (Conflict, value=2)**: Actions A and B cannot execute in the same cycle
- **CF (Conflict-Free, value=3)**: Actions A and B can execute concurrently

### Method Types
1. **Value Methods**: Pure combinational functions with no side effects
2. **Action Methods**: Methods that can modify state or have side effects
3. **Rules**: Top-level actions that execute based on conditions

### Primitives
Primitives bridge the gap between Txn methods and FIRRTL ports:
- Provide hardware implementations for basic components
- Support parametric types for reusability
- Examples: `Register<i32>`, `Wire<i8>`
- Automatically instantiated when referenced but not defined

## Method Attributes for FIRRTL Generation

Sharp provides fine-grained control over FIRRTL signal generation through method attributes:

### Signal Naming Attributes

```mlir
txn.action_method @doReset() attributes {
  prefix = "rst",      // Signal prefix (default: method name)
  result = "_data",    // Data signal postfix (default: "OUT")
  enable = "_en",      // Enable signal postfix (default: "EN")
  ready = "_rdy"       // Ready signal postfix (default: "RDY")
}
```

Generated signals: `rst_en` (input), `rst_rdy` (output)

### Protocol Optimization Attributes

```mlir
txn.action_method @alwaysReady() attributes {
  always_ready        // No ready signal needed
}

txn.value_method @pureCombo() attributes {
  always_enable       // No enable signal needed
}
```

These attributes optimize away handshake signals when methods have constant availability.

## Translation Process

### 1. Module Dependency Analysis

The conversion begins by analyzing module dependencies to ensure correct processing order:

```cpp
// Build dependency graph from instance relationships
DenseMap<StringRef, SmallVector<StringRef>> dependencies;
module.walk([&](InstanceOp inst) {
  dependencies[currentModule].push_back(inst.getModuleName());
});

// Topological sort ensures dependencies are processed first
SmallVector<ModuleOp> sortedModules = topologicalSort(dependencies);

// Identify top-level module (not instantiated by others)
StringRef topModule = findTopLevelModule(sortedModules);
```

Key insights:
- Primitives and leaf modules are processed first
- The top-level module becomes the FIRRTL circuit name
- Circular dependencies are detected and reported as errors
- Missing modules trigger automatic primitive construction

### 2. FIRRTL Module Structure Generation

Each Txn module generates a corresponding FIRRTL module with carefully constructed ports:

```cpp
SmallVector<PortInfo> ports;

// Standard ports
ports.push_back({builder.getStringAttr("clock"), 
                 ClockType::get(ctx), Direction::In});
ports.push_back({builder.getStringAttr("reset"), 
                 UIntType::get(ctx, 1), Direction::In});

// Value method ports (output only)
for (auto method : module.getOps<ValueMethodOp>()) {
  auto prefix = method.getPrefix().value_or(method.getSymName());
  auto resultPostfix = method.getResult().value_or("OUT");
  ports.push_back({
    builder.getStringAttr(prefix + resultPostfix), 
    convertType(method.getResultType()), 
    Direction::Out
  });
}

// Action method ports (enable input, ready output, data as needed)
for (auto method : module.getOps<ActionMethodOp>()) {
  auto prefix = method.getPrefix().value_or(method.getSymName());
  
  // Input data ports for method arguments
  for (auto [idx, argType] : enumerate(method.getArgumentTypes())) {
    ports.push_back({
      builder.getStringAttr(prefix + "OUT_arg" + std::to_string(idx)),
      convertType(argType),
      Direction::In
    });
  }
  
  // Enable input (unless always_enable)
  if (!method.getAlwaysEnable()) {
    ports.push_back({
      builder.getStringAttr(prefix + method.getEnable().value_or("EN")),
      UIntType::get(ctx, 1),
      Direction::In
    });
  }
  
  // Ready output (unless always_ready)
  if (!method.getAlwaysReady()) {
    ports.push_back({
      builder.getStringAttr(prefix + method.getReady().value_or("RDY")),
      UIntType::get(ctx, 1),
      Direction::Out
    });
  }
}
```

### 3. Will-Fire Logic - The Heart of Conflict Resolution

The will-fire (WF) logic determines which actions can execute in the current cycle. This is the most critical part of the translation as it enforces the transaction semantics.

#### Static Mode (Conservative)
In static mode, conflicts are determined pessimistically based on what methods *might* be called:

```
wf[action] = enabled[action] && !conflicts_with_earlier[action] && !conflict_inside[action]
```

Key algorithm:
1. **Action-level conflict inference**: If action A calls method M1 and action B calls method M2, and M1 conflicts with M2, then A conflicts with B
2. **Schedule enforcement**: Earlier actions in the schedule prevent later conflicting actions
3. **Internal conflict detection**: Actions that might call conflicting methods internally are prevented

**Conflict with earlier actions**:
```
conflicts_with_earlier[a] = OR(wf[a1] && conflict(a1, a) for all a1 that is scheduled before a)
```

**Conflict inside an action**:
```
conflict_inside[a] = OR(conflict(M1, M2) && reach(m1) && reach(m2) for every m1, m2 in a, m1 is before m2)
// where reach(m) is a hardware signal representing the condition under which method call m 
// can be reached from the root of the action (considering all txn.if conditions along the path)
```

**NOTE**: The `static` mode uses the conflict matrix of current module to determine "Conflict with earlier actions" and uses the conflict matrices of the called modules to determine "Conflict inside an action".

#### Dynamic Mode (Precise) 
Dynamic mode uses reachability analysis to determine actual conflicts:

```
wf[action] = enabled[action] && 
             AND{for every method m in action: 
                 NOT(reach(m, action) && conflict_with_earlier(m))}
```

This mode generates more complex hardware but allows more concurrency by considering actual execution paths.

**Method Call Tracking**: Track which action methods have been called
```
method_called[M] = OR{wf[a] && OR(reach(m, action) for m in M) for every action that has been scheduled and calls M} 
                  || OR(reach(m, current_action) for every m in M and m has been processed in the current action) 
```

**Conflict with earlier actions**: For the method call `m in M` to be processed in the current action,
```
conflict_with_earlier(m) = OR(method_called[M'] && conflict(M', M) for every M' in method_called)
```

**NOTE**: The `dynamic` mode uses the conflict matrix of the called modules to determine "Conflict with earlier actions", which unifies all the conflict scenarios for the current action ("inside an action" and "with earlier actions" in `static` mode). The conflict matrix of the current module is not used for the current module's will-fire logic.

**Trade-offs**:
- Static mode: Simpler hardware, conservative conflict resolution, may block valid concurrent executions
- Dynamic mode: Complex hardware with reachability tracking, precise conflict detection, maximum concurrency

### 4. Reachability Analysis

Reachability analysis is crucial for precise conflict detection. It computes hardware signals representing when each method call can be reached:

```cpp
class ReachabilityAnalysis {
  void analyzeAction(ActionOp action, ConversionContext &ctx) {
    // Start with unconditional reachability
    Value pathCondition = createTrue();
    
    // Walk the action tracking control flow
    action.walk([&](Operation *op) {
      if (auto ifOp = dyn_cast<IfOp>(op)) {
        // Branch creates two path conditions
        Value thenCond = createAnd(pathCondition, ifOp.getCondition());
        Value elseCond = createAnd(pathCondition, createNot(ifOp.getCondition()));
        
        // Recursively process branches
        processRegion(ifOp.getThenRegion(), thenCond, ctx);
        processRegion(ifOp.getElseRegion(), elseCond, ctx);
        
        return WalkResult::skip(); // Don't walk into regions
      }
      
      if (auto callOp = dyn_cast<CallOp>(op)) {
        // Record: this call is reachable when pathCondition is true
        ctx.reachabilityConditions[callOp] = pathCondition;
      }
    });
  }
};
```

**Implementation Details**:
- Path conditions are built using FIRRTL's boolean operations
- Nested if statements multiply path conditions
- Each method call gets a unique reachability condition
- The analysis runs before will-fire calculation

#### Example: Complex Reachability
```mlir
txn.rule @complex {
  %c1 = arith.cmpi eq, %x, %c0 : i32
  txn.if %c1 {
    txn.call @reg::@write(%v1) : (i32) -> ()     // reach = %c1
  } else {
    %c2 = arith.cmpi gt, %y, %c10 : i32
    txn.if %c2 {
      txn.call @reg::@write(%v2) : (i32) -> ()   // reach = !%c1 && %c2
    }
  }
}
```

The two write calls conflict, but `conflict_inside = %c1 && (!%c1 && %c2) = false`, so no internal conflict exists.

### 5. Conflict Inside Calculation

This detects when an action might call conflicting methods:

```cpp
Value calculateConflictInside(ActionOp action, ConversionContext &ctx) {
  // First run reachability analysis for this action
  ReachabilityAnalysis reachAnalysis;
  reachAnalysis.analyzeAction(action, ctx);
  
  // Get all method calls with their reachability
  auto methodCalls = collectMethodCalls(action);
  
  Value conflictInside = createFalse();
  
  // Check every pair of method calls
  for (size_t i = 0; i < methodCalls.size(); ++i) {
    for (size_t j = i + 1; j < methodCalls.size(); ++j) {
      auto [inst1, method1] = methodCalls[i];
      auto [inst2, method2] = methodCalls[j];
      
      // Skip if different instances (no conflict possible)
      if (inst1 != inst2) continue;
      
      // Check if methods conflict
      if (hasConflict(method1, method2)) {
        // Conflict occurs when both are reachable
        Value reach1 = ctx.reachabilityConditions[callOp1];
        Value reach2 = ctx.reachabilityConditions[callOp2];
        Value bothReach = createAnd(reach1, reach2);
        
        conflictInside = createOr(conflictInside, bothReach);
      }
    }
  }
  
  return conflictInside;
}
```

**Critical Insights**:
- Only method calls to the *same instance* can conflict internally
- The conflict matrix of the called module determines method conflicts
- Reachability conditions prevent false conflicts from exclusive branches
- The resulting signal blocks action execution when internal conflicts would occur

### 6. Method Implementation Details

#### Value Methods
Value methods are pure combinational logic:

```cpp
void convertValueMethod(ValueMethodOp method, ConversionContext &ctx) {
  // Build combinational logic for method body
  Value result = convertMethodBody(method.getBody(), ctx);
  
  // Connect to output port
  auto outputPort = ctx.getPort(method.getName() + "_OUT");
  ctx.builder.create<ConnectOp>(method.getLoc(), outputPort, result);
}
```

#### Action Methods
Action methods involve will-fire logic and side effects:

```cpp
void convertActionMethod(ActionMethodOp method, ConversionContext &ctx) {
  // Get enable signal
  Value enable = method.getAlwaysEnable() ? 
    createTrue() : ctx.getPort(method.getName() + "_EN");
  
  // Calculate conflicts with earlier actions
  Value noConflicts = calculateNoConflicts(method, ctx);
  
  // Will-fire = enabled && no conflicts
  Value willFire = createAnd(enable, noConflicts);
  ctx.willFireSignals[method.getName()] = willFire;
  
  // Execute body conditionally
  ctx.builder.create<WhenOp>(willFire, [&] {
    convertMethodBody(method.getBody(), ctx);
  });
  
  // Set ready signal
  if (!method.getAlwaysReady()) {
    auto readyPort = ctx.getPort(method.getName() + "_RDY");
    ctx.builder.create<ConnectOp>(readyPort, noConflicts);
  }
}
```

### 7. Parametric Primitive Support

The conversion automatically instantiates primitives with correct types:

```cpp
FModuleOp getOrCreatePrimitive(StringRef primName, Type dataType) {
  // Generate unique name: Register_i32_impl, Wire_i8_impl, etc.
  std::string uniqueName = primName.str() + "_" + 
                          getTypeName(dataType) + "_impl";
  
  // Check if already created
  if (auto existing = symbolTable.lookup<FModuleOp>(uniqueName))
    return existing;
  
  // Create new primitive instance
  if (primName == "Register") {
    return createRegisterPrimitive(uniqueName, dataType);
  } else if (primName == "Wire") {
    return createWirePrimitive(uniqueName, dataType);
  }
  // ... other primitives
}
```

**Key Features**:
- Type parameters extracted from instance operations: `@Register<i32>`
- Unique FIRRTL modules generated per type: `Register_i32_impl`
- Primitives constructed on-demand during conversion
- Proper bit width calculation from MLIR integer types

## Advanced Topics

### Combinational Loop Detection

The converter must detect and prevent combinational loops:

```cpp
class CombinationalLoopDetector {
  DenseMap<StringRef, SmallVector<StringRef>> dependencies;
  
  void analyze(ModuleOp module) {
    // Build dependency graph
    module.walk([&](CallOp call) {
      if (isValueMethod(call.getCallee())) {
        // Value method calls create combinational dependencies
        dependencies[currentMethod].push_back(call.getCallee());
      }
    });
    
    // Detect cycles using DFS
    if (hasCycle()) {
      emitError("Combinational loop detected: " + formatCycle());
    }
  }
};
```

**Implementation Status**: Awaiting primitive attribute support to define combinational paths through primitives.

### Optimization Opportunities

1. **Common Subexpression Elimination**: Reuse will-fire calculations
2. **Dead Code Elimination**: Remove unreachable method calls  
3. **Conflict Reduction**: Simplify conflict checks when possible
4. **Constant Propagation**: Fold constant conditions in reachability
5. **Schedule Optimization**: Reorder conflict-free actions for better pipelining

## Implementation Details

### Pass Structure

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
      
    case ConflictRelation::SA: // a1 after a2 (shouldn't happen in well-ordered schedule)
      assert(false && "Invalid schedule order - SA constraint violated");
      
    case ConflictRelation::CF: // Conflict-free
      break;
  }
  
  return false;
}
```

## Implementation Status

### Completed Features
- ✅ Full TxnToFIRRTL conversion pass with both static and dynamic modes
- ✅ Parametric primitive support with automatic instantiation
- ✅ Conflict inside detection with reachability analysis
- ✅ Method attribute support for signal naming and optimization
- ✅ Proper handling of module hierarchy and dependencies
- ✅ 45/45 tests passing

### Current Limitations
- Multi-cycle operations not yet supported (requires timing attributes)
- Combinational loop detection awaits primitive path attributes
- Non-synthesizable primitives will fail translation

### Usage Examples

```bash
# Basic conversion (uses dynamic mode by default)
sharp-opt --convert-txn-to-firrtl input.mlir

# Use conservative static mode
sharp-opt --convert-txn-to-firrtl="will-fire-mode=static" input.mlir

# With other passes in pipeline
sharp-opt --convert-txn-to-firrtl --firrtl-lower-types output.mlir
```

## Example: Complete Translation

### Input Txn Module
```mlir
txn.module @Counter {
  %count = txn.instance @count of @Register<i32> : !txn.module<"Register">
  
  txn.rule @increment {
    %val = txn.call @count::@read() : () -> i32
    %one = arith.constant 1 : i32
    %inc = arith.addi %val, %one : i32
    txn.call @count::@write(%inc) : (i32) -> ()
    txn.return
  }
  
  txn.action_method @reset() {
    %zero = arith.constant 0 : i32
    txn.call @count::@write(%zero) : (i32) -> ()
    txn.return
  }
  
  txn.schedule [@increment, @reset] {
    conflict_matrix = {
      "increment,reset" = 2 : i32  // C (conflict)
    }
  }
}
```

**Key Points**:
- Parametric instance `@Register<i32>` triggers automatic primitive construction
- Both methods call write on the same register (conflict)
- Schedule specifies explicit conflict relationship

### Generated FIRRTL Structure (Conceptual)

**Note**: The following FIRRTL code is shown in plain text format for illustration purposes only. The actual implementation MUST generate this structure using CIRCT FIRRTL dialect APIs.

```firrtl
circuit Counter:
  module Register_i32_impl:  // Auto-generated primitive
    input clock: Clock
    input reset: UInt<1>
    output read_data: UInt<32>
    input write_enable: UInt<1>
    input write_data: UInt<32>
    
    reg register: UInt<32>, clock with :
      reset => (reset, UInt<32>(0))
      
    ; Read is always available
    connect read_data, register
    
    ; Write updates register when enabled
    when write_enable:
      connect register, write_data
  
  module Counter:
    input clock: Clock
    input reset: UInt<1>
    
    ; Action method ports
    input resetEN: UInt<1>
    output resetRDY: UInt<1>
    
    ; Instance of parametric register
    inst count of Register_i32_impl
    connect count.clock, clock
    connect count.reset, reset
    
    ; Rule: increment (always enabled)
    node increment_condition = UInt<1>(1)  ; always enabled
    
    ; Will-fire: increment (no conflicts with earlier actions)
    node increment_wf = increment_condition
    
    ; Rule body
    connect count.read_enable, increment_wf
    node inc_value = add(count.read_data, UInt<32>(1))
    connect count.write_enable, increment_wf
    connect count.write_data, inc_value
    
    ; Method: reset
    ; Check conflicts with increment (from conflict matrix)
    node reset_conflicts_increment = and(CM_increment_reset_C, increment_wf)
    node reset_wf = and(resetEN, not(reset_conflicts_increment))
    
    when reset_wf:
      connect count.write_enable, UInt<1>(1)
      connect count.write_data, UInt<32>(0)
    
    connect resetRDY, not(reset_conflicts_increment)
```

**Translation Steps**:
1. Register_i32_impl module generated from Register<i32> primitive
2. Will-fire logic enforces conflict matrix (increment blocks reset)
3. Ready signal indicates when reset can execute
4. Both methods generate appropriate enable/data signals for the register

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

## Action Scheduling Algorithm

### Overview

The Action Scheduling Algorithm is an analysis pass that automatically completes partial schedules to ensure all actions (rules and methods) are properly ordered while minimizing conflicts. This pass runs after conflict matrix inference and before FIRRTL translation.

### Problem Statement

Given:
- A module with actions A = {a₁, a₂, ..., aₙ}
- A (possibly incomplete) partial schedule S_partial ⊆ A
- A conflict matrix CM where CM[aᵢ, aⱼ] ∈ {SB, SA, C, CF}

Find:
- A complete schedule S_complete = [s₁, s₂, ..., sₙ] containing all actions in A
- Such that S_partial order is preserved in S_complete
- Minimizing the number of "bad edges": conflicts where aᵢ SB aⱼ but i > j in the schedule, or aᵢ SA aⱼ but i < j in the schedule

### Algorithm Design

#### Phase 1: Dependency Graph Construction

```cpp
struct SchedulingGraph {
  // Nodes are actions
  DenseSet<StringRef> actions;
  
  // Edges represent ordering constraints
  // edge (a, b) means a must come before b
  DenseMap<StringRef, SmallVector<StringRef>> mustPrecede;
  
  // Track partial schedule constraints
  DenseMap<StringRef, int> partialOrder;
};

void buildSchedulingGraph(ModuleOp module, ConflictMatrix &cm, 
                         SchedulingGraph &graph) {
  // 1. Add all actions as nodes
  for (auto rule : module.getOps<RuleOp>())
    graph.actions.insert(rule.getSymName());
  for (auto method : module.getOps<ActionMethodOp>())
    graph.actions.insert(method.getSymName());
    
  // 2. Extract partial schedule constraints
  if (auto schedule = module.getOps<ScheduleOp>().begin()) {
    auto partialActions = schedule->getActions();
    for (size_t i = 0; i < partialActions.size(); ++i) {
      graph.partialOrder[partialActions[i]] = i;
      // Add edges to maintain partial order
      for (size_t j = i + 1; j < partialActions.size(); ++j) {
        graph.mustPrecede[partialActions[i]].push_back(partialActions[j]);
      }
    }
  }
  
  // 3. Add edges from conflict matrix (SA relationships)
  for (auto [a1, a2] : graph.actions × graph.actions) {
    if (cm[a1, a2] == ConflictRelation::SA) {
      // a1 Sequential After a2 means a2 must precede a1
      graph.mustPrecede[a2].push_back(a1);
    }
  }
}
```

#### Phase 2: Topological Sort with Conflict Minimization

```cpp
struct SchedulingCost {
  int sbViolations = 0;  // Count of SB relationships violated
  int conflicts = 0;     // Count of C relationships not separated
};

SmallVector<StringRef> computeOptimalSchedule(SchedulingGraph &graph, 
                                              ConflictMatrix &cm) {
  // 1. Check for cycles (would make scheduling impossible)
  if (hasCycle(graph.mustPrecede)) {
    emitError("Cyclic dependencies in action scheduling");
    return {};
  }
  
  // 2. Compute all valid topological orderings
  SmallVector<SmallVector<StringRef>> validSchedules;
  generateTopologicalOrderings(graph, validSchedules);
  
  // 3. Evaluate each schedule and pick the best
  SmallVector<StringRef> bestSchedule;
  SchedulingCost bestCost = {INT_MAX, INT_MAX};
  
  for (auto &schedule : validSchedules) {
    SchedulingCost cost = evaluateSchedule(schedule, cm);
    if (cost.sbViolations < bestCost.sbViolations ||
        (cost.sbViolations == bestCost.sbViolations && 
         cost.conflicts < bestCost.conflicts)) {
      bestCost = cost;
      bestSchedule = schedule;
    }
  }
  
  return bestSchedule;
}

SchedulingCost evaluateSchedule(ArrayRef<StringRef> schedule, 
                                ConflictMatrix &cm) {
  SchedulingCost cost;
  DenseMap<StringRef, size_t> position;
  
  // Build position map
  for (size_t i = 0; i < schedule.size(); ++i)
    position[schedule[i]] = i;
  
  // Count violations
  for (size_t i = 0; i < schedule.size(); ++i) {
    for (size_t j = i + 1; j < schedule.size(); ++j) {
      auto rel = cm[schedule[i], schedule[j]];
      
      if (rel == ConflictRelation::SB) {
        // schedule[i] should be before schedule[j], which it is
        // No violation
      } else if (rel == ConflictRelation::SA) {
        // schedule[i] should be after schedule[j], but it's before
        // This should be impossible if graph was built correctly
        assert(false && "SA constraint violated");
      } else if (rel == ConflictRelation::C) {
        // Conflict exists, but at least they're separated
        cost.conflicts++;
      }
    }
  }
  
  // Also check reverse direction for SB violations
  for (auto [a1, a2] : cm.getAllPairs()) {
    if (cm[a1, a2] == ConflictRelation::SB) {
      if (position[a1] > position[a2]) {
        cost.sbViolations++;
      }
    }
  }
  
  return cost;
}
```

#### Phase 3: Heuristic Algorithm (for large action sets)

When the number of actions is large, enumerating all topological orderings becomes intractable. Use a greedy heuristic:

```cpp
SmallVector<StringRef> computeHeuristicSchedule(SchedulingGraph &graph,
                                                ConflictMatrix &cm) {
  SmallVector<StringRef> schedule;
  DenseSet<StringRef> scheduled;
  DenseMap<StringRef, int> inDegree;
  
  // Compute initial in-degrees
  for (auto action : graph.actions) {
    inDegree[action] = 0;
  }
  for (auto &[from, toList] : graph.mustPrecede) {
    for (auto to : toList) {
      inDegree[to]++;
    }
  }
  
  // Kahn's algorithm with conflict-aware selection
  while (scheduled.size() < graph.actions.size()) {
    // Find all actions with in-degree 0
    SmallVector<StringRef> ready;
    for (auto action : graph.actions) {
      if (!scheduled.count(action) && inDegree[action] == 0) {
        ready.push_back(action);
      }
    }
    
    // Select best action from ready set
    StringRef best = selectBestAction(ready, scheduled, schedule, cm);
    schedule.push_back(best);
    scheduled.insert(best);
    
    // Update in-degrees
    for (auto successor : graph.mustPrecede[best]) {
      inDegree[successor]--;
    }
  }
  
  return schedule;
}

StringRef selectBestAction(ArrayRef<StringRef> ready,
                           const DenseSet<StringRef> &scheduled,
                           ArrayRef<StringRef> currentSchedule,
                           ConflictMatrix &cm) {
  StringRef bestAction;
  int bestScore = INT_MAX;
  
  for (auto candidate : ready) {
    int score = 0;
    
    // Penalize placing this action if it has SB relationships
    // with already scheduled actions
    for (auto scheduled : currentSchedule) {
      if (cm[candidate, scheduled] == ConflictRelation::SB) {
        score += 10;  // Heavy penalty for SB violation
      } else if (cm[candidate, scheduled] == ConflictRelation::C) {
        score += 1;   // Light penalty for conflicts
      }
    }
    
    // Prefer actions from partial schedule in their original order
    if (auto pos = partialOrder.find(candidate); 
        pos != partialOrder.end()) {
      score -= 100;  // Strong preference for maintaining partial order
    }
    
    if (score < bestScore) {
      bestScore = score;
      bestAction = candidate;
    }
  }
  
  return bestAction;
}
```

### Implementation as Analysis Pass

```cpp
class ActionSchedulingPass : public OperationPass<ModuleOp> {
  void runOnOperation() override {
    auto module = getOperation();
    
    // Skip if schedule is already complete
    if (hasCompleteSchedule(module))
      return;
    
    // Get conflict matrix (must run after inference pass)
    auto cmAnalysis = getAnalysis<ConflictMatrixAnalysis>();
    auto &cm = cmAnalysis.getConflictMatrix(module);
    
    // Build scheduling graph
    SchedulingGraph graph;
    buildSchedulingGraph(module, cm, graph);
    
    // Compute optimal schedule
    SmallVector<StringRef> schedule;
    if (graph.actions.size() <= 10) {
      // Use exact algorithm for small modules
      schedule = computeOptimalSchedule(graph, cm);
    } else {
      // Use heuristic for larger modules
      schedule = computeHeuristicSchedule(graph, cm);
    }
    
    if (schedule.empty()) {
      signalPassFailure();
      return;
    }
    
    // Replace or create schedule operation
    replaceScheduleOp(module, schedule);
  }
  
  bool hasCompleteSchedule(ModuleOp module) {
    auto scheduleOps = module.getOps<ScheduleOp>();
    if (scheduleOps.empty())
      return false;
      
    auto schedule = *scheduleOps.begin();
    DenseSet<StringRef> scheduled(schedule.getActions().begin(),
                                  schedule.getActions().end());
    
    // Check if all actions are in the schedule
    for (auto rule : module.getOps<RuleOp>()) {
      if (!scheduled.count(rule.getSymName()))
        return false;
    }
    for (auto method : module.getOps<ActionMethodOp>()) {
      if (!scheduled.count(method.getSymName()))
        return false;
    }
    
    return true;
  }
};
```

### Example Usage

Input with partial schedule:
```mlir
txn.module @Example {
  txn.rule @r1 { ... }
  txn.rule @r2 { ... }
  txn.action_method @m1() { ... }
  txn.action_method @m2() { ... }
  
  // Partial schedule only specifies r1 before m1
  txn.schedule [@r1, @m1] {
    conflict_matrix = {
      "r1,r2" = 0 : i32,    // r1 SB r2
      "r2,m2" = 0 : i32,    // r2 SB m2
      "m1,m2" = 2 : i32     // m1 C m2
    }
  }
}
```

After scheduling pass:
```mlir
txn.module @Example {
  txn.rule @r1 { ... }
  txn.rule @r2 { ... }
  txn.action_method @m1() { ... }
  txn.action_method @m2() { ... }
  
  // Complete schedule maintains r1 < m1 and minimizes conflicts
  txn.schedule [@r1, @r2, @m1, @m2] {
    conflict_matrix = {
      "r1,r2" = 0 : i32,    // r1 SB r2 ✓
      "r2,m2" = 0 : i32,    // r2 SB m2 ✓
      "m1,m2" = 2 : i32     // m1 C m2 (unavoidable)
    }
  }
}
```

### Properties and Guarantees

1. **Completeness**: The algorithm always produces a complete schedule if one exists
2. **Partial Order Preservation**: The original partial schedule order is maintained
3. **Optimality**: For small modules (≤10 actions), finds optimal schedule
4. **Efficiency**: O(n!) for exact algorithm, O(n²) for heuristic
5. **Determinism**: Given the same input, produces the same schedule

## Future Directions

1. **Multi-cycle Support**: Extend timing attributes for pipelined operations
2. **Formal Verification**: Generate assertions for conflict properties
3. **Performance Analysis**: Static timing and resource utilization estimates
4. **Debug Infrastructure**: Transaction-level debugging support
5. **Advanced Scheduling**: Support for out-of-order execution with dependencies
6. **Power Optimization**: Clock gating for inactive methods

## References

- Koika: A Hardware Design Language (PLDI 2020)
- Bluespec Language Reference
- CIRCT FIRRTL Dialect Documentation
- Sharp Transaction Dialect Specification (docs/txn.md)