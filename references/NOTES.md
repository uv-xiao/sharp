# Sharp Project - Research Notes

## Overview

This document synthesizes three key papers on hardware design languages and verification:
1. **Fjfj (2025)**: Sequential verification of concurrent hardware
2. **Kôika (2020)**: Rule-based hardware design with explicit scheduling
3. **EQueue (2022)**: Compiler-driven simulation of hardware accelerators

These papers represent complementary approaches to managing complexity in hardware design through different abstractions and verification strategies.

---

## Paper Summaries

### 1. Making Concurrent Hardware Verification Sequential (Bourgeat et al., 2025)

**Problem**: Traditional hardware verification faces exponential complexity (2^n combinations) when verifying concurrent method calls.

**Solution**: Fjfj language with one-action-method restriction enables sequential reasoning while preserving concurrent execution.

**Key Concepts**:
- **One Action Method Restriction**: Only one state-modifying method per cycle per module
- **Transaction Semantics**: Abort propagation for failed preconditions
- **Sequential Verification**: Verify methods individually, compose modularly
- **Refinement-Based**: Prove implementations match specifications

**Technical Approach**:
- Language embedded in Coq for mechanized proofs
- Logical cycles compiled to physical clock cycles
- Multiple non-conflicting rules execute concurrently
- Automatic handling of control signals (ready/enable)

**Case Studies**: 5-stage RISC-V pipeline, N×M crossbar, programmable switch

### 2. The Essence of Bluespec: A Core Language for Rule-Based Hardware Design (Bourgeat et al., 2020)

**Problem**: Bluespec's performance depends on opaque static analysis and user hints, making it unpredictable.

**Solution**: Kôika provides explicit control over scheduling with deterministic cycle-accurate semantics.

**Key Concepts**:
- **Explicit Scheduling**: Direct control over rule execution order
- **ORAAT Property**: One-Rule-At-A-Time semantics preserved during concurrent execution
- **Ephemeral History Registers (EHRs)**: Enable intra-cycle data forwarding
- **Log-based Semantics**: Dynamic tracking ensures correctness

**Technical Approach**:
- Mechanized in Coq with verified compiler to RTL
- Rules generate logs of reads/writes
- Dynamic checks prevent ORAAT violations
- EHR ports (rd0/wr0, rd1/wr1) for forwarding

**Example - Collatz Sequence**:
```
rule divide = 
  let v = r.rd0 in
  if iseven(v) then r.wr0(v >> 1)

rule multiply =
  let v = r.rd1 in  
  if isodd(v) then r.wr1(3 * v + 1)

schedule collatz = [divide; multiply]
```

### 3. Compiler-Driven Simulation of Reconfigurable Hardware Accelerators (Li et al., 2022)

**Problem**: Traditional simulators are either too low-level (RTL) or require one-off engineering for each accelerator.

**Solution**: EQueue MLIR dialect enables multi-level simulation with separation of structure and behavior.

**Key Concepts**:
- **Structure Specification**: Explicit component hierarchy (processors, memories, DMA)
- **Event-Based Control**: Launch operations with dependencies
- **Multi-Level Simulation**: From tensor operations to detailed hardware
- **Performance Models**: Pluggable timing/power models per component

**Technical Approach**:
- Built on MLIR infrastructure
- Generic simulation engine interprets EQueue programs
- Progressive lowering through compilation pipeline
- Visualization of execution traces and bandwidth

**Example Structure**:
```mlir
kernel = equeue.create_proc(ARMr6)
sram = equeue.create_mem(SRAM, [64], 4)
pe0_dep = equeue.launch(...) in (launch_dep, pe0) {
  ofmap = addi(ifmap, 4)
}
```

---

## Unified Analysis

### Common Design Principles

1. **Separation of Concerns**
   - **Fjfj**: Separates verification (sequential) from execution (concurrent)
   - **Kôika**: Separates functional behavior from scheduling decisions
   - **EQueue**: Separates hardware structure from simulation logic

2. **Explicit Control Over Implicit Decisions**
   - **Fjfj**: Explicit module interfaces and method restrictions
   - **Kôika**: Explicit schedules replace hidden static analysis
   - **EQueue**: Explicit data movement and event dependencies

3. **Multi-Level Abstraction**
   - **Fjfj**: Logical cycles abstract from physical implementation
   - **Kôika**: Rules compose into schedules with preserved semantics
   - **EQueue**: Progressive refinement from tensors to hardware details

4. **Formal Foundations**
   - **Fjfj**: Mechanized refinement proofs in Coq
   - **Kôika**: Verified compiler with ORAAT theorem
   - **EQueue**: Precise event-based operational semantics

### Complementary Strengths

| Aspect | Fjfj | Kôika | EQueue |
|--------|------|-------|---------|
| **Primary Focus** | Modular verification | Performance predictability | Design exploration |
| **Abstraction** | Sequential reasoning | Rule-based design | Event-driven simulation |
| **Verification** | Refinement proofs | ORAAT preservation | Performance estimation |
| **Concurrency** | One-action restriction | EHR forwarding | Event dependencies |
| **Implementation** | Coq proofs | Verified RTL compiler | MLIR simulation |

### Evolution and Relationships

1. **Historical Progression**:
   - Kôika (2020) addresses Bluespec's opaque scheduling
   - EQueue (2022) enables flexible architectural exploration
   - Fjfj (2025) solves exponential verification complexity

2. **Technical Connections**:
   - All three build on rule-based/transaction concepts
   - Fjfj extends ideas from Kôika for verification
   - EQueue complements both with performance analysis

3. **Methodological Alignment**:
   - Emphasis on predictability and explicit control
   - Support for modular composition
   - Integration with formal methods tools

---

## Integration Opportunities for Sharp/Txn

### Direct Applications

1. **From Fjfj**:
   - One-action-method restriction for verifiable interfaces
   - Transaction abort semantics already adopted
   - Refinement-based verification methodology

2. **From Kôika**:
   - Explicit scheduling (already adopted as `txn.schedule`)
   - EHR mechanism for intra-cycle communication
   - Log-based dynamic correctness checking

3. **From EQueue**:
   - Multi-level simulation capabilities
   - Performance estimation framework
   - Event-based dependency tracking

### Proposed Synthesis

#### Architecture
```
┌─────────────────────────────────────┐
│         Txn Dialect (Sharp)         │
├─────────────────────────────────────┤
│ Verification Layer (Fjfj-inspired)  │
│ - One-action restriction            │
│ - Refinement proofs                 │
├─────────────────────────────────────┤
│ Execution Layer (Kôika-inspired)    │
│ - Explicit scheduling               │
│ - EHR forwarding                    │
│ - ORAAT preservation                │
├─────────────────────────────────────┤
│ Simulation Layer (EQueue-inspired)  │
│ - Multi-level models                │
│ - Performance estimation            │
│ - Event visualization               │
└─────────────────────────────────────┘
```

#### Implementation Strategy

1. **Phase 1: Enhanced Scheduling**
   - Extend `txn.schedule` with Kôika's forwarding semantics
   - Add EHR-like ports to registers for intra-cycle communication
   - Implement dynamic ORAAT checking

2. **Phase 2: Verification Infrastructure**
   - Adopt Fjfj's one-action restriction where appropriate
   - Build refinement checking tools
   - Integrate with Coq for mechanized proofs

3. **Phase 3: Performance Analysis**
   - Add EQueue-style simulation annotations
   - Build multi-level performance models
   - Create visualization tools for execution traces

### Research Questions

1. Can we unify Kôika's EHRs with Fjfj's one-action restriction?
2. How to integrate EQueue's simulation with formal verification?
3. Can we automatically derive performance models from verified designs?
4. What's the right abstraction level for each design phase?

---

## Conclusions

These three papers represent a coherent evolution in hardware design methodology:
- **Correctness**: From opaque Bluespec to explicit Kôika to verifiable Fjfj
- **Performance**: From static analysis to dynamic scheduling to multi-level simulation
- **Abstraction**: From RTL details to rule-based design to transaction-level modeling

The Sharp/Txn dialect is well-positioned to integrate these advances into a unified framework that supports the entire hardware design lifecycle from specification through verification to implementation.