# Sharp Project Notes

## Papers and References

### Making Concurrent Hardware Verification Sequential - Bourgeat et al. (2025)

#### Overview
This paper addresses the exponential complexity problem in verifying concurrent hardware modules by introducing a sequential verification methodology that preserves hardware's concurrent nature.

#### Key Problem
- Traditional hardware verification requires considering all 2^n combinations of concurrent method calls
- This leads to exponential proof complexity as module APIs grow
- Makes modular hardware verification significantly harder than software verification

#### Core Insights and Techniques

1. **One Action Method Restriction**
   - Allow only one state-modifying (action) method per logical cycle per module
   - Multiple read-only (value) methods can execute concurrently
   - Enables sequential reasoning while maintaining concurrent execution

2. **Sequential Characterization**
   - Through language restrictions, enable sequential reasoning about hardware modules
   - Preserves concurrent execution in the final hardware
   - Reduces verification complexity from exponential to linear

3. **Transactional Semantics**
   - Introduce abort semantics that propagate when preconditions aren't met
   - Similar to software transaction models
   - Clean handling of backpressure and control flow

4. **Fjfj Language**
   - New Bluespec-inspired formal language embedded in Coq
   - Supports sequential verification of concurrent hardware
   - Provides mechanized proof framework

#### Methodology

1. **Module Specification**
   - Rules: spontaneous state transitions
   - Value methods: pure observations (read-only)
   - Action methods: state transformations

2. **Sequential Verification Process**
   - Verify each method/rule individually
   - Use refinement relations between implementation and specification
   - Leverage one-action-method restriction to avoid concurrent interference

3. **Compilation Strategy**
   - Logical cycles (sequential semantics) compiled to physical clock cycles
   - Multiple non-conflicting rules can execute in same physical cycle
   - Automatic handling of control signals (ready/enable)

#### Technical Contributions

- **Refinement Theorem**: Mechanized proof enabling modular composition
- **Formal Semantics**: Complete operational semantics for Fjfj with abort propagation
- **Primitive Module Abstraction**: Custom primitive modules for specifications
- **Mechanized Examples**: Three verified case studies in Coq

#### Case Studies

1. **5-Stage RISC-V Pipeline**
   - Demonstrates handling of complex control flow and hazards
   - Verifies refinement to ISA specification

2. **Parameterized N×M Crossbar**
   - Shows handling of parameterized designs
   - Verifies packet routing correctness

3. **Programmable Network Switch**
   - Complex stateful processing with match-action tables
   - Verifies packet processing semantics

#### Limitations

- No automatic synthesis from Fjfj to RTL (manual translation to Bluespec required)
- One-action-method restriction may require design restructuring
- Currently limited to natural numbers for method arguments/returns
- Learning curve for sequential-within-concurrent model

#### Relevance to Sharp Project

1. **Modular Verification**: Independent verification and composition of processor components
2. **Sequential Reasoning**: Simplifies verification of pipelines and concurrent features
3. **Refinement-Based Approach**: Proving optimized implementations match specifications
4. **Transaction Semantics**: Robust interface design between components
5. **Parameterized Designs**: Useful for configurable processor features
6. **Coq Integration**: Aligns with formal verification goals
7. **Primitive Abstractions**: High-level specifications of architectural components

#### Key Takeaway
The paper's approach makes concurrent hardware verification tractable through sequential reasoning while preserving hardware's inherent concurrency - particularly valuable for verifying complex processor designs.

---

## Implementation Ideas for Sharp

### Potential Applications

1. **Pipeline Verification**
   - Apply sequential verification to individual pipeline stages
   - Use refinement to prove pipeline implements ISA correctly
   - Handle hazards and forwarding with transaction semantics

2. **Module Interface Design**
   - Adopt value/action method distinction in Sharp dialect
   - Use abort semantics for clean error handling
   - Design composable module interfaces

3. **Verification Infrastructure**
   - Consider Coq integration for formal proofs
   - Implement refinement checking in Sharp tools
   - Support parameterized component verification

### Next Steps
- Study Fjfj's formal semantics for dialect design insights
- Explore how to integrate sequential verification into Sharp's workflow
- Consider transaction-like semantics for architectural components