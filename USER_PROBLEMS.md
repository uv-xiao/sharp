
## Tutorial

### Chapter 1

**Conflict Matrix Inference** FIXED✅

```bash
build/bin/sharp-opt examples/sharp-tutorial/chapter1-basics/toggle.mlir --sharp-infer-conflict-matrix                                                                               
```

```mlir
module {
  txn.module @Toggle {
    %0 = txn.instance @state of @Register<i1> : !txn.module<"Register">
    txn.value_method @read() -> i1 {
      %1 = txn.call @state::@read() : () -> i1
      txn.return %1 : i1
    }
    txn.action_method @toggle() {
      %1 = txn.call @state::@read() : () -> i1
      %true = arith.constant true
      %2 = arith.xori %1, %true : i1
      txn.call @state::@write(%2) : (i1) -> ()
      txn.yield
    }
    txn.rule @default {
      %1 = txn.call @state::@read() : () -> i1
      txn.call @state::@write(%1) : (i1) -> ()
    }
    txn.schedule [@toggle, @default] {conflict_matrix = {"default,default" = 2 : i32, "default,toggle" = 3 : i32, "toggle,default" = 3 : i32, "toggle,toggle" = 2 : i32}}
  }
} 
```

This is not correct, since default and toggle both calls "@state::@write" and they chould conflict. The problem is two-fold: (1) the @lib/Analysis/ConflictMatrixInference.cpp is not complete, so the 
conflict relationship is missed; (2) the ConflictMatrixInference should run in a topological order: the submodule should be analyzed first, which is not required yet; (3) before doing 
ConflictMatrixInference, the missing primitives should be added by another pass (maybe named PrimitiveGen?) to call constructors under @/home/uvxiao/sharp/lib/Dialect/Txn/primitives/. Fix these issues and 
add description in @docs/analysis.md.