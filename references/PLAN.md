- Add method relations in modules, including "sequence ahead" (SA), "conflict" (C), 
- Add timing attributes to rules/methods. It can be either combinational or multi-cycle: Combinational | MultiCycle(cycle: i32)
- Add passes to translate sharp.txn into firrtl (and finally into verilog/systemverilog)
  - Reference: 
    - references/Bourgeat-2020-Koika.pdf, code at https://github.com/mit-plv/koika/blob/master/coq/CircuitGeneration.v
  - Synthesizable primitives (e.g. register) should be translated into firrtl primitives, the corresponding firrtl primitives should be added to a directory (maybe lib/Dialect/Txn/firrtl_primitives/), every primitive correspond to a file which contains the firrtl module of the primitive. Currently, we need Register and Ephemeral History Register (EHR) primitives.
  - Nonsynthesizable primitives and multi-cycle rules/methods should be checked by an analysis. Currently, they are not allowed for the translation (throw out a failure message).
  