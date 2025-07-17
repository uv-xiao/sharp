// Simplified AES-like encryption engine
txn.module @AESEngine {
  // State array
  txn.instance @state of @Register<i128> 
  
  // Round counter
  txn.instance @round of @Register<i8> 
  
  txn.action_method @load_plaintext(%data: i128) {
    txn.call @state::@write(%data) : (i128) -> ()
    %zero = arith.constant 0 : i8
    txn.call @round::@write(%zero) : (i8) -> ()
    txn.yield
  }
  
  txn.action_method @round_function() {
    %s = txn.call @state::@read() : () -> i128
    %r = txn.call @round::@read() : () -> i8
    
    // Simplified round: just XOR with round number expanded
    %r_ext = arith.extui %r : i8 to i128
    %c16 = arith.constant 16 : i128
    %round_key = arith.muli %r_ext, %c16 : i128
    
    // AddRoundKey (simplified)
    %next_state = arith.xori %s, %round_key : i128
    
    txn.call @state::@write(%next_state) : (i128) -> ()
    
    // Increment round
    %one = arith.constant 1 : i8
    %next_round = arith.addi %r, %one : i8
    txn.call @round::@write(%next_round) : (i8) -> ()
    
    txn.yield
  }
  
  txn.value_method @get_ciphertext() -> i128 {
    %s = txn.call @state::@read() : () -> i128
    txn.return %s : i128
  }
  
  txn.value_method @is_complete() -> i1 {
    %r = txn.call @round::@read() : () -> i8
    %ten = arith.constant 10 : i8
    %done = arith.cmpi eq, %r, %ten : i8
    txn.return %done : i1
  }
  
  // Auto-process rounds
  txn.rule @auto_round {
    %done = txn.call @this.is_complete() : () -> i1
    %true = arith.constant true
    %not_done = arith.xori %done, %true : i1
    scf.if %not_done {
      txn.call @this.round_function() : () -> ()
    }
    txn.yield
  }
  
  txn.schedule [@load_plaintext, @round_function, @get_ciphertext, @is_complete, @auto_round] {
    security_features = {
      "constant_time" = true
    }
  }
}