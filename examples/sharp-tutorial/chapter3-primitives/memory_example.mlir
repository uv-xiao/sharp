// Example demonstrating Memory primitive for caching
txn.module @CacheController {
  // 256-entry cache memory
  %cache = txn.instance @cache of @Memory<i64> : !txn.module<"Memory">
  %tags = txn.instance @tags of @Memory<i32> : !txn.module<"Memory">
  
  // Cache lookup
  txn.value_method @lookup(%addr: i32) -> (i1, i64) {
    // Extract tag and index
    %c8 = arith.constant 8 : i32
    %tag = arith.shrui %addr, %c8 : i32
    %c255 = arith.constant 255 : i32
    %index = arith.andi %addr, %c255 : i32
    
    // Check tag
    %stored_tag = txn.call @tags::@read(%index) : (i32) -> i32
    %hit = arith.cmpi eq, %tag, %stored_tag : i32
    
    // Read data
    %data = txn.call @cache::@read(%index) : (i32) -> i64
    
    txn.return %hit, %data : i1, i64
  }
  
  // Cache update
  txn.action_method @update(%addr: i32, %data: i64) {
    %c8 = arith.constant 8 : i32
    %tag = arith.shrui %addr, %c8 : i32
    %c255 = arith.constant 255 : i32
    %index = arith.andi %addr, %c255 : i32
    
    // Update tag and data
    txn.call @tags::@write(%index, %tag) : (i32, i32) -> ()
    txn.call @cache::@write(%index, %data) : (i32, i64) -> ()
    
    txn.yield
  }
  
  // Cache invalidation
  txn.action_method @invalidate() {
    txn.call @cache::@clear() : () -> ()
    txn.call @tags::@clear() : () -> ()
    txn.yield
  }
  
  txn.schedule [@lookup, @update, @invalidate]
}