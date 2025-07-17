// Define custom CAM (Content Addressable Memory)
txn.primitive @CAM<width: i32, depth: i32> {
  // Define methods
  txn.method @write(%addr: i32, %data: !width) -> ()
  txn.method @search(%pattern: !width) -> i32  // Returns address or -1
  txn.method @clear() -> ()
  
  // Software semantics
  txn.software_semantics {
    %storage = std::map<i32, !width>
    %last_match = i32
  }
}

// Using custom primitive
txn.module @NetworkRouter {
  txn.instance @cam of @CAM<32, 1024> 
  
  txn.action_method @add_route(%prefix: i32, %port: i32) {
    txn.call @cam::@write(%prefix, %port) : (i32, i32) -> ()
    txn.yield
  }
  
  txn.value_method @lookup(%addr: i32) -> i32 {
    %port = txn.call @cam::@search(%addr) : (i32) -> i32
    txn.return %port : i32
  }
  
  txn.schedule [@add_route, @lookup]
}

// Primitive with internal state machine
txn.primitive @Arbiter<clients: i32> {
  txn.method @request(%id: i32) -> ()
  txn.method @grant() -> i32
  txn.method @release(%id: i32) -> ()
  
  txn.hardware_semantics {
    // Priority encoder
    %requests = hw.aggregate %clients : i1
    %grant_id = hw.priority_encode %requests : i32
  }
  
  txn.conflict_matrix {
    "request,grant" = 2 : i32,     // C
    "grant,release" = 2 : i32,     // C
    "request,release" = 2 : i32    // C
  }
}

// Multi-port memory primitive
txn.primitive @MultiPortRAM<width: i32, depth: i32, read_ports: i32, write_ports: i32> {
  txn.method @read(%port: i32, %addr: i32) -> !width
  txn.method @write(%port: i32, %addr: i32, %data: !width) -> ()
  
  txn.timing {
    "read" = "combinational",
    "write" = "static(1)"
  }
  
  txn.constraints {
    // No two writes to same address
    conflict_if(%port1 != %port2 && %addr1 == %addr2) {
      "write[%port1],write[%port2]" = 3 : i32  // CF
    }
  }
}