//===- SpecMemory.cpp - Sharp Txn SpecMemory primitive implementation ----===//
//
// Part of the Sharp Project.
// SpecMemory is a memory with configurable latency for verification.
//
//===----------------------------------------------------------------------===//

#include "sharp/Dialect/Txn/TxnPrimitives.h"
#include "sharp/Dialect/Txn/TxnOps.h"
#include "sharp/Dialect/Txn/TxnTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace sharp {
namespace txn {

::sharp::txn::PrimitiveOp createSpecMemoryPrimitive(OpBuilder &builder, Location loc,
                                                    StringRef name, Type dataType,
                                                    unsigned addressWidth,
                                                    unsigned defaultLatency) {
  // Create interface type for the primitive
  auto moduleType = ::sharp::txn::ModuleType::get(builder.getContext(), 
                                                  StringAttr::get(builder.getContext(), name));
  
  // Create the primitive operation
  auto primitive = builder.create<::sharp::txn::PrimitiveOp>(loc, 
                                                            StringAttr::get(builder.getContext(), name),
                                                            builder.getStringAttr("spec"), 
                                                            TypeAttr::get(moduleType),
                                                            /*type_parameters=*/ArrayAttr());
  
  // Create a new builder for the primitive body
  OpBuilder::InsertionGuard guard(builder);
  Block *body = &primitive.getBody().emplaceBlock();
  builder.setInsertionPointToStart(body);
  
  // Define the methods
  auto addrType = builder.getIntegerType(addressWidth);
  auto readType = builder.getFunctionType({addrType}, {dataType});
  auto writeType = builder.getFunctionType({addrType, dataType}, {});
  auto setLatencyType = builder.getFunctionType({builder.getI32Type()}, {});
  auto getLatencyType = builder.getFunctionType({}, {builder.getI32Type()});
  auto clearType = builder.getFunctionType({}, {});
  
  // Create read method (value method with dynamic timing)
  builder.create<::sharp::txn::ValueMethodOp>(
      loc, builder.getStringAttr("read"), TypeAttr::get(readType),
      /*sym_visibility=*/StringAttr(), /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr(),
      /*result=*/StringAttr(), /*prefix=*/StringAttr());
  
  // Create write method (action method)
  builder.create<::sharp::txn::ActionMethodOp>(
      loc, builder.getStringAttr("write"), TypeAttr::get(writeType),
      /*sym_visibility=*/StringAttr(), /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr(),
      /*ready=*/StringAttr(), /*enable=*/StringAttr(),
      /*result=*/StringAttr(), /*prefix=*/StringAttr(),
      /*always_ready=*/UnitAttr(), /*always_enable=*/UnitAttr());
  
  // Create setLatency method (action method)
  builder.create<::sharp::txn::ActionMethodOp>(
      loc, builder.getStringAttr("setLatency"), TypeAttr::get(setLatencyType),
      /*sym_visibility=*/StringAttr(), /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr(),
      /*ready=*/StringAttr(), /*enable=*/StringAttr(),
      /*result=*/StringAttr(), /*prefix=*/StringAttr(),
      /*always_ready=*/UnitAttr(), /*always_enable=*/UnitAttr());
  
  // Create getLatency method (value method)
  builder.create<::sharp::txn::ValueMethodOp>(
      loc, builder.getStringAttr("getLatency"), TypeAttr::get(getLatencyType),
      /*sym_visibility=*/StringAttr(), /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr(),
      /*result=*/StringAttr(), /*prefix=*/StringAttr());
  
  // Create clear method (action method)
  builder.create<::sharp::txn::ActionMethodOp>(
      loc, builder.getStringAttr("clear"), TypeAttr::get(clearType),
      /*sym_visibility=*/StringAttr(), /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr(),
      /*ready=*/StringAttr(), /*enable=*/StringAttr(),
      /*result=*/StringAttr(), /*prefix=*/StringAttr(),
      /*always_ready=*/UnitAttr(), /*always_enable=*/UnitAttr());
  
  // Create schedule with conflict matrix
  auto conflictMatrix = builder.getDictionaryAttr({
    builder.getNamedAttr("read,read", builder.getI32IntegerAttr(3)),         // CF
    builder.getNamedAttr("read,write", builder.getI32IntegerAttr(2)),        // C
    builder.getNamedAttr("write,read", builder.getI32IntegerAttr(2)),        // C
    builder.getNamedAttr("write,write", builder.getI32IntegerAttr(2)),       // C
    builder.getNamedAttr("clear,read", builder.getI32IntegerAttr(2)),        // C
    builder.getNamedAttr("read,clear", builder.getI32IntegerAttr(2)),        // C
    builder.getNamedAttr("clear,write", builder.getI32IntegerAttr(2)),       // C
    builder.getNamedAttr("write,clear", builder.getI32IntegerAttr(2)),       // C
    builder.getNamedAttr("clear,clear", builder.getI32IntegerAttr(2)),       // C
    builder.getNamedAttr("setLatency,read", builder.getI32IntegerAttr(2)),   // C
    builder.getNamedAttr("read,setLatency", builder.getI32IntegerAttr(2)),   // C
    builder.getNamedAttr("setLatency,write", builder.getI32IntegerAttr(3)),  // CF
    builder.getNamedAttr("write,setLatency", builder.getI32IntegerAttr(3)),  // CF
    builder.getNamedAttr("setLatency,clear", builder.getI32IntegerAttr(3)),  // CF
    builder.getNamedAttr("clear,setLatency", builder.getI32IntegerAttr(3)),  // CF
    builder.getNamedAttr("getLatency,read", builder.getI32IntegerAttr(3)),   // CF
    builder.getNamedAttr("read,getLatency", builder.getI32IntegerAttr(3)),   // CF
    builder.getNamedAttr("getLatency,write", builder.getI32IntegerAttr(3)),  // CF
    builder.getNamedAttr("write,getLatency", builder.getI32IntegerAttr(3)),  // CF
    builder.getNamedAttr("getLatency,clear", builder.getI32IntegerAttr(3)),  // CF
    builder.getNamedAttr("clear,getLatency", builder.getI32IntegerAttr(3)),  // CF
    builder.getNamedAttr("getLatency,setLatency", builder.getI32IntegerAttr(0)), // SB
    builder.getNamedAttr("setLatency,getLatency", builder.getI32IntegerAttr(1)), // SA
    builder.getNamedAttr("getLatency,getLatency", builder.getI32IntegerAttr(3)), // CF
    builder.getNamedAttr("setLatency,setLatency", builder.getI32IntegerAttr(2))  // C
  });
  
  auto actions = builder.getArrayAttr({
    SymbolRefAttr::get(builder.getContext(), "read"),
    SymbolRefAttr::get(builder.getContext(), "write"),
    SymbolRefAttr::get(builder.getContext(), "setLatency"),
    SymbolRefAttr::get(builder.getContext(), "getLatency"),
    SymbolRefAttr::get(builder.getContext(), "clear")
  });
  
  builder.create<::sharp::txn::ScheduleOp>(loc, actions, conflictMatrix);
  
  // Mark as spec primitive with software semantics
  primitive->setAttr("spec", builder.getUnitAttr());
  primitive->setAttr("default_latency", builder.getI32IntegerAttr(defaultLatency));
  primitive->setAttr("software_semantics", builder.getDictionaryAttr({
    builder.getNamedAttr("state", builder.getStringAttr(R"(
    std::unordered_map<int32_t, int64_t> memory_data;
    int32_t read_latency = 1;
    static constexpr size_t MEMORY_SIZE = 65536; // 64K entries
    
    // For latency simulation in TL mode
    struct ReadRequest {
        int32_t addr;
        int cycles_remaining;
    };
    std::queue<ReadRequest> pending_reads;
)")),
    builder.getNamedAttr("read", builder.getStringAttr(R"(
    int64_t read(int32_t addr) {
        // In TL simulation, we model latency by returning after configured cycles
        // This is a simplified model - real implementation would use event scheduling
        if (addr >= 0 && addr < MEMORY_SIZE) {
            auto it = memory_data.find(addr);
            return (it != memory_data.end()) ? it->second : 0;
        }
        return 0;
    }
)")),
    builder.getNamedAttr("write", builder.getStringAttr(R"(
    void write(int32_t addr, int64_t data) {
        if (addr >= 0 && addr < MEMORY_SIZE) {
            memory_data[addr] = data;
        }
    }
)")),
    builder.getNamedAttr("setLatency", builder.getStringAttr(R"(
    void setLatency(int32_t latency) {
        if (latency > 0) {
            read_latency = latency;
        }
    }
)")),
    builder.getNamedAttr("getLatency", builder.getStringAttr(R"(
    int32_t getLatency() {
        return read_latency;
    }
)")),
    builder.getNamedAttr("clear", builder.getStringAttr(R"(
    void clear() {
        memory_data.clear();
    }
)"))
  }));
  
  return primitive;
}

} // namespace txn
} // namespace sharp
} // namespace mlir