//===- Memory.cpp - Sharp Txn Memory primitive implementation ------------===//
//
// Part of the Sharp Project.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Memory primitive for the Sharp Txn dialect.
// Memory provides address-based storage with read/write operations.
//
//===----------------------------------------------------------------------===//

#include "sharp/Dialect/Txn/TxnPrimitives.h"
#include "sharp/Dialect/Txn/TxnOps.h"
#include "sharp/Dialect/Txn/TxnTypes.h"

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace sharp {
namespace txn {

::sharp::txn::PrimitiveOp createMemoryPrimitive(OpBuilder &builder, Location loc,
                                                StringRef name, Type dataType,
                                                ArrayAttr typeArgs,
                                                unsigned addressWidth) {
  // Generate the full name for the interface type
  std::string fullName = ::sharp::txn::module_name_with_type_args(name, typeArgs);
  std::string legalizedName = ::sharp::txn::legalizeName(fullName);
  
  // Create interface type for the primitive
  auto moduleType = builder.getIndexType();
  
  // Create the primitive operation
  auto primitive = builder.create<::sharp::txn::PrimitiveOp>(loc, 
                                                            StringAttr::get(builder.getContext(), legalizedName),
                                                            /*type_parameters=*/typeArgs,
                                                            /*const_parameters=*/ArrayAttr());
  
  // Store the original full name and set firrtl.impl to legalized name
  primitive->setAttr("full_name", StringAttr::get(builder.getContext(), fullName));
  primitive->setAttr("firrtl.impl", StringAttr::get(builder.getContext(), legalizedName));
  
  // Create a new builder for the primitive body
  OpBuilder::InsertionGuard guard(builder);
  Block *body = &primitive.getBody().emplaceBlock();
  builder.setInsertionPointToStart(body);
  
  // Define the methods
  auto addrType = builder.getIntegerType(addressWidth);
  auto readType = builder.getFunctionType({addrType}, {dataType});
  auto writeType = builder.getFunctionType({addrType, dataType}, {});
  auto clearType = builder.getFunctionType({}, {});
  
  // Create read method (value method)
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
      /*always_ready=*/UnitAttr(), /*always_enable=*/UnitAttr(), /*guardCount=*/0);
  
  // Create clear method (action method)
  builder.create<::sharp::txn::ActionMethodOp>(
      loc, builder.getStringAttr("clear"), TypeAttr::get(clearType),
      /*sym_visibility=*/StringAttr(), /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr(),
      /*ready=*/StringAttr(), /*enable=*/StringAttr(),
      /*result=*/StringAttr(), /*prefix=*/StringAttr(),
      /*always_ready=*/UnitAttr(), /*always_enable=*/UnitAttr(), /*guardCount=*/0);
  
  // Create schedule with conflict matrix
  auto conflictMatrix = builder.getDictionaryAttr({
    builder.getNamedAttr("read,read", builder.getI32IntegerAttr(3)),    // CF
    builder.getNamedAttr("read,write", builder.getI32IntegerAttr(2)),   // C
    builder.getNamedAttr("write,read", builder.getI32IntegerAttr(2)),   // C
    builder.getNamedAttr("write,write", builder.getI32IntegerAttr(2)),  // C
    builder.getNamedAttr("clear,read", builder.getI32IntegerAttr(2)),   // C
    builder.getNamedAttr("read,clear", builder.getI32IntegerAttr(2)),   // C
    builder.getNamedAttr("clear,write", builder.getI32IntegerAttr(2)),  // C
    builder.getNamedAttr("write,clear", builder.getI32IntegerAttr(2)),  // C
    builder.getNamedAttr("clear,clear", builder.getI32IntegerAttr(2))   // C
  });
  
  auto actions = builder.getArrayAttr({
    SymbolRefAttr::get(builder.getContext(), "read"),
    SymbolRefAttr::get(builder.getContext(), "write"),
    SymbolRefAttr::get(builder.getContext(), "clear")
  });
  
  builder.create<::sharp::txn::ScheduleOp>(loc, actions, conflictMatrix);
  
  // Mark as spec primitive with software semantics
  primitive->setAttr("spec", builder.getUnitAttr());
  primitive->setAttr("software_semantics", builder.getDictionaryAttr({
    builder.getNamedAttr("state", builder.getStringAttr(R"(
    std::unordered_map<int32_t, int64_t> memory_data;
    static constexpr size_t MEMORY_SIZE = 1024;
)")),
    builder.getNamedAttr("read", builder.getStringAttr(R"(
    int64_t read(int32_t addr) {
        if (addr >= 0 && addr < MEMORY_SIZE) {
            auto it = memory_data.find(addr);
            return (it != memory_data.end()) ? it->second : 0;
        }
        return 0; // Out of bounds returns 0
    }
)")),
    builder.getNamedAttr("write", builder.getStringAttr(R"(
    void write(int32_t addr, int64_t data) {
        if (addr >= 0 && addr < MEMORY_SIZE) {
            memory_data[addr] = data;
        }
        // Out of bounds writes are ignored
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

circt::firrtl::FModuleOp createMemoryFIRRTLModule(OpBuilder &builder, Location loc,
                                                  StringRef name, Type dataType,
                                                  unsigned addressWidth) {
  // TODO: Implement FIRRTL memory module generation
  // This would create a module with:
  // - Read/write ports with address/data/enable signals
  // - Internal memory array storage
  // - Clear functionality
  // 
  // For now, return nullptr like FIFO does
  return nullptr;
}

} // namespace txn
} // namespace sharp
} // namespace mlir