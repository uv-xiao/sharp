//===- SpecFIFO.cpp - Sharp Txn SpecFIFO primitive implementation --------===//
//
// Part of the Sharp Project.
// SpecFIFO is an unbounded FIFO for specification and verification.
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

::sharp::txn::PrimitiveOp createSpecFIFOPrimitive(OpBuilder &builder, Location loc,
                                                  StringRef name, Type dataType) {
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
  auto enqueueType = builder.getFunctionType({dataType}, {});
  auto dequeueType = builder.getFunctionType({}, {dataType});
  auto isEmptyType = builder.getFunctionType({}, {builder.getI1Type()});
  auto sizeType = builder.getFunctionType({}, {builder.getI32Type()});
  auto peekType = builder.getFunctionType({}, {dataType});
  
  // Create methods
  builder.create<::sharp::txn::ActionMethodOp>(
      loc, builder.getStringAttr("enqueue"), TypeAttr::get(enqueueType),
      /*sym_visibility=*/StringAttr(), /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr(),
      /*timing=*/StringAttr(), /*ready=*/StringAttr(), /*enable=*/StringAttr(),
      /*result=*/StringAttr(), /*prefix=*/StringAttr(),
      /*always_ready=*/UnitAttr(), /*always_enable=*/UnitAttr());
  
  builder.create<::sharp::txn::ActionMethodOp>(
      loc, builder.getStringAttr("dequeue"), TypeAttr::get(dequeueType),
      /*sym_visibility=*/StringAttr(), /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr(),
      /*timing=*/StringAttr(), /*ready=*/StringAttr(), /*enable=*/StringAttr(),
      /*result=*/StringAttr(), /*prefix=*/StringAttr(),
      /*always_ready=*/UnitAttr(), /*always_enable=*/UnitAttr());
  
  builder.create<::sharp::txn::ValueMethodOp>(
      loc, builder.getStringAttr("isEmpty"), TypeAttr::get(isEmptyType),
      /*sym_visibility=*/StringAttr(), /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr(),
      /*timing=*/builder.getStringAttr("combinational"), /*result=*/StringAttr(), /*prefix=*/StringAttr());
  
  builder.create<::sharp::txn::ValueMethodOp>(
      loc, builder.getStringAttr("size"), TypeAttr::get(sizeType),
      /*sym_visibility=*/StringAttr(), /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr(),
      /*timing=*/builder.getStringAttr("combinational"), /*result=*/StringAttr(), /*prefix=*/StringAttr());
  
  builder.create<::sharp::txn::ValueMethodOp>(
      loc, builder.getStringAttr("peek"), TypeAttr::get(peekType),
      /*sym_visibility=*/StringAttr(), /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr(),
      /*timing=*/builder.getStringAttr("combinational"), /*result=*/StringAttr(), /*prefix=*/StringAttr());
  
  // Create schedule with conflict matrix
  auto conflictMatrix = builder.getDictionaryAttr({
    builder.getNamedAttr("enqueue,enqueue", builder.getI32IntegerAttr(0)), // SB
    builder.getNamedAttr("dequeue,dequeue", builder.getI32IntegerAttr(0)), // SB
    builder.getNamedAttr("enqueue,dequeue", builder.getI32IntegerAttr(0)), // SB
    builder.getNamedAttr("dequeue,enqueue", builder.getI32IntegerAttr(1)), // SA
    builder.getNamedAttr("isEmpty,enqueue", builder.getI32IntegerAttr(0)), // SB
    builder.getNamedAttr("enqueue,isEmpty", builder.getI32IntegerAttr(1)), // SA
    builder.getNamedAttr("isEmpty,dequeue", builder.getI32IntegerAttr(0)), // SB
    builder.getNamedAttr("dequeue,isEmpty", builder.getI32IntegerAttr(1)), // SA
    builder.getNamedAttr("peek,dequeue", builder.getI32IntegerAttr(0)),    // SB
    builder.getNamedAttr("dequeue,peek", builder.getI32IntegerAttr(1)),    // SA
    builder.getNamedAttr("peek,enqueue", builder.getI32IntegerAttr(3)),    // CF
    builder.getNamedAttr("enqueue,peek", builder.getI32IntegerAttr(3)),    // CF
    builder.getNamedAttr("size,enqueue", builder.getI32IntegerAttr(0)),    // SB
    builder.getNamedAttr("enqueue,size", builder.getI32IntegerAttr(1)),    // SA
    builder.getNamedAttr("size,dequeue", builder.getI32IntegerAttr(0)),    // SB
    builder.getNamedAttr("dequeue,size", builder.getI32IntegerAttr(1)),    // SA
    builder.getNamedAttr("peek,peek", builder.getI32IntegerAttr(3)),       // CF
    builder.getNamedAttr("size,size", builder.getI32IntegerAttr(3)),       // CF
    builder.getNamedAttr("isEmpty,isEmpty", builder.getI32IntegerAttr(3)), // CF
    builder.getNamedAttr("isEmpty,size", builder.getI32IntegerAttr(3)),    // CF
    builder.getNamedAttr("size,isEmpty", builder.getI32IntegerAttr(3)),    // CF
    builder.getNamedAttr("peek,size", builder.getI32IntegerAttr(3)),       // CF
    builder.getNamedAttr("size,peek", builder.getI32IntegerAttr(3)),       // CF
    builder.getNamedAttr("peek,isEmpty", builder.getI32IntegerAttr(3)),    // CF
    builder.getNamedAttr("isEmpty,peek", builder.getI32IntegerAttr(3))     // CF
  });
  
  auto actions = builder.getArrayAttr({
    SymbolRefAttr::get(builder.getContext(), "enqueue"),
    SymbolRefAttr::get(builder.getContext(), "dequeue"),
    SymbolRefAttr::get(builder.getContext(), "isEmpty"),
    SymbolRefAttr::get(builder.getContext(), "size"),
    SymbolRefAttr::get(builder.getContext(), "peek")
  });
  
  builder.create<::sharp::txn::ScheduleOp>(loc, actions, conflictMatrix);
  
  // Mark as spec primitive with software semantics
  primitive->setAttr("spec", builder.getUnitAttr());
  primitive->setAttr("software_semantics", builder.getDictionaryAttr({
    builder.getNamedAttr("state", builder.getStringAttr(R"(
    std::queue<int64_t> fifo_data;
)")),
    builder.getNamedAttr("enqueue", builder.getStringAttr(R"(
    void enqueue(int64_t data) {
        fifo_data.push(data);
    }
)")),
    builder.getNamedAttr("dequeue", builder.getStringAttr(R"(
    int64_t dequeue() {
        if (!fifo_data.empty()) {
            int64_t val = fifo_data.front();
            fifo_data.pop();
            return val;
        }
        return 0; // Default value when empty
    }
)")),
    builder.getNamedAttr("isEmpty", builder.getStringAttr(R"(
    bool isEmpty() {
        return fifo_data.empty();
    }
)")),
    builder.getNamedAttr("size", builder.getStringAttr(R"(
    int32_t size() {
        return static_cast<int32_t>(fifo_data.size());
    }
)")),
    builder.getNamedAttr("peek", builder.getStringAttr(R"(
    int64_t peek() {
        if (!fifo_data.empty()) {
            return fifo_data.front();
        }
        return 0; // Default value when empty
    }
)"))
  }));
  
  return primitive;
}

} // namespace txn
} // namespace sharp
} // namespace mlir