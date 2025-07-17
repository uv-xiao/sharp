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
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
namespace sharp {
namespace txn {

::sharp::txn::PrimitiveOp createSpecFIFOPrimitive(OpBuilder &builder, Location loc,
                                                  StringRef name, Type dataType,
                                                  ArrayAttr typeArgs) {
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
  
  // Store the original full name - spec primitives don't need firrtl.impl
  primitive->setAttr("full_name", StringAttr::get(builder.getContext(), fullName));
  
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
  
  // Create enqueue method with body
  auto enqueueMethod = builder.create<::sharp::txn::ActionMethodOp>(
      loc, builder.getStringAttr("enqueue"), TypeAttr::get(enqueueType),
      /*sym_visibility=*/StringAttr(), /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr(),
      /*ready=*/StringAttr(), /*enable=*/StringAttr(),
      /*result=*/StringAttr(), /*prefix=*/StringAttr(),
      /*always_ready=*/UnitAttr(), /*always_enable=*/UnitAttr(), /*guardCount=*/0);
  Block *enqueueBody = &enqueueMethod.getBody().emplaceBlock();
  enqueueBody->addArgument(dataType, loc);  // Add data argument
  builder.setInsertionPointToEnd(enqueueBody);
  builder.create<::sharp::txn::YieldOp>(loc);
  
  // Create dequeue method with body
  builder.setInsertionPointAfter(enqueueMethod);
  auto dequeueMethod = builder.create<::sharp::txn::ActionMethodOp>(
      loc, builder.getStringAttr("dequeue"), TypeAttr::get(dequeueType),
      /*sym_visibility=*/StringAttr(), /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr(),
      /*ready=*/StringAttr(), /*enable=*/StringAttr(),
      /*result=*/StringAttr(), /*prefix=*/StringAttr(),
      /*always_ready=*/UnitAttr(), /*always_enable=*/UnitAttr(), /*guardCount=*/0);
  Block *dequeueBody = &dequeueMethod.getBody().emplaceBlock();
  builder.setInsertionPointToEnd(dequeueBody);
  auto dummyDeq = builder.create<arith::ConstantOp>(loc, dataType, builder.getIntegerAttr(dataType, 0));
  builder.create<::sharp::txn::ReturnOp>(loc, dummyDeq.getResult());

  // Create isEmpty method with body
  builder.setInsertionPointAfter(dequeueMethod);
  auto isEmptyMethod = builder.create<::sharp::txn::ValueMethodOp>(
      loc, builder.getStringAttr("isEmpty"), TypeAttr::get(isEmptyType),
      /*sym_visibility=*/StringAttr(), /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr(),
      /*result=*/StringAttr(), /*prefix=*/StringAttr());
  Block *isEmptyBody = &isEmptyMethod.getBody().emplaceBlock();
  builder.setInsertionPointToEnd(isEmptyBody);
  auto dummyEmpty = builder.create<arith::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(false));
  builder.create<::sharp::txn::ReturnOp>(loc, dummyEmpty.getResult());

  // Create size method with body
  builder.setInsertionPointAfter(isEmptyMethod);
  auto sizeMethod = builder.create<::sharp::txn::ValueMethodOp>(
      loc, builder.getStringAttr("size"), TypeAttr::get(sizeType),
      /*sym_visibility=*/StringAttr(), /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr(),
      /*result=*/StringAttr(), /*prefix=*/StringAttr());
  Block *sizeBody = &sizeMethod.getBody().emplaceBlock();
  builder.setInsertionPointToEnd(sizeBody);
  auto dummySize = builder.create<arith::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(0));
  builder.create<::sharp::txn::ReturnOp>(loc, dummySize.getResult());

  // Create peek method with body
  builder.setInsertionPointAfter(sizeMethod);
  auto peekMethod = builder.create<::sharp::txn::ValueMethodOp>(
      loc, builder.getStringAttr("peek"), TypeAttr::get(peekType),
      /*sym_visibility=*/StringAttr(), /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr(),
      /*result=*/StringAttr(), /*prefix=*/StringAttr());
  Block *peekBody = &peekMethod.getBody().emplaceBlock();
  builder.setInsertionPointToEnd(peekBody);
  auto dummyPeek = builder.create<arith::ConstantOp>(loc, dataType, builder.getIntegerAttr(dataType, 0));
  builder.create<::sharp::txn::ReturnOp>(loc, dummyPeek.getResult());
  
  // Reset insertion point to primitive body for schedule
  builder.setInsertionPointToEnd(body);
  
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