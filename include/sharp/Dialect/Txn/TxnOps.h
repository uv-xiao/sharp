//===- TxnOps.h - Txn dialect operations ----------------------------------===//
//
// Part of the Sharp Project.
//
//===----------------------------------------------------------------------===//

#ifndef SHARP_DIALECT_TXN_OPS_H
#define SHARP_DIALECT_TXN_OPS_H

// Include required headers from MLIR
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

// Include the dialect and types
#include "sharp/Dialect/Txn/TxnDialect.h"
#include "sharp/Dialect/Txn/TxnTypes.h"

namespace sharp {
namespace txn {

// Forward declarations needed by generated code
class AbortOp;
class ActionMethodOp;
class CallOp;
class ClockByOp;
class FirActionMethodOp;
class FirValueMethodOp;
class FutureOp;
class IfOp;
class InstanceOp;
class LaunchOp;
class ModuleOp;
class PrimitiveOp;
class ResetByOp;
class ReturnOp;
class RuleOp;
class ScheduleOp;
class ValueMethodOp;
class YieldOp;

} // namespace txn
} // namespace sharp

// Get the generated operation definitions
#define GET_OP_CLASSES
#include "sharp/Dialect/Txn/Txn.h.inc"

#endif // SHARP_DIALECT_TXN_OPS_H