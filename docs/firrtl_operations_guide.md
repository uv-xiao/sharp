# FIRRTL Dialect Operations Guide

This document provides a reference for using CIRCT's FIRRTL dialect operations in C++ code.

## Overview

FIRRTL (Flexible Intermediate Representation for RTL) is CIRCT's dialect for hardware description. When converting from Sharp Txn to FIRRTL, we need to use CIRCT's C++ APIs to create FIRRTL operations programmatically.

## Common Headers

```cpp
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
```

## Type System

### Basic Types

```cpp
// Integer types
auto uint1Type = UIntType::get(ctx, 1);      // 1-bit unsigned
auto uint32Type = UIntType::get(ctx, 32);    // 32-bit unsigned
auto sint16Type = SIntType::get(ctx, 16);    // 16-bit signed

// Clock type
auto clockType = ClockType::get(ctx);

// Reset type (typically UInt<1>)
auto resetType = UIntType::get(ctx, 1);

// Analog type for inout ports
auto analogType = AnalogType::get(ctx, width);
```

### Aggregate Types

```cpp
// Bundle type (struct-like)
SmallVector<BundleType::BundleElement> elements;
elements.push_back({"data", false, UIntType::get(ctx, 32)});
elements.push_back({"valid", false, UIntType::get(ctx, 1)});
auto bundleType = BundleType::get(ctx, elements);

// Vector type (array-like)
auto vecType = FVectorType::get(UIntType::get(ctx, 8), 16); // 16 x UInt<8>
```

## Module Creation

### Creating a FIRRTL Module

```cpp
// Define ports
SmallVector<PortInfo> ports;

// Add clock and reset
ports.push_back({builder.getStringAttr("clock"), 
                 ClockType::get(ctx), Direction::In});
ports.push_back({builder.getStringAttr("reset"), 
                 UIntType::get(ctx, 1), Direction::In});

// Add data ports
ports.push_back({builder.getStringAttr("in_data"), 
                 UIntType::get(ctx, 32), Direction::In});
ports.push_back({builder.getStringAttr("out_data"), 
                 UIntType::get(ctx, 32), Direction::Out});

// Create module
auto firrtlModule = builder.create<FModuleOp>(
    loc, 
    builder.getStringAttr("MyModule"),
    ConventionAttr::get(ctx, Convention::Internal),
    ports,
    /*annotations=*/ArrayAttr{});

// Set insertion point inside module
builder.setInsertionPointToStart(firrtlModule.getBodyBlock());
```

## Common Operations

### Constants

```cpp
// Integer constant
auto const1 = builder.create<ConstantOp>(loc, 
    UIntType::get(ctx, 32), APInt(32, 42));

// Invalid value (uninitialized)
auto invalid = builder.create<InvalidValueOp>(loc, 
    UIntType::get(ctx, 32));
```

### Wires and Nodes

```cpp
// Wire (mutable)
auto wire = builder.create<WireOp>(loc, 
    UIntType::get(ctx, 32), "my_wire");

// Node (immutable, for expressions)
auto node = builder.create<NodeOp>(loc, 
    someValue, builder.getStringAttr("my_node"));
```

### Registers

```cpp
// Simple register
auto reg = builder.create<RegOp>(loc, 
    UIntType::get(ctx, 32), clockSignal, "my_reg");

// Register with reset
auto regReset = builder.create<RegResetOp>(loc,
    UIntType::get(ctx, 32), clockSignal, resetSignal, 
    resetValue, "my_reg_reset");
```

### Connections

```cpp
// Connect (strong connect, types must match exactly)
builder.create<ConnectOp>(loc, destination, source);

// Partial connect (weak connect, allows width mismatches)
builder.create<PartialConnectOp>(loc, destination, source);
```

### Arithmetic Operations

```cpp
// Addition
auto sum = builder.create<AddPrimOp>(loc, lhs, rhs);

// Subtraction  
auto diff = builder.create<SubPrimOp>(loc, lhs, rhs);

// Multiplication
auto prod = builder.create<MulPrimOp>(loc, lhs, rhs);

// Division
auto quot = builder.create<DivPrimOp>(loc, lhs, rhs);

// Remainder
auto rem = builder.create<RemPrimOp>(loc, lhs, rhs);
```

### Comparison Operations

```cpp
// Less than
auto lt = builder.create<LTPrimOp>(loc, lhs, rhs);

// Less than or equal
auto leq = builder.create<LEQPrimOp>(loc, lhs, rhs);

// Greater than
auto gt = builder.create<GTPrimOp>(loc, lhs, rhs);

// Greater than or equal
auto geq = builder.create<GEQPrimOp>(loc, lhs, rhs);

// Equal
auto eq = builder.create<EQPrimOp>(loc, lhs, rhs);

// Not equal
auto neq = builder.create<NEQPrimOp>(loc, lhs, rhs);
```

### Logical Operations

```cpp
// AND
auto andOp = builder.create<AndPrimOp>(loc, lhs, rhs);

// OR
auto orOp = builder.create<OrPrimOp>(loc, lhs, rhs);

// XOR
auto xorOp = builder.create<XorPrimOp>(loc, lhs, rhs);

// NOT
auto notOp = builder.create<NotPrimOp>(loc, value);
```

### Bit Operations

```cpp
// Extract bits [hi:lo]
auto bits = builder.create<BitsPrimOp>(loc, value, hi, lo);

// Concatenate
auto cat = builder.create<CatPrimOp>(loc, lhs, rhs);

// Shift left
auto shl = builder.create<ShlPrimOp>(loc, value, amount);

// Shift right
auto shr = builder.create<ShrPrimOp>(loc, value, amount);
```

### Control Flow

```cpp
// When (if-then)
builder.create<WhenOp>(loc, condition, /*hasElse=*/false, 
    [&]() {
        // Then block
        builder.create<ConnectOp>(loc, wire, value1);
    });

// When with else
builder.create<WhenOp>(loc, condition, /*hasElse=*/true,
    [&]() {
        // Then block
        builder.create<ConnectOp>(loc, wire, value1);
    },
    [&]() {
        // Else block  
        builder.create<ConnectOp>(loc, wire, value2);
    });
```

### Instances

```cpp
// Create instance of another module
SmallVector<Attribute> portNames;
SmallVector<Direction> portDirections;
SmallVector<Type> portTypes;
// ... populate port info ...

auto inst = builder.create<InstanceOp>(loc,
    portTypes, builder.getStringAttr("inst_name"),
    FlatSymbolRefAttr::get(ctx, "ModuleName"),
    portNames, portDirections,
    /*annotations=*/ArrayAttr{},
    /*portAnnotations=*/ArrayAttr{},
    /*lowerToBind=*/false,
    /*innerSym=*/StringAttr{});

// Access instance ports
Value clockPort = inst.getResult(0);  // By index
```

## Common Patterns

### Creating a Register with Enable

```cpp
// Manual implementation of enabled register
auto wire = builder.create<WireOp>(loc, dataType, "reg_next");
auto reg = builder.create<RegOp>(loc, dataType, clock, "my_reg");

// When enable is high, update the register
builder.create<WhenOp>(loc, enable, false, [&]() {
    builder.create<ConnectOp>(loc, wire, newValue);
}, [&]() {
    builder.create<ConnectOp>(loc, wire, reg);
});

builder.create<ConnectOp>(loc, reg, wire);
```

### Mux Pattern

```cpp
// Create a mux using when/else
auto muxWire = builder.create<WireOp>(loc, dataType, "mux_out");

builder.create<WhenOp>(loc, selector, true,
    [&]() {
        builder.create<ConnectOp>(loc, muxWire, trueValue);
    },
    [&]() {
        builder.create<ConnectOp>(loc, muxWire, falseValue);
    });

// Or use the mux primitive
auto muxResult = builder.create<MuxPrimOp>(loc, selector, trueValue, falseValue);
```

### Module Port Access

```cpp
// In a FIRRTL module, ports are block arguments
auto module = builder.create<FModuleOp>(...);
Block *body = module.getBodyBlock();

// Access ports by index
Value clock = body->getArgument(0);
Value reset = body->getArgument(1);
Value inputData = body->getArgument(2);

// Or iterate through ports
for (auto [port, arg] : llvm::zip(module.getPorts(), body->getArguments())) {
    if (port.name == "clock") {
        Value clockPort = arg;
    }
}
```

## Type Conversion from MLIR

```cpp
// Convert MLIR integer type to FIRRTL
FIRRTLType convertType(Type type) {
    if (auto intType = type.dyn_cast<IntegerType>()) {
        // Use UIntType for unsigned semantics
        return UIntType::get(type.getContext(), intType.getWidth());
    }
    // Handle other types...
    return nullptr;
}
```

## Best Practices

1. **Always use builders** - Don't create operations directly
2. **Set proper names** - Use descriptive names for wires, nodes, and registers
3. **Handle clock and reset properly** - Ensure all sequential elements have proper clock/reset
4. **Use appropriate connection types** - Connect for exact matches, PartialConnect for width mismatches
5. **Create nodes for expressions** - This helps with debugging and readability
6. **Check types carefully** - FIRRTL has strict typing rules

## Example: Counter Module

```cpp
// Create a simple counter module
auto counterModule = builder.create<FModuleOp>(loc, "Counter", 
    ConventionAttr::get(ctx, Convention::Internal), ports, ArrayAttr{});

builder.setInsertionPointToStart(counterModule.getBodyBlock());

// Get ports
Value clock = counterModule.getBodyBlock()->getArgument(0);
Value reset = counterModule.getBodyBlock()->getArgument(1); 
Value enable = counterModule.getBodyBlock()->getArgument(2);
Value count_out = counterModule.getBodyBlock()->getArgument(3);

// Create counter register
auto counter = builder.create<RegResetOp>(loc,
    UIntType::get(ctx, 32), clock, reset,
    builder.create<ConstantOp>(loc, UIntType::get(ctx, 32), APInt(32, 0)),
    "counter");

// Increment logic
auto one = builder.create<ConstantOp>(loc, UIntType::get(ctx, 32), APInt(32, 1));
auto incremented = builder.create<AddPrimOp>(loc, counter, one);

// Update counter when enabled
builder.create<WhenOp>(loc, enable, false, [&]() {
    builder.create<ConnectOp>(loc, counter, incremented);
});

// Connect output
builder.create<ConnectOp>(loc, count_out, counter);
```

## References

- [CIRCT FIRRTL Dialect Documentation](https://circt.llvm.org/docs/Dialects/FIRRTL/)
- [FIRRTL Specification](https://github.com/chipsalliance/firrtl-spec)
- [CIRCT FIRRTL Operations](https://github.com/llvm/circt/blob/main/include/circt/Dialect/FIRRTL/FIRRTLOps.td)