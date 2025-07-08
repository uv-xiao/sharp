# FIRRTL Operations Guide

## Overview

Quick reference for CIRCT's FIRRTL dialect operations in C++.

## Headers
```cpp
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
```

## Types
```cpp
// Basic types
auto uint32 = UIntType::get(ctx, 32);
auto sint16 = SIntType::get(ctx, 16);
auto clock = ClockType::get(ctx);

// Bundle (struct)
SmallVector<BundleType::BundleElement> elements;
elements.push_back({"data", false, uint32});
elements.push_back({"valid", false, UIntType::get(ctx, 1)});
auto bundle = BundleType::get(ctx, elements);

// Vector (array)
auto vec = FVectorType::get(UIntType::get(ctx, 8), 16);
```

## Module Creation
```cpp
// Define ports
SmallVector<PortInfo> ports;
ports.push_back({builder.getStringAttr("clock"), ClockType::get(ctx), Direction::In});
ports.push_back({builder.getStringAttr("data"), UIntType::get(ctx, 32), Direction::In});

// Create module
auto module = builder.create<FModuleOp>(loc, 
    builder.getStringAttr("MyModule"),
    ConventionAttr::get(ctx, Convention::Internal),
    ports, ArrayAttr{});
```

## Common Operations
```cpp
// Constants
auto c42 = builder.create<ConstantOp>(loc, UIntType::get(ctx, 32), APInt(32, 42));

// Wire/Register
auto wire = builder.create<WireOp>(loc, UIntType::get(ctx, 32), "my_wire");
auto reg = builder.create<RegOp>(loc, UIntType::get(ctx, 32), clock);

// Connect
builder.create<ConnectOp>(loc, wire, value);

// Arithmetic
auto add = builder.create<AddPrimOp>(loc, a, b);
auto and = builder.create<AndPrimOp>(loc, a, b);

// Comparison
auto eq = builder.create<EQPrimOp>(loc, a, b);
auto lt = builder.create<LTPrimOp>(loc, a, b);

// Mux
auto mux = builder.create<MuxPrimOp>(loc, condition, trueVal, falseVal);

// Instance
auto inst = builder.create<InstanceOp>(loc, 
    moduleType, "inst_name", "ModuleName",
    /*lowerToBind=*/false, /*annotations=*/ArrayAttr{});
```

## Control Flow
```cpp
// When (conditional connect)
auto when = builder.create<WhenOp>(loc, condition, /*hasElse=*/true);
builder.setInsertionPointToStart(&when.getThenBlock());
builder.create<ConnectOp>(loc, wire, value1);
builder.setInsertionPointToStart(&when.getElseBlock());
builder.create<ConnectOp>(loc, wire, value2);
```

## Type Conversion
```cpp
// Width adjustment
auto pad = builder.create<PadPrimOp>(loc, narrowValue, desiredWidth);
auto bits = builder.create<BitsPrimOp>(loc, wideValue, highBit, lowBit);

// Sign conversion
auto toUInt = builder.create<AsUIntPrimOp>(loc, sintValue);
auto toSInt = builder.create<AsSIntPrimOp>(loc, uintValue);
```

## Common Patterns

### Creating a Register with Reset
```cpp
auto reg = builder.create<RegResetOp>(loc, 
    UIntType::get(ctx, 32),  // Type
    clock,                    // Clock
    reset,                    // Reset signal
    resetValue);              // Reset value
```

### Accessing Bundle Fields
```cpp
auto field = builder.create<SubfieldOp>(loc, bundle, "fieldName");
```

### Creating Parameterized Module
```cpp
// In module attributes
auto params = builder.getDictionaryAttr({
    {builder.getStringAttr("WIDTH"), builder.getI32IntegerAttr(32)},
    {builder.getStringAttr("DEPTH"), builder.getI32IntegerAttr(16)}
});
```