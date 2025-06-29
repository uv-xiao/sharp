//===- PrimitivesTest.cpp - Tests for Txn primitive constructors ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sharp/Dialect/Txn/TxnPrimitives.h"
#include "sharp/Dialect/Txn/TxnOps.h"
#include "sharp/Dialect/Txn/TxnDialect.h"

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "gtest/gtest.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

class TxnPrimitivesTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<::sharp::txn::TxnDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<circt::firrtl::FIRRTLDialect>();
    context.loadDialect<circt::hw::HWDialect>();
    context.loadDialect<circt::seq::SeqDialect>();
  }

  MLIRContext context;
};

TEST_F(TxnPrimitivesTest, CreateRegisterPrimitive) {
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  
  // Create a module to hold the primitive
  auto module = builder.create<ModuleOp>(loc);
  builder.setInsertionPointToStart(module.getBody());
  
  // Create Register primitive
  auto i32Type = builder.getI32Type();
  auto regPrimitive = ::mlir::sharp::txn::createRegisterPrimitive(builder, loc, "TestRegister", i32Type);
  
  // Verify the primitive was created correctly
  ASSERT_TRUE(regPrimitive);
  EXPECT_EQ(regPrimitive.getSymName(), "TestRegister");
  EXPECT_EQ(regPrimitive.getType(), "hw");
  
  // Check that it has the correct operations
  bool hasFirValueMethod = false;
  bool hasFirActionMethod = false;
  bool hasClockBy = false;
  bool hasResetBy = false;
  bool hasSchedule = false;
  
  regPrimitive.getBody().walk([&](Operation *op) {
    if (isa<::sharp::txn::FirValueMethodOp>(op))
      hasFirValueMethod = true;
    else if (isa<::sharp::txn::FirActionMethodOp>(op))
      hasFirActionMethod = true;
    else if (isa<::sharp::txn::ClockByOp>(op))
      hasClockBy = true;
    else if (isa<::sharp::txn::ResetByOp>(op))
      hasResetBy = true;
    else if (isa<::sharp::txn::ScheduleOp>(op))
      hasSchedule = true;
  });
  
  EXPECT_TRUE(hasFirValueMethod) << "Register should have fir_value_method";
  EXPECT_TRUE(hasFirActionMethod) << "Register should have fir_action_method";
  EXPECT_TRUE(hasClockBy) << "Register should have clock_by";
  EXPECT_TRUE(hasResetBy) << "Register should have reset_by";
  EXPECT_TRUE(hasSchedule) << "Register should have schedule";
  
  // Check the primitive has reference to FIRRTL implementation
  auto firrtlImplAttr = regPrimitive->getAttrOfType<StringAttr>("firrtl.impl");
  ASSERT_TRUE(firrtlImplAttr) << "Register should have firrtl.impl attribute";
  EXPECT_EQ(firrtlImplAttr.getValue(), "TestRegister_impl");
  
  // Check the methods have port mapping attributes
  regPrimitive.getBody().walk([&](::sharp::txn::FirValueMethodOp readOp) {
    if (readOp.getSymName() == "read") {
      auto portAttr = readOp->getAttrOfType<StringAttr>("firrtl.port");
      ASSERT_TRUE(portAttr) << "Read method should have firrtl.port attribute";
      EXPECT_EQ(portAttr.getValue(), "read_data");
    }
  });
  
  regPrimitive.getBody().walk([&](::sharp::txn::FirActionMethodOp writeOp) {
    if (writeOp.getSymName() == "write") {
      auto dataPortAttr = writeOp->getAttrOfType<StringAttr>("firrtl.data_port");
      auto enablePortAttr = writeOp->getAttrOfType<StringAttr>("firrtl.enable_port");
      ASSERT_TRUE(dataPortAttr) << "Write method should have firrtl.data_port attribute";
      ASSERT_TRUE(enablePortAttr) << "Write method should have firrtl.enable_port attribute";
      EXPECT_EQ(dataPortAttr.getValue(), "write_data");
      EXPECT_EQ(enablePortAttr.getValue(), "write_enable");
    }
  });
  
  // Verify the module
  EXPECT_TRUE(succeeded(verify(module)));
  
  // Create FIRRTL circuit to test FIRRTL module
  auto circuit = builder.create<circt::firrtl::CircuitOp>(loc, builder.getStringAttr("TestCircuit"));
  builder.setInsertionPointToStart(circuit.getBodyBlock());
  
  // Create main module with same name as circuit (FIRRTL requirement)
  SmallVector<circt::firrtl::PortInfo> mainPorts;
  auto mainModule = builder.create<circt::firrtl::FModuleOp>(
      loc, builder.getStringAttr("TestCircuit"),
      circt::firrtl::ConventionAttr::get(builder.getContext(), 
                                          circt::firrtl::Convention::Internal),
      mainPorts);
  
  // Create FIRRTL module
  auto firrtlModule = ::mlir::sharp::txn::createRegisterFIRRTLModule(builder, loc, "TestRegister", i32Type);
  ASSERT_TRUE(firrtlModule);
  
  // Verify FIRRTL module structure
  EXPECT_EQ(firrtlModule.getModuleName(), "TestRegister_impl");
  EXPECT_EQ(firrtlModule.getNumPorts(), 6u); // clock, reset, read_data, read_enable, write_data, write_enable
  
  // Verify the circuit
  EXPECT_TRUE(succeeded(verify(circuit)));
  
  // Print for debugging
  llvm::errs() << "Register primitive:\n";
  regPrimitive.print(llvm::errs());
  llvm::errs() << "\n\nRegister FIRRTL module:\n";
  firrtlModule.print(llvm::errs());
  llvm::errs() << "\n";
}

TEST_F(TxnPrimitivesTest, CreateWirePrimitive) {
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  
  // Create a module to hold the primitive
  auto module = builder.create<ModuleOp>(loc);
  builder.setInsertionPointToStart(module.getBody());
  
  // Create Wire primitive
  auto i32Type = builder.getI32Type();
  auto wirePrimitive = ::mlir::sharp::txn::createWirePrimitive(builder, loc, "TestWire", i32Type);
  
  // Verify the primitive was created correctly
  ASSERT_TRUE(wirePrimitive);
  EXPECT_EQ(wirePrimitive.getSymName(), "TestWire");
  EXPECT_EQ(wirePrimitive.getType(), "hw");
  
  // Check conflict matrix for Wire (read SB write)
  ::sharp::txn::ScheduleOp schedule;
  wirePrimitive.getBody().walk([&](::sharp::txn::ScheduleOp op) {
    schedule = op;
  });
  
  ASSERT_TRUE(schedule);
  auto cmAttr = schedule.getConflictMatrix();
  ASSERT_TRUE(cmAttr.has_value());
  
  auto cm = cast<DictionaryAttr>(*cmAttr);
  auto readWriteConflict = cm.get("read,write");
  ASSERT_TRUE(readWriteConflict);
  EXPECT_EQ(cast<IntegerAttr>(readWriteConflict).getInt(), 0); // SB
  
  auto writeReadConflict = cm.get("write,read");
  ASSERT_TRUE(writeReadConflict);
  EXPECT_EQ(cast<IntegerAttr>(writeReadConflict).getInt(), 1); // SA
  
  // Check the primitive has reference to FIRRTL implementation
  auto firrtlImplAttr = wirePrimitive->getAttrOfType<StringAttr>("firrtl.impl");
  ASSERT_TRUE(firrtlImplAttr) << "Wire should have firrtl.impl attribute";
  EXPECT_EQ(firrtlImplAttr.getValue(), "TestWire_impl");
  
  // Check the methods have port mapping attributes
  wirePrimitive.getBody().walk([&](::sharp::txn::FirValueMethodOp readOp) {
    if (readOp.getSymName() == "read") {
      auto portAttr = readOp->getAttrOfType<StringAttr>("firrtl.port");
      ASSERT_TRUE(portAttr) << "Read method should have firrtl.port attribute";
      EXPECT_EQ(portAttr.getValue(), "read_data");
    }
  });
  
  wirePrimitive.getBody().walk([&](::sharp::txn::FirActionMethodOp writeOp) {
    if (writeOp.getSymName() == "write") {
      auto dataPortAttr = writeOp->getAttrOfType<StringAttr>("firrtl.data_port");
      auto enablePortAttr = writeOp->getAttrOfType<StringAttr>("firrtl.enable_port");
      ASSERT_TRUE(dataPortAttr) << "Write method should have firrtl.data_port attribute";
      ASSERT_TRUE(enablePortAttr) << "Write method should have firrtl.enable_port attribute";
      EXPECT_EQ(dataPortAttr.getValue(), "write_data");
      EXPECT_EQ(enablePortAttr.getValue(), "write_enable");
    }
  });
  
  // Verify the module
  EXPECT_TRUE(succeeded(verify(module)));
  
  // Create FIRRTL circuit to test FIRRTL module
  auto circuit = builder.create<circt::firrtl::CircuitOp>(loc, builder.getStringAttr("TestCircuit"));
  builder.setInsertionPointToStart(circuit.getBodyBlock());
  
  // Create main module with same name as circuit (FIRRTL requirement)
  SmallVector<circt::firrtl::PortInfo> mainPorts;
  auto mainModule = builder.create<circt::firrtl::FModuleOp>(
      loc, builder.getStringAttr("TestCircuit"),
      circt::firrtl::ConventionAttr::get(builder.getContext(), 
                                          circt::firrtl::Convention::Internal),
      mainPorts);
  
  // Create FIRRTL module
  auto firrtlModule = ::mlir::sharp::txn::createWireFIRRTLModule(builder, loc, "TestWire", i32Type);
  ASSERT_TRUE(firrtlModule);
  
  // Verify FIRRTL module structure
  EXPECT_EQ(firrtlModule.getModuleName(), "TestWire_impl");
  EXPECT_EQ(firrtlModule.getNumPorts(), 5u); // clock, reset, read_data, write_data, write_enable
  
  // Verify the circuit
  EXPECT_TRUE(succeeded(verify(circuit)));
  
  // Print for debugging
  llvm::errs() << "Wire primitive:\n";
  wirePrimitive.print(llvm::errs());
  llvm::errs() << "\n\nWire FIRRTL module:\n";
  firrtlModule.print(llvm::errs());
  llvm::errs() << "\n";
}