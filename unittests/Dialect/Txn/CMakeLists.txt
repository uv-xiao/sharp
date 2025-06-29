add_sharp_unittest(SharpTxnPrimitivesTest
  PrimitivesTest.cpp
)

target_link_libraries(SharpTxnPrimitivesTest
  PRIVATE
  SharpTxn
  SharpTxnPrimitives
  MLIRIR
  MLIRFuncDialect
  MLIRArithDialect
  CIRCTHW
  CIRCTSeq
  CIRCTFIRRTL
)