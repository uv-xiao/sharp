add_mlir_public_c_api_library(SharpCAPICore
  Txn.cpp
  
  LINK_LIBS PUBLIC
    SharpTxn
    MLIRCAPIIR
    # MLIR dialects needed
    MLIRCAPISCF
    MLIRCAPISMT
    MLIRCAPIIndex
    MLIRCAPIArith
    # CIRCT dialects needed
    CIRCTCAPIFIRRTL
    CIRCTCAPIComb
    CIRCTCAPIHWArith
    CIRCTCAPISeq
    CIRCTCAPISV
)