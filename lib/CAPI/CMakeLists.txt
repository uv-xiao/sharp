add_mlir_public_c_api_library(SharpCAPICore
  Core.cpp
  
  LINK_LIBS PUBLIC
    SharpCore
    MLIRCAPIIR
)