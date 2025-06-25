add_mlir_dialect_library(SharpCore
  CoreDialect.cpp
  CoreOps.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/sharp/Dialect/Core
  
  DEPENDS
  SharpCoreIncGen
  
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)