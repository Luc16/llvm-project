add_mlir_dialect_library(MLIRBLIRDialect
	BLIRDialect.cpp
	BLIROps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/BLIR

  DEPENDS
  MLIRBLIROpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRSideEffectInterfaces
  )
