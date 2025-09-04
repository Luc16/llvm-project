add_mlir_dialect_library(MLIRBLIRDialect
	BLIRDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/BLIR

  DEPENDS
  MLIRBLIRIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRSideEffectInterfaces
  )
