set(LLVM_TARGET_DEFINITIONS ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/BLIR/Transforms/Passes.td)
mlir_tablegen(${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/BLIR/Transforms/Passes.h.inc -gen-pass-decls -name BLIR)
add_public_tablegen_target(MLIRBLIRPassIncGen)

add_mlir_dialect_library(MLIRBLIRTransforms
  MatMulToLoops.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/BLIR

  DEPENDS
  MLIRBLIRPassIncGen

  LINK_LIBS PUBLIC
  MLIRBLIRDialect
  MLIRAffineDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRPass
  MLIRTransforms
)
