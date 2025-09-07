set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name BLIR)
add_mlir_dialect_tablegen_target(MLIRBLIRTransformsIncGen)

add_mlir_doc(Passes BLIRPasses ./ -gen-pass-doc)
