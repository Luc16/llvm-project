#ifndef LIB_DIALECT_BLIR_TRANSFORM_PASSES_TD_
#define LIB_DIALECT_BLIR_TRANSFORM_PASSES_TD_

include "mlir/Pass/PassBase.td"

def BLIRMatMulToFuncPass : Pass<"blir-matmul-to-func", "func::FuncOp"> {
  let summary = "Convert BLIR matrix multiplication operations to loop nests";
  let description = [{
    This pass identifies BLIR matrix multiplication operations and lowers them
    to explicit affine loop nests, enabling further optimizations and transformations.
  }];
  let dependentDialects = ["affine::AffineDialect", "arith::ArithDialect", "memref::MemRefDialect"];
}



#endif // LIB_DIALECT_BLIR_TRANSFORM_PASSES_TD_
