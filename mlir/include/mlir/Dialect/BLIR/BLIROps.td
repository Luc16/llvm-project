#ifndef LIB_DIALECT_BLIR_BLIROps_TD_
#define LIB_DIALECT_BLIR_BLIROps_TD_

include "BLIRDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def BLIR_MatMulOp : Op<BLIR_Dialect, "matmul", []> {
  let summary = "Matrix multiplication operation";
  let description = [{
    This operation performs matrix multiplication on two input matrices.
    The inputs must be 2D tensors of compatible dimensions.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
	AnyType:$output,
    DefaultValuedAttr<BoolAttr, "false">:$transa,
    DefaultValuedAttr<BoolAttr, "false">:$transb,
    DefaultValuedAttr<F32Attr, "1.0">:$alpha,
    DefaultValuedAttr<F32Attr, "0.0">:$beta
  );
  

  // Assembly format que inclui os atributos opcionais
  let assemblyFormat = [{
    $lhs `,` $rhs `,` $output
    (`transa` `=` $transa^)?
    (`transb` `=` $transb^)?
    (`alpha` `=` $alpha^)?
    (`beta` `=` $beta^)?
    attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)
  }];
}

#endif // LIB_DIALECT_BLIR_BLIROps_TD_
