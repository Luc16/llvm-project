#ifndef LIB_DIALECT_BLIR_BLIROPS_H_
#define LIB_DIALECT_BLIR_BLIROPS_H_

#include "mlir/Dialect/BLIR/BLIRDialect.h"
#include "mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"
#include "mlir/IR/Dialect.h"       // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/BLIR/BLIROps.h.inc"

#endif  // LIB_DIALECT_BLIR_BLIROPS_H_
