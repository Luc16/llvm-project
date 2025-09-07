#ifndef LIB_DIALECT_BLIR_TRANSFORM_PASSES_H_
#define LIB_DIALECT_BLIR_TRANSFORM_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
namespace blir {

#define GEN_PASS_DECL_BLIRMATMULTOLOOPSPASS
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/BLIR/Transforms/Passes.h.inc"

} // namespace blir
} // namespace mlir

#endif // LIB_DIALECT_BLIR_TRANSFORM_PASSES_H_
