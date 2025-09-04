#include "mlir/Dialect/BLIR/BLIRDialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/BLIR/BLIRDialect.cpp.inc"

namespace mlir {
namespace blir {

// This includes the generated implementation for the dialect class.

void BLIRDialect::initialize() {
}

} // namespace blir
} // namespace mlir
