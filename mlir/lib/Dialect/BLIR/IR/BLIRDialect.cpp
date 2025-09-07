#include "mlir/Dialect/BLIR/BLIRDialect.h"
#include "mlir/Dialect/BLIR/BLIROps.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/BLIR/BLIRDialect.cpp.inc"


using namespace mlir;
using namespace mlir::blir;

#define GET_OP_CLASSES
#include "mlir/Dialect/BLIR/BLIROps.cpp.inc"

void BLIRDialect::initialize() {
	addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/BLIR/BLIROps.cpp.inc"
	>();
}

