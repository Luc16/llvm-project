#ifndef LIB_DIALECT_BLIR_BLIRDialect_TD_
#define LIB_DIALECT_BLIR_BLIRDialect_TD_

include "mlir/IR/DialectBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

def BLIR_Dialect : Dialect {
  let name = "blir";
  let cppNamespace = "::mlir::blir";
  let summary = "BLIR dialect";
  let description = [{
	The BLIR dialect is designed to represent operations and constructs
	specific to the BLIR (Basic Linear Algebra Intermediate Representation)
	framework, facilitating optimizations and transformations for linear
	algebra computations.
  }]; 
}

#endif // LIB_DIALECT_BLIR_BLIRDialect_TD_
