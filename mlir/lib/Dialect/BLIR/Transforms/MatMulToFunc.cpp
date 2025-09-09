#include "mlir/Dialect/BLIR/IR/BLIROps.h"
#include "mlir/Dialect/BLIR/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h" 

namespace mlir::blir {
#define GEN_PASS_DEF_BLIRMATMULTOFUNCPASS
#include "mlir/Dialect/BLIR/Transforms/Passes.h.inc"
} // namespace mlir::blir

using namespace mlir;
using namespace mlir::blir;

static std::string createMatMulMangledFunctionName(const TypeRange &operands,
											const TypeRange &results) {
	std::string mangledFuncName;
    llvm::raw_string_ostream os(mangledFuncName);
    os << "blir_matmul";
	for (Type type : operands) {
      os << "_";
      // Check if the type is a MemRefType
      if (auto memref = dyn_cast<MemRefType>(type)) {
        // It is a memref, so manually construct the name from its parts.
        auto shape = memref.getShape();
        for (size_t i = 0; i < shape.size(); ++i) {
          if (ShapedType::isDynamic(shape[i])) {
            os << "?"; // Use '?' for dynamic dimensions
          } else {
            os << shape[i];
          }
          if (i < shape.size() - 1) {
            os << "x";
          }
        }
        os << "x" << memref.getElementType();
      } else {
        // It's not a memref (e.g., i1, f32), so print it normally.
        os << type;
      }
    }
    // (Optional) Include result types if they can vary.
    for (Type type : results) {
      os << "_" << type;
    }

    // 3. Sanitize the name to make it a valid symbol.
    // MLIR symbol names can't contain characters like '<', '>', ',', or ' '.
    std::replace(mangledFuncName.begin(), mangledFuncName.end(), '<', '_');
    std::replace(mangledFuncName.begin(), mangledFuncName.end(), '>', '_');
    std::replace(mangledFuncName.begin(), mangledFuncName.end(), ',', '_');
    std::replace(mangledFuncName.begin(), mangledFuncName.end(), ' ', '_');

    // std::replace(mangledFuncName.begin(), mangledFuncName.end(), 'x', '_'); // For shapes like 4x4
	
	return mangledFuncName;
}


struct MatMulToFuncPattern : OpRewritePattern<blir::MatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(blir::MatMulOp matMulOp,
								PatternRewriter &rewriter) const override {

	auto module = matMulOp->getParentOfType<ModuleOp>();
    if (!module) {
      return failure(); // Should not happen in well-formed IR
    }
	Location loc = matMulOp.getLoc();

	auto i1Type = rewriter.getI1Type();
    auto f32Type = rewriter.getF32Type();


	auto operandTypes = matMulOp.getOperandTypes();
	auto resultTypes = matMulOp->getResultTypes();
	SmallVector<Type, 7> operandTypesVec(operandTypes);
	operandTypesVec.push_back(i1Type);    
    operandTypesVec.push_back(i1Type);    
    operandTypesVec.push_back(f32Type);   
    operandTypesVec.push_back(f32Type);

	std::string funcName = createMatMulMangledFunctionName(operandTypes, resultTypes);
    auto funcOp = module.lookupSymbol<func::FuncOp>(funcName);


	// TODO: Make this thread-safe
	if (!funcOp) {
		auto funcType = rewriter.getFunctionType(operandTypesVec, resultTypes);

		OpBuilder::InsertionGuard guard(rewriter);
		rewriter.setInsertionPointToStart(module.getBody());

		funcOp = rewriter.create<func::FuncOp>(loc, funcName, funcType);
		funcOp.setPrivate();

		Block *entryBlock = rewriter.createBlock(&funcOp.getBody(), {}, operandTypesVec,
				SmallVector<Location>(operandTypesVec.size(), loc));

		// Set insertion point inside the new function's block to add the return.
		rewriter.setInsertionPointToStart(entryBlock);

		Value lhsMemRef = entryBlock->getArgument(0);
		Value rhsMemRef = entryBlock->getArgument(1);
		Value outputMemRef = entryBlock->getArgument(2);

		auto lhsType = dyn_cast<MemRefType>(lhsMemRef.getType());
		auto rhsType = dyn_cast<MemRefType>(rhsMemRef.getType());
		// auto outputType = dyn_cast<MemRefType>(entryBlock->getArgument(2).getType());

		// Getting dimensions (assuming 2D matrices)
		int64_t M = lhsType.getShape()[0];
		int64_t K = lhsType.getShape()[1];
		int64_t N = rhsType.getShape()[1];


		// Creating constants for loop bounds
		// Value cM = rewriter.create<arith::ConstantIndexOp>(loc, M);
		// Value cK = rewriter.create<arith::ConstantIndexOp>(loc, K);
		// Value cN = rewriter.create<arith::ConstantIndexOp>(loc, N);
		// Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
		// Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);


		// for i = 0 to M
		auto outerLoop = rewriter.create<affine::AffineForOp>(loc, 0, M, 1);
		rewriter.setInsertionPointToStart(outerLoop.getBody());

		// for k = 0 to K
		auto innerLoop = rewriter.create<affine::AffineForOp>(loc, 0, K, 1);
		rewriter.setInsertionPointToStart(innerLoop.getBody());

		// for j = 0 to N
		auto middleLoop = rewriter.create<affine::AffineForOp>(loc, 0, N, 1);
		rewriter.setInsertionPointToStart(middleLoop.getBody());

		Value i = outerLoop.getInductionVar();
		Value j = middleLoop.getInductionVar();
		Value k = innerLoop.getInductionVar();

		Value lhsVal = rewriter.create<affine::AffineLoadOp>(loc, lhsMemRef, ValueRange{i, k});
		Value rhsVal = rewriter.create<affine::AffineLoadOp>(loc, rhsMemRef, ValueRange{k, j});
		Value outputVal = rewriter.create<affine::AffineLoadOp>(loc, outputMemRef, ValueRange{i, j});

		Type elementType = lhsVal.getType();
		Value mul, add;

		if (elementType.isFloat()) {
			mul = rewriter.create<arith::MulFOp>(loc, lhsVal, rhsVal);
			add = rewriter.create<arith::AddFOp>(loc, outputVal, mul);
		} 
		else if (elementType.isInteger()) {
			mul = rewriter.create<arith::MulIOp>(loc, lhsVal, rhsVal);
			add = rewriter.create<arith::AddIOp>(loc, outputVal, mul);
		} 
		else {
			// Handle error, e.g., by returning a failure or asserting.
			assert(false && "Unsupported type for multiplication and addition");
		}

		rewriter.create<affine::AffineStoreOp>(loc, add, outputMemRef, ValueRange{i, j});

		rewriter.setInsertionPointToEnd(entryBlock);

		rewriter.create<func::ReturnOp>(loc, ValueRange{});
	}


	rewriter.setInsertionPoint(matMulOp);

	SmallVector<Value, 7> callOperands(matMulOp.getOperands());
	callOperands.push_back(rewriter.create<arith::ConstantOp>(loc, i1Type, matMulOp.getTransaAttr()));
	callOperands.push_back(rewriter.create<arith::ConstantOp>(loc, i1Type, matMulOp.getTransbAttr()));
	callOperands.push_back(rewriter.create<arith::ConstantOp>(loc, f32Type, matMulOp.getAlphaAttr()));
	callOperands.push_back(rewriter.create<arith::ConstantOp>(loc, f32Type, matMulOp.getBetaAttr()));

	auto funcCall = rewriter.create<func::CallOp>(loc, funcOp, callOperands);
	rewriter.replaceOp(matMulOp, funcCall);

	return success();
  }

};


struct BLIRMatMulToFuncPass final
    : blir::impl::BLIRMatMulToFuncPassBase<BLIRMatMulToFuncPass> {
  void runOnOperation() override {
	RewritePatternSet patterns(&getContext());
	patterns.add<MatMulToFuncPattern>(patterns.getContext());
	if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
	  signalPassFailure();
	}

  }
};


//
//
// struct MatMulToFuncPattern : public OpRewritePattern<MatMulOp> {
//   using OpRewritePattern<MatMulOp>::OpRewritePattern;
//
//   LogicalResult matchAndRewrite(MatMulOp op,
//                                 PatternRewriter &rewriter) const override {
//     Location loc = op.getLoc();
//
//     // Obter os operandos
//     Value lhs = op.getLhs();
//     Value rhs = op.getRhs();
//     Value output = op.getOutput();
//
//     // Obter tipos dos tensors
//     auto lhsType = lhs.getType().dyn_cast<RankedTensorType>();
//     auto rhsType = rhs.getType().dyn_cast<RankedTensorType>();
//     auto outputType = output.getType().dyn_cast<RankedTensorType>();
//
//     if (!lhsType || !rhsType || !outputType) {
//       return rewriter.notifyMatchFailure(op, "non-ranked tensor types not supported");
//     }
//
//     // Obter dimensões (assumindo matrizes 2D)
//     int64_t M = lhsType.getShape()[0];
//     int64_t K = lhsType.getShape()[1];
//     int64_t N = rhsType.getShape()[1];
//
//     // Converter tensors para memrefs (para usar affine loops)
//     auto lhsMemRefType = MemRefType::get(lhsType.getShape(), lhsType.getElementType());
//     auto rhsMemRefType = MemRefType::get(rhsType.getShape(), rhsType.getElementType());
//     auto outputMemRefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
//
//     Value lhsMemRef = rewriter.create<memref::CastOp>(loc, lhsMemRefType, lhs);
//     Value rhsMemRef = rewriter.create<memref::CastOp>(loc, rhsMemRefType, rhs);
//     Value outputMemRef = rewriter.create<memref::CastOp>(loc, outputMemRefType, output);
//
//     // Criar constantes para os bounds dos loops
//     Value cM = rewriter.create<arith::ConstantIndexOp>(loc, M);
//     Value cK = rewriter.create<arith::ConstantIndexOp>(loc, K);
//     Value cN = rewriter.create<arith::ConstantIndexOp>(loc, N);
//     Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
//     Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
//
//     // Criar os loops aninhados para multiplicação de matrizes
//     // for i = 0 to M
//     auto outerLoop = rewriter.create<affine::AffineForOp>(loc, c0, cM, c1);
//     rewriter.setInsertionPointToStart(outerLoop.getBody());
//
//     // for j = 0 to N
//     auto middleLoop = rewriter.create<affine::AffineForOp>(loc, c0, cN, c1);
//     rewriter.setInsertionPointToStart(middleLoop.getBody());
//
//     // for k = 0 to K
//     auto innerLoop = rewriter.create<affine::AffineForOp>(loc, c0, cK, c1);
//     rewriter.setInsertionPointToStart(innerLoop.getBody());
//
//     // Corpo do loop: output[i][j] += lhs[i][k] * rhs[k][j]
//     Value i = outerLoop.getInductionVar();
//     Value j = middleLoop.getInductionVar();
//     Value k = innerLoop.getInductionVar();
//
//     // Carregar valores
//     Value lhsVal = rewriter.create<affine::AffineLoadOp>(loc, lhsMemRef, ValueRange{i, k});
//     Value rhsVal = rewriter.create<affine::AffineLoadOp>(loc, rhsMemRef, ValueRange{k, j});
//     Value outputVal = rewriter.create<affine::AffineLoadOp>(loc, outputMemRef, ValueRange{i, j});
//
//     // Multiplicar
//     Value mul = rewriter.create<arith::MulFOp>(loc, lhsVal, rhsVal);
//
//     // Somar com o valor atual
//     Value add = rewriter.create<arith::AddFOp>(loc, outputVal, mul);
//
//     // Armazenar o resultado
//     rewriter.create<affine::AffineStoreOp>(loc, add, outputMemRef, ValueRange{i, j});
//
//     // Remover a operação original
//     rewriter.eraseOp(op);
//
//     return success();
//   }
// };
