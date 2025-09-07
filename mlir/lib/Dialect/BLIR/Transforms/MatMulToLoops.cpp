#include "mlir/Dialect/BLIR/IR/BLIROps.h"
#include "mlir/Dialect/BLIR/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::blir {
#define GEN_PASS_DEF_BLIRMATMULTOLOOPSPASS
#include "mlir/Dialect/BLIR/Transforms/Passes.h.inc"
} // namespace mlir::blir

using namespace mlir;
using namespace mlir::blir;


struct BLIRMatMulToLoopsPass final
    : blir::impl::BLIRMatMulToLoopsPassBase<BLIRMatMulToLoopsPass> {
  void runOnOperation() override {
    func::FuncOp f = getOperation();
	f.walk([&](blir::MatMulOp op) {
		// Get the operands of your matmul op
		mlir::Value lhs = op.getLhs();
		mlir::Value rhs = op.getRhs();
		mlir::Value output = op.getOutput();
		bool isTransposedA = op.getTransa();
		bool isTransposedB = op.getTransb();
		// Get the value as an APFloat
		llvm::APFloat alphaAP = op.getAlpha();
		llvm::APFloat betaAP = op.getBeta();

		// Convert to a standard C++ double
		double alpha = alphaAP.convertToDouble();
		double beta = betaAP.convertToDouble();
		

		op.emitRemark() << "\n\nFound a MatMulOp:\n\tOperand A has type: " << lhs.getType() << "\n\tOperand B has type: " << rhs.getType() << "\n\tOperand C has type: " << output.getType() << "\n\tTransA: " << (isTransposedA ? "true" : "false") << "\n\tTransB: " << (isTransposedB ? "true" : "false") << "\n\tAlpha: " << alpha << "\n\tBeta: " << beta << "\n";

	});

  }
};


//
//
// struct MatMulToLoopsPattern : public OpRewritePattern<MatMulOp> {
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
//
// struct BLIRMatMulToLoopsPass
//     : public impl::BLIRMatMulToLoopsPassBase<BLIRMatMulToLoopsPass> {
//   void runOnOperation() override {
//     auto func = getOperation();
//     MLIRContext *context = &getContext();
//
//     RewritePatternSet patterns(context);
//     patterns.add<MatMulToLoopsPattern>(context);
//
//     if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
//       signalPassFailure();
//   }
// };
//
//
// std::unique_ptr<Pass> mlir::blir::createMatMulToLoopsPass() {
//   return std::make_unique<BLIRMatMulToLoopsPass>();
// }
