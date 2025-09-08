// RUN: mlir-opt %s --blir-matmul-to-loops | FileCheck %s

// CHECK-LABEL: func @test_matmul_basic
// CHECK-NOT:   blir.matmul
// CHECK:       call @blir_matmul_4x4xf32_4x4xf32_4x4xf32
func.func @test_matmul_basic(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>, %arg2: memref<4x4xf32>) -> memref<4x4xf32> {
	blir.matmul %arg0, %arg1, %arg2 transa=true transb=false alpha=2.0 beta=1.0 : memref<4x4xf32>, memref<4x4xf32> -> memref<4x4xf32>
	return %arg2 : memref<4x4xf32>
}

