// RUN: mlir-opt %s --blir-matmul-to-func | FileCheck %s

// CHECK-LABEL: func @test_matmul_transa
// CHECK-NOT:   blir.matmul
// CHECK:         call @blir_matmul_4x4xf64_4x4xf64_4x4xf64
func.func @test_matmul_transa(%arg0: memref<4x4xf64>, %arg1: memref<4x4xf64>, %arg2: memref<4x4xf64>) -> memref<4x4xf64> {
    blir.matmul %arg0, %arg1, %arg2 transa=true : memref<4x4xf64>, memref<4x4xf64> -> memref<4x4xf64>
    return %arg2 : memref<4x4xf64>
}

// CHECK-LABEL: func @test_matmul_full
// CHECK-NOT:   blir.matmul
// CHECK:         call @blir_matmul_4x3xf32_3x4xf32_4x4xf32
func.func @test_matmul_full(%arg0: memref<4x3xf32>, %arg1: memref<3x4xf32>, %arg2: memref<4x4xf32>) -> memref<4x4xf32> {
    blir.matmul %arg0, %arg1, %arg2 transa=true transb=false alpha=2.0 beta=1.0 : memref<4x3xf32>, memref<3x4xf32> -> memref<4x4xf32>
    return %arg2 : memref<4x4xf32>
}


// CHECK-LABEL: func @test_matmul_rect
// CHECK-NOT:   blir.matmul
// CHECK:         call @blir_matmul_2x3xf32_3x4xf32_2x4xf32
func.func @test_matmul_rect(%arg0: memref<2x3xf32>, %arg1: memref<3x4xf32>, %arg2: memref<2x4xf32>) -> memref<2x4xf32> {
    blir.matmul %arg0, %arg1, %arg2 : memref<2x3xf32>, memref<3x4xf32> -> memref<2x4xf32>
    return %arg2 : memref<2x4xf32>
}
