// RUN: mlir-opt %s --blir-matmul-to-func | FileCheck %s

// CHECK-LABEL: func @test_matmul_int
// CHECK-NOT:   blir.matmul
// CHECK:         call @blir_matmul_3x3xi32_3x3xi32_3x3xi32
func.func @test_matmul_int(%arg0: memref<3x3xi32>, %arg1: memref<3x3xi32>, %arg2: memref<3x3xi32>) -> memref<3x3xi32> {
    blir.matmul %arg0, %arg1, %arg2 alpha=5.0 : memref<3x3xi32>, memref<3x3xi32> -> memref<3x3xi32>
    return %arg2 : memref<3x3xi32>
}


// CHECK-LABEL: func @test_matmul_int_2
// CHECK-NOT:   blir.matmul
// CHECK:         call @blir_matmul_2x2xi16_2x2xi16_2x2xi16
func.func @test_matmul_int_2(%arg0: memref<2x2xi16>, %arg1: memref<2x2xi16>, %arg2: memref<2x2xi16>) -> memref<2x2xi16> {
	blir.matmul %arg0, %arg1, %arg2 transb=true beta=0.5 : memref<2x2xi16>, memref<2x2xi16> -> memref<2x2xi16>
	return %arg2 : memref<2x2xi16>
}
