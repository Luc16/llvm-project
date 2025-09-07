// RUN: mlir-opt %s

// CHECK-LABEL: @test_syntax
func.func @test_matmul_basic(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>, %arg2: memref<4x4xf32>) -> memref<4x4xf32> {
	blir.matmul %arg0, %arg1, %arg2 : memref<4x4xf32>, memref<4x4xf32> -> memref<4x4xf32>
		return %arg2 : memref<4x4xf32>
}

func.func @test_matmul_transa(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>, %arg2: memref<4x4xf32>) -> memref<4x4xf32> {
	blir.matmul %arg0, %arg1, %arg2 transa=true : memref<4x4xf32>, memref<4x4xf32> -> memref<4x4xf32>
		return %arg2 : memref<4x4xf32>
}

func.func @test_matmul_full(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>, %arg2: memref<4x4xf32>) -> memref<4x4xf32> {
	blir.matmul %arg0, %arg1, %arg2 transa=true transb=false alpha=2.0 beta=1.0 : memref<4x4xf32>, memref<4x4xf32> -> memref<4x4xf32>
		return %arg2 : memref<4x4xf32>
}

func.func @test_matmul_int(%arg0: memref<3x3xi32>, %arg1: memref<3x3xi32>, %arg2: memref<3x3xi32>) -> memref<3x3xi32> {
	blir.matmul %arg0, %arg1, %arg2 alpha=1.0 : memref<3x3xi32>, memref<3x3xi32> -> memref<3x3xi32>
		return %arg2 : memref<3x3xi32>
}

func.func @test_matmul_rect(%arg0: memref<2x3xf32>, %arg1: memref<3x4xf32>, %arg2: memref<2x4xf32>) -> memref<2x4xf32> {
	blir.matmul %arg0, %arg1, %arg2 : memref<2x3xf32>, memref<3x4xf32> -> memref<2x4xf32>
		return %arg2 : memref<2x4xf32>
}
