// RUN: mlir-opt %s --blir-matmul-to-func | FileCheck %s


func.func private @timestamp() -> i64
func.func private @print_gflops(f64, f64)

// CHECK-LABEL: func @main
// CHECK-NOT:   blir.matmul
// CHECK:       call @blir_matmul_2048x2048xf64_2048x2048xf64_2048x2048xf64
func.func @main() {
	%A = memref.alloc() : memref<2048x2048xf64>
	%B = memref.alloc() : memref<2048x2048xf64>
	%C = memref.alloc() : memref<2048x2048xf64>

	%cf1 = arith.constant 1.00000e+00 : f64
	linalg.fill ins(%cf1 : f64) outs(%A : memref<2048x2048xf64>)
    linalg.fill ins(%cf1 : f64) outs(%B : memref<2048x2048xf64>)
    linalg.fill ins(%cf1 : f64) outs(%C : memref<2048x2048xf64>)

	

    %start_time = func.call @timestamp() : () -> i64
	blir.matmul %A, %B, %C : memref<2048x2048xf64>, memref<2048x2048xf64> -> memref<2048x2048xf64>
    %end_time = func.call @timestamp() : () -> i64


	%size = arith.constant 2048.0 : f64
    %two = arith.constant 2.0 : f64
    %size_squared = arith.mulf %size, %size : f64         // N*N
    %size_cubed = arith.mulf %size_squared, %size : f64  // N*N*N
    %total_flops = arith.mulf %size_cubed, %two : f64     // (N^3) * 2

    %billion = arith.constant 1000000000.0 : f64

    // Calculate elapsed time in nanoseconds
    %elapsed_ns_i = arith.subi %end_time, %start_time : i64
    // Convert elapsed time to a double
    %elapsed_ns_f = arith.uitofp %elapsed_ns_i : i64 to f64

    // Calculate elapsed time in seconds
    %elapsed_s = arith.divf %elapsed_ns_f, %billion : f64
    
    // Calculate GFLOP/s
    %gflops = arith.divf %total_flops, %elapsed_ns_f : f64

    // 5. Print the final result
    call @print_gflops(%gflops, %elapsed_s) : (f64, f64) -> ()

	return
}
