// ===================================================================
// TOPOLOGICALLY COMPILED MODULE: benchmark_example
// Synthesized via Higher Sheaf Hodge-Laplacian Invariant Analysis
// Vertices: 11 | Edges: 11 | dim H^0: 2 | dim H^1: 0
// ===================================================================
#include <stdint.h>
#include <stdio.h>
#include <math.h>

// Monomorphic Unboxed Native Kernel
double topopyc_benchmark_example(double x, int64_t n) {
    double acc = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        acc += sin(x * (double)i) * 0.5 + x * 2.0;
    }
    return acc;
}

int main() {
    printf("Running topopyc_benchmark_example...\n");
    double res = topopyc_benchmark_example(3.1415926535, 10000000);
    printf("Result: %f\n", res);
    return 0;
}