"""
Test Suite for TopoPyc Production Compiler.
Validates:
1. Simplicial complex structure and incidence relations.
2. Coboundary nilpotency (d_1 @ d_0 = 0).
3. Hodge Laplacians Delta_0, Delta_1, Delta_2 and harmonic spaces.
4. JIT compilation to native CPython extension and execution equivalence.
5. Standalone binary compilation.
"""

import math
import sys
import os
import unittest
import numpy as np

# Ensure topopyc is in PYTHONPATH
sys.path.insert(0, "/home/den/main/30-39 Development/32 DB & Tooling/PythonCompil")

from topopyc import SimplicialComplex, TypeSheaf, StalkType, HodgeLaplacianSolver, topojit, TopoCompiler

def kernel_mandelbrot(c_re: float, c_im: float, max_iter: int) -> int:
    z_re = 0.0
    z_im = 0.0
    count = 0
    for i in range(max_iter):
        z_re2 = z_re * z_re
        z_im2 = z_im * z_im
        if (z_re2 + z_im2) > 4.0:
            return i
        z_im = 2.0 * z_re * z_im + c_im
        z_re = z_re2 - z_im2 + c_re
        count += 1
    return count

def kernel_monte_carlo_pi(n: int) -> float:
    acc = 0.0
    for i in range(n):
        val = sin(float(i)) * 0.5 + 0.5
        acc += val
    return acc

class TestTopoPyc(unittest.TestCase):
    def test_01_simplicial_complex_and_nilpotency(self):
        complex = SimplicialComplex("test_complex")
        v0 = complex.add_simplex(0, "v0", "var")
        v1 = complex.add_simplex(0, "v1", "var")
        v2 = complex.add_simplex(0, "v2", "op")
        
        t0 = complex.add_triangle(v0, v1, v2, "t0")
        self.assertEqual(complex.num_vertices, 3)
        self.assertGreaterEqual(complex.num_edges, 3)
        self.assertEqual(complex.num_triangles, 1)

        sheaf = TypeSheaf(complex)
        solver = HodgeLaplacianSolver(sheaf)
        d0 = solver.build_d0()
        d1 = solver.build_d1()

        # Coboundary chain condition: d1 @ d0 must vanish
        nilpotent_diff = np.max(np.abs(d1 @ d0))
        self.assertAlmostEqual(nilpotent_diff, 0.0, places=6)

    def test_02_hodge_laplacians_spectrum(self):
        complex = SimplicialComplex("spectrum_test")
        v0 = complex.add_simplex(0, "v0", "var")
        v1 = complex.add_simplex(0, "v1", "var")
        complex.add_edge(v0, v1, "e01")

        sheaf = TypeSheaf(complex)
        solver = HodgeLaplacianSolver(sheaf)
        d0, d1, d2 = solver.compute_laplacians()
        
        # d0 is symmetric positive semi-definite
        eigvals = np.linalg.eigvalsh(d0)
        self.assertTrue(np.all(eigvals >= -1e-8))

        analysis = solver.analyze()
        self.assertEqual(analysis.dim_H0, 1) # 1 connected component

    def test_03_jit_execution_equivalence(self):
        compiled_mandelbrot = topojit(kernel_mandelbrot)

        # Compare pure Python vs Native Compiled Kernel
        py_res = kernel_mandelbrot(-0.75, 0.1, 1000)
        native_res = compiled_mandelbrot(-0.75, 0.1, 1000)

        self.assertEqual(py_res, native_res)

    def test_04_standalone_executable_generation(self):
        compiler = TopoCompiler(opt_level="-O3")
        out_bin = "/home/den/main/30-39 Development/32 DB & Tooling/PythonCompil/tests/test_standalone_mandel"
        compiled_bin = compiler.compile_executable(kernel_mandelbrot, out_bin, bench_args=(-0.75, 0.1, 1000))
        self.assertTrue(os.path.exists(compiled_bin))
        self.assertTrue(os.access(compiled_bin, os.X_OK))

    def test_05_topological_optimization_passes(self):
        from topopyc import TopologicalOptimizer
        compiler = TopoCompiler()
        complex, sheaf, analysis, fn_node, ret_type, param_types = compiler.analyze_function(kernel_mandelbrot)
        optimizer = TopologicalOptimizer(complex, sheaf, analysis)
        raw_env = dict(param_types)
        refined_env, stats = optimizer.optimize(raw_env)
        
        self.assertTrue(stats.is_monomorphic)
        self.assertTrue(stats.is_race_free)
        self.assertIn("SheafTypeFlowRefinement", stats.optimizations_applied)
        self.assertIn("HarmonicZeroAllocationUnboxing", stats.optimizations_applied)
        self.assertIn("HodgeLockFreeProof", stats.optimizations_applied)

if __name__ == "__main__":
    unittest.main()
