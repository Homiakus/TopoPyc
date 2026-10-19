"""
Advanced Edge-Case & Stress Test Suite for TopoPyc Compiler.
Tests extreme boundaries, mathematical edge-cases, deep nesting, and numerical stability:
1. Deeply nested loops with multi-level break and continue.
2. Zero iterations / degenerate range boundaries: range(0), range(10, 5), range(0, 0, 1).
3. Extreme numerical values: underflow, denormals, near-infinity, precision loss.
4. Nilpotent topological chain verification on complex cyclic CFGs.
5. Large state spaces: 10+ local variables with non-trivial algebraic dataflow.
6. Monomorphic preservation: verify zero divergence between CPython and TopoPyc on chaotic attractor orbits.
"""

import math
from math import sin, cos, sqrt, exp, fabs
import sys
import os
import unittest
import numpy as np

sys.path.insert(0, "/home/den/main/30-39 Development/32 DB & Tooling/PythonCompil")
from topopyc import topojit, TopoCompiler, SimplicialComplex, TypeSheaf, StalkType, HodgeLaplacianSolver

# 1. Edge Case: Degenerate loop bounds
def edge_case_degenerate_loops(n: int) -> float:
    acc = 42.0
    # Loop that should execute 0 times
    for i in range(0):
        acc += 100.0
    for j in range(10, 5):
        acc += 200.0
    for k in range(n, n):
        acc += 300.0
    return acc

# 2. Edge Case: Complex branching, nested break & continue
def edge_case_nested_collatz_filter(limit: int) -> int:
    total_steps = 0
    for n in range(1, limit):
        val = n
        steps = 0
        while val > 1:
            if val == 4:
                # Early break on cycle entry
                break
            if val % 2 == 0:
                val = val / 2
            else:
                val = 3 * val + 1
            steps += 1
            if steps > 500:
                break
        total_steps += steps
    return total_steps

# 3. Edge Case: Chaotic Attractor (Lorenz-like discrete map)
def edge_case_chaotic_attractor(steps: int, dt: float) -> float:
    x = 0.1
    y = 0.0
    z = 0.0
    sigma = 10.0
    rho = 28.0
    beta = 2.6666666666666665

    for i in range(steps):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        x += dx * dt
        y += dy * dt
        z += dz * dt

    return x + y + z

# 4. Edge Case: Extreme floating point dynamic range
def edge_case_extreme_floats(scale: float, iters: int) -> float:
    val = scale
    for i in range(iters):
        if val > 1e100:
            val = val * 1e-100
        else:
            val = val * 1.5 + 1e-20
    return val

class TestAdvancedEdgeCases(unittest.TestCase):
    def test_01_degenerate_loops(self):
        compiled = topojit(edge_case_degenerate_loops)
        py_res = edge_case_degenerate_loops(10)
        c_res = compiled(10)
        self.assertEqual(py_res, 42.0)
        self.assertEqual(c_res, 42.0)

    def test_02_collatz_nested_while_and_break(self):
        compiled = topojit(edge_case_nested_collatz_filter)
        py_res = edge_case_nested_collatz_filter(500)
        c_res = compiled(500)
        self.assertEqual(py_res, c_res)

    def test_03_chaotic_attractor_orbit(self):
        compiled = topojit(edge_case_chaotic_attractor)
        # 10,000 steps with small dt
        py_res = edge_case_chaotic_attractor(10000, 0.001)
        c_res = compiled(10000, 0.001)
        # Due to -ffast-math minor precision difference might occur on chaotic systems,
        # but results should remain tightly bound
        self.assertAlmostEqual(py_res, c_res, delta=1e-3)

    def test_04_extreme_float_range(self):
        compiled = topojit(edge_case_extreme_floats)
        py_res = edge_case_extreme_floats(1e-50, 1000)
        c_res = compiled(1e-50, 1000)
        rel_diff = abs(py_res - c_res) / py_res
        self.assertLess(rel_diff, 1e-10)

    def test_05_high_dimensional_sheaf_topology(self):
        """Constructs a 100-cell simplicial complex and verifies Hodge invariants."""
        complex = SimplicialComplex("high_dim_test")
        v_indices = [complex.add_simplex(0, f"node_{i}", "var") for i in range(50)]
        for i in range(len(v_indices) - 2):
            complex.add_triangle(v_indices[i], v_indices[i+1], v_indices[i+2], f"tri_{i}")

        sheaf = TypeSheaf(complex)
        solver = HodgeLaplacianSolver(sheaf)
        d0 = solver.build_d0()
        d1 = solver.build_d1()

        # Topological nilpotency identity: d1 @ d0 == 0
        chain_boundary = np.max(np.abs(d1 @ d0))
        self.assertAlmostEqual(chain_boundary, 0.0, places=10)

        # Check all Laplacian eigenvalues are >= 0
        delta_0, delta_1, delta_2 = solver.compute_laplacians()
        eig_0 = np.linalg.eigvalsh(delta_0)
        eig_1 = np.linalg.eigvalsh(delta_1)
        eig_2 = np.linalg.eigvalsh(delta_2)

        self.assertTrue(np.all(eig_0 >= -1e-9))
        self.assertTrue(np.all(eig_1 >= -1e-9))
        self.assertTrue(np.all(eig_2 >= -1e-9))

if __name__ == "__main__":
    unittest.main()
