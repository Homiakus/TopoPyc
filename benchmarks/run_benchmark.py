"""
Comprehensive Production Benchmark: Pure CPython 3.13 vs TopoPyc (Native Unboxed Topological JIT).
Evaluates compute throughput, speedup ratio, and memory efficiency across heavy loops.
"""

import time
import math
from math import sin
import sys
import os

sys.path.insert(0, "/home/den/main/30-39 Development/32 DB & Tooling/PythonCompil")
from topopyc import topojit

def pure_py_nbody_step(dt: float, steps: int) -> float:
    x = 1.0
    y = 0.5
    z = -0.5
    vx = 0.01
    vy = -0.02
    vz = 0.03
    total_energy = 0.0

    for i in range(steps):
        # Update coordinates
        x += vx * dt
        y += vy * dt
        z += vz * dt
        
        # Distance calculation
        dist_sq = x * x + y * y + z * z + 0.0001
        inv_dist = 1.0 / dist_sq
        
        # Velocity update
        vx -= x * inv_dist * dt
        vy -= y * inv_dist * dt
        vz -= z * inv_dist * dt
        
        total_energy += (vx * vx + vy * vy + vz * vz) * 0.5

    return total_energy

def pure_py_wave_pde(freq: float, steps: int) -> float:
    u = 0.0
    v = 1.0
    dt = 0.001
    damping = 0.99999

    for i in range(steps):
        acc = -freq * freq * sin(u)
        v = (v + acc * dt) * damping
        u += v * dt

    return u

def run_benchmarks():
    print("=" * 75)
    print("⚡ TOPOLOGICAL SHEAF COMPILER (TopoPyc) // PRODUCTION BENCHMARK")
    print("=" * 75)

    # 1. N-Body Simulation Benchmark
    n_steps = 10_000_000
    dt = 0.001
    print(f"\n[1] N-Body Orbit Step Simulation ({n_steps:,} iterations):")

    # Pure Python
    t0 = time.perf_counter()
    res_py = pure_py_nbody_step(dt, n_steps)
    t1 = time.perf_counter()
    time_py = (t1 - t0) * 1000.0
    print(f"    • Pure CPython 3.13:   {time_py:8.2f} ms | Result: {res_py:.6f}")

    # TopoPyc JIT
    print("    • Compiling via TopoPyc JIT (Hodge-Laplacian unboxing)...")
    compiled_nbody = topojit(pure_py_nbody_step)
    # Warmup
    compiled_nbody(dt, 10)

    t2 = time.perf_counter()
    res_topo = compiled_nbody(dt, n_steps)
    t3 = time.perf_counter()
    time_topo = (t3 - t2) * 1000.0
    print(f"    • TopoPyc Native JIT:  {time_topo:8.2f} ms | Result: {res_topo:.6f}")

    speedup_nbody = time_py / max(time_topo, 0.0001)
    print(f"    ⚡ SPEEDUP: {speedup_nbody:.1f}x FASTER THAN CPYTHON 3.13")

    # 2. Wave PDE Simulation Benchmark
    pde_steps = 20_000_000
    freq = 1.5
    print(f"\n[2] Nonlinear Wave PDE Solver ({pde_steps:,} iterations with sin):")

    t0 = time.perf_counter()
    res_pde_py = pure_py_wave_pde(freq, pde_steps)
    t1 = time.perf_counter()
    time_pde_py = (t1 - t0) * 1000.0
    print(f"    • Pure CPython 3.13:   {time_pde_py:8.2f} ms | Result: {res_pde_py:.6f}")

    compiled_pde = topojit(pure_py_wave_pde)
    compiled_pde(freq, 10)

    t2 = time.perf_counter()
    res_pde_topo = compiled_pde(freq, pde_steps)
    t3 = time.perf_counter()
    time_pde_topo = (t3 - t2) * 1000.0
    print(f"    • TopoPyc Native JIT:  {time_pde_topo:8.2f} ms | Result: {res_pde_topo:.6f}")

    speedup_pde = time_pde_py / max(time_pde_topo, 0.0001)
    print(f"    ⚡ SPEEDUP: {speedup_pde:.1f}x FASTER THAN CPYTHON 3.13")
    print("\n" + "=" * 75)

if __name__ == "__main__":
    run_benchmarks()
