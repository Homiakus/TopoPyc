"""
Comprehensive Edge-Case & Diverse Workload Benchmark: Pure CPython 3.13 vs TopoPyc.
Measures speedup, latency, and throughput across 5 distinct real-world computational domains:
1. Chaos / Dynamical Systems (Lorenz Strange Attractor - 15M steps)
2. Number Theory / Integer Search (Collatz Conjecture Max Trajectory - 2M numbers)
3. Cryptographic / Bitwise Permutation (Simulated Feistel Round Hashing - 25M rounds)
4. Numerical Integration / Monte-Carlo (Circle Quadrature - 30M samples)
5. Recursive / Nested Branching (Fractal Lyapunov Exponent Calculation - 10M points)
"""

import time
import math
from math import sin, cos, sqrt, fabs
import sys
import os

sys.path.insert(0, "/home/den/main/30-39 Development/32 DB & Tooling/PythonCompil")
from topopyc import topojit

# 1. Chaotic Dynamics
def bench_lorenz_attractor(steps: int, dt: float) -> float:
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

# 2. Number Theory
def bench_collatz_max_path(limit: int) -> int:
    max_steps = 0
    for n in range(1, limit):
        val = n
        steps = 0
        while val > 1:
            if val % 2 == 0:
                val = val / 2
            else:
                val = 3 * val + 1
            steps += 1
        if steps > max_steps:
            max_steps = steps
    return max_steps

# 3. Simulated Feistel Permutations
def bench_feistel_rounds(rounds: int, key: float) -> float:
    l = 12345.678
    r = 87654.321
    for i in range(rounds):
        # Round function
        f = sin(r * key + float(i)) * 1000.0
        new_r = l + f
        l = r
        r = new_r
    return l + r

# 4. Monte-Carlo Circle Quadrature
def bench_monte_carlo_pi(samples: int) -> float:
    inside = 0
    inv_max = 1.0 / float(samples)
    for i in range(samples):
        # Deterministic pseudorandom Weyl sequence
        x = (float(i) * 0.618033988749895)
        x = x - float(int(x))
        y = (float(i) * 0.414213562373095)
        y = y - float(int(y))
        if (x * x + y * y) <= 1.0:
            inside += 1
    return 4.0 * float(inside) * inv_max

# 5. Lyapunov Exponent
def bench_lyapunov_exponent(points: int, r: float) -> float:
    x = 0.5
    total_lyap = 0.0
    for i in range(points):
        x = r * x * (1.0 - x)
        deriv = fabs(r * (1.0 - 2.0 * x))
        if deriv > 0.000001:
            total_lyap += (deriv * 0.1) # proxy log approx
    return total_lyap

def run_stress_benchmarks():
    print("=" * 82)
    print("🚀 TOPOLOGICAL SHEAF COMPILER (TopoPyc) // FULL EDGE-CASE & STRESS BENCHMARK")
    print("=" * 82)

    results = []

    # Benchmark 1: Lorenz Attractor
    steps = 15_000_000
    dt = 0.001
    print(f"\n[1] Chaotic System (Lorenz Attractor - {steps:,} steps):")
    t0 = time.perf_counter()
    res_py = bench_lorenz_attractor(steps, dt)
    t_py = (time.perf_counter() - t0) * 1000.0
    print(f"    • Pure CPython 3.13:   {t_py:8.2f} ms")

    compiled_lorenz = topojit(bench_lorenz_attractor)
    compiled_lorenz(10, dt)
    t0 = time.perf_counter()
    res_c = compiled_lorenz(steps, dt)
    t_c = (time.perf_counter() - t0) * 1000.0
    speedup = t_py / max(t_c, 0.001)
    print(f"    • TopoPyc Native JIT:  {t_c:8.2f} ms | SPEEDUP: {speedup:5.1f}x")
    results.append(("Lorenz Attractor (15M)", t_py, t_c, speedup))

    # Benchmark 2: Collatz Number Theory
    collatz_limit = 500_000
    print(f"\n[2] Integer Search (Collatz Trajectories - {collatz_limit:,} numbers):")
    t0 = time.perf_counter()
    res_py = bench_collatz_max_path(collatz_limit)
    t_py = (time.perf_counter() - t0) * 1000.0
    print(f"    • Pure CPython 3.13:   {t_py:8.2f} ms | Max path: {res_py}")

    compiled_collatz = topojit(bench_collatz_max_path)
    compiled_collatz(10)
    t0 = time.perf_counter()
    res_c = compiled_collatz(collatz_limit)
    t_c = (time.perf_counter() - t0) * 1000.0
    speedup = t_py / max(t_c, 0.001)
    print(f"    • TopoPyc Native JIT:  {t_c:8.2f} ms | Max path: {res_c} | SPEEDUP: {speedup:5.1f}x")
    results.append(("Collatz Search (500K)", t_py, t_c, speedup))

    # Benchmark 3: Feistel Permutations
    rounds = 10_000_000
    key = 0.31415
    print(f"\n[3] Feistel Hashing & Permutations ({rounds:,} rounds):")
    t0 = time.perf_counter()
    res_py = bench_feistel_rounds(rounds, key)
    t_py = (time.perf_counter() - t0) * 1000.0
    print(f"    • Pure CPython 3.13:   {t_py:8.2f} ms")

    compiled_feistel = topojit(bench_feistel_rounds)
    compiled_feistel(10, key)
    t0 = time.perf_counter()
    res_c = compiled_feistel(rounds, key)
    t_c = (time.perf_counter() - t0) * 1000.0
    speedup = t_py / max(t_c, 0.001)
    print(f"    • TopoPyc Native JIT:  {t_c:8.2f} ms | SPEEDUP: {speedup:5.1f}x")
    results.append(("Feistel Permutation (10M)", t_py, t_c, speedup))

    # Benchmark 4: Monte-Carlo Pi
    samples = 25_000_000
    print(f"\n[4] Deterministic Monte-Carlo Quadrature ({samples:,} samples):")
    t0 = time.perf_counter()
    res_py = bench_monte_carlo_pi(samples)
    t_py = (time.perf_counter() - t0) * 1000.0
    print(f"    • Pure CPython 3.13:   {t_py:8.2f} ms | Pi est: {res_py:.6f}")

    compiled_mc = topojit(bench_monte_carlo_pi)
    compiled_mc(10)
    t0 = time.perf_counter()
    res_c = compiled_mc(samples)
    t_c = (time.perf_counter() - t0) * 1000.0
    speedup = t_py / max(t_c, 0.001)
    print(f"    • TopoPyc Native JIT:  {t_c:8.2f} ms | Pi est: {res_c:.6f} | SPEEDUP: {speedup:5.1f}x")
    results.append(("Monte-Carlo Pi (25M)", t_py, t_c, speedup))

    # Benchmark 5: Lyapunov Exponent
    points = 20_000_000
    r_param = 3.99
    print(f"\n[5] Nonlinear Lyapunov Orbit Mapping ({points:,} points):")
    t0 = time.perf_counter()
    res_py = bench_lyapunov_exponent(points, r_param)
    t_py = (time.perf_counter() - t0) * 1000.0
    print(f"    • Pure CPython 3.13:   {t_py:8.2f} ms")

    compiled_lyap = topojit(bench_lyapunov_exponent)
    compiled_lyap(10, r_param)
    t0 = time.perf_counter()
    res_c = compiled_lyap(points, r_param)
    t_c = (time.perf_counter() - t0) * 1000.0
    speedup = t_py / max(t_c, 0.001)
    print(f"    • TopoPyc Native JIT:  {t_c:8.2f} ms | SPEEDUP: {speedup:5.1f}x")
    results.append(("Lyapunov Exponent (20M)", t_py, t_c, speedup))

    print("\n" + "=" * 82)
    print(f"{'Workload Name':<30} | {'CPython 3.13':<12} | {'TopoPyc JIT':<12} | {'Speedup':<10}")
    print("-" * 82)
    for name, py_t, c_t, sp in results:
        print(f"{name:<30} | {py_t:9.2f} ms | {c_t:9.2f} ms | {sp:7.1f}x")
    print("=" * 82)

if __name__ == "__main__":
    run_stress_benchmarks()
