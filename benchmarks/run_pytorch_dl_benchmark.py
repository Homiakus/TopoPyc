"""
PyTorch Real-World Deep Learning Kernels Comparison:
CPython vs PyTorch Eager vs TopoPyc (Native Unboxed Topological JIT)
"""

import time
import math
import sys
import torch

sys.path.insert(0, "/home/den/main/30-39 Development/32 DB & Tooling/PythonCompil")
from topopyc import topojit

# 1. Fused GELU Transformation Loop
def gelu_poly(x_val: float, steps: int) -> float:
    acc = x_val
    for i in range(steps):
        x3 = acc * acc * acc
        inner = 0.79788456 * (acc + 0.044715 * x3)
        e2 = math.exp(inner if inner < 15.0 else 15.0)
        tanh_val = (e2 - 1.0) / (e2 + 1.0)
        acc = 0.5 * acc * (1.0 + tanh_val)
        if acc < 0.0001:
            acc = acc + 0.1
        elif acc > 10.0:
            acc = acc * 0.5
    return acc

def gelu_poly_torch(tx: torch.Tensor, steps: int) -> torch.Tensor:
    acc = tx
    for i in range(steps):
        x3 = acc * acc * acc
        inner = 0.79788456 * (acc + 0.044715 * x3)
        tanh_val = torch.tanh(inner)
        acc = 0.5 * acc * (1.0 + tanh_val)
        if acc < 0.0001:
            acc = acc + 0.1
        elif acc > 10.0:
            acc = acc * 0.5
    return acc

# 2. Attention Scaled Dot-Product Token Reduction
def attention_reduction(dim: int, query_val: float) -> float:
    acc_score = 0.0
    inv_sqrt_d = 1.0 / math.sqrt(float(dim))
    for i in range(dim):
        key_val = math.sin(float(i) * 0.1) * 0.5 + 0.5
        val_val = math.cos(float(i) * 0.05)
        score = (query_val * key_val) * inv_sqrt_d
        weight = math.exp(score * 0.1)
        acc_score += weight * val_val
    return acc_score

# 3. Deep Recurrent Transition Sequence (RNN / Time Series)
def rnn_sequence(seq_len: int, h_init: float, x_init: float) -> float:
    h = h_init
    x = x_init
    whh = 0.85
    wxh = 0.45
    for i in range(seq_len):
        raw = whh * h + wxh * x + math.sin(float(i)) * 0.1
        e = math.exp(raw if raw < 10.0 else 10.0)
        h = (e - 1.0) / (e + 1.0)
        x = h * 0.5
    return h

def main():
    print("=" * 80)
    print("⚡ PYTORCH REAL-WORLD DEEP LEARNING BENCHMARK")
    print("=" * 80)

    # Kernel 1
    steps = 5_000_000
    x_init = 1.25
    print(f"\n[1] Fused GELU Transformer Sublayer ({steps:,} iterations):")
    
    t0 = time.perf_counter()
    res_py = gelu_poly(x_init, steps)
    t_py = (time.perf_counter() - t0) * 1000.0
    print(f"    • Pure CPython 3.13:      {t_py:8.2f} ms | Result: {res_py:.6f}")

    sub = 50_000
    tx = torch.tensor(x_init, dtype=torch.float32)
    t0 = time.perf_counter()
    _ = gelu_poly_torch(tx, sub)
    t_torch = (time.perf_counter() - t0) * 1000.0 * (steps / sub)
    print(f"    • PyTorch Tensor Eager:   {t_torch:8.2f} ms (projected {steps:,})")

    compiled_gelu = topojit(gelu_poly)
    compiled_gelu(x_init, 10) # warmup
    t0 = time.perf_counter()
    res_topo = compiled_gelu(x_init, steps)
    t_topo = (time.perf_counter() - t0) * 1000.0
    print(f"    • TopoPyc Native JIT:     {t_topo:8.2f} ms | Result: {res_topo:.6f}")
    print(f"    ⚡ SPEEDUP: {t_py / max(t_topo, 0.001):.1f}x vs CPython | {t_torch / max(t_topo, 0.001):.1f}x vs PyTorch")

    # Kernel 2
    dim = 15_000_000
    q_val = 0.75
    print(f"\n[2] Self-Attention Token Score Reduction ({dim:,} tokens):")
    
    t0 = time.perf_counter()
    res_py = attention_reduction(dim, q_val)
    t_py = (time.perf_counter() - t0) * 1000.0
    print(f"    • Pure CPython 3.13:      {t_py:8.2f} ms | Result: {res_py:.6f}")

    compiled_attn = topojit(attention_reduction)
    compiled_attn(10, q_val)
    t0 = time.perf_counter()
    res_topo = compiled_attn(dim, q_val)
    t_topo = (time.perf_counter() - t0) * 1000.0
    print(f"    • TopoPyc Native JIT:     {t_topo:8.2f} ms | Result: {res_topo:.6f}")
    print(f"    ⚡ SPEEDUP: {t_py / max(t_topo, 0.001):.1f}x vs CPython")

    # Kernel 3
    seq_len = 10_000_000
    print(f"\n[3] Recurrent State Transition Sequence ({seq_len:,} steps):")
    
    t0 = time.perf_counter()
    res_py = rnn_sequence(seq_len, 0.1, 0.5)
    t_py = (time.perf_counter() - t0) * 1000.0
    print(f"    • Pure CPython 3.13:      {t_py:8.2f} ms | Result: {res_py:.6f}")

    compiled_rnn = topojit(rnn_sequence)
    compiled_rnn(10, 0.1, 0.5)
    t0 = time.perf_counter()
    res_topo = compiled_rnn(seq_len, 0.1, 0.5)
    t_topo = (time.perf_counter() - t0) * 1000.0
    print(f"    • TopoPyc Native JIT:     {t_topo:8.2f} ms | Result: {res_topo:.6f}")
    print(f"    ⚡ SPEEDUP: {t_py / max(t_topo, 0.001):.1f}x vs CPython")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
