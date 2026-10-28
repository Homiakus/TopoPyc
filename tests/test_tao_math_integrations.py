import pytest
import numpy as np
from topopyc import (
    SimplicialComplex,
    TypeSheaf,
    StalkType,
    HodgeLaplacianSolver,
    SimplicialRicciOptimizer,
    SimplecticMemoryPass,
    DSpaceConcurrencyPass,
    TopoLens,
    record_lens,
    array_lens,
    topojit
)

def test_sparse_hodge_laplacians():
    sc = SimplicialComplex("sparse_test")
    v0 = sc.add_simplex(0, "v0", "var")
    v1 = sc.add_simplex(0, "v1", "var")
    v2 = sc.add_simplex(0, "v2", "var")
    e0 = sc.add_edge(v0, v1, "e01")
    e1 = sc.add_edge(v1, v2, "e12")
    
    sheaf = TypeSheaf(sc)
    solver = HodgeLaplacianSolver(sheaf)
    
    d0_sparse = solver.build_d0_sparse()
    d0_dense = solver.build_d0()
    assert d0_sparse.shape == d0_dense.shape
    assert np.allclose(d0_sparse.toarray(), d0_dense)

def test_ricci_curvature_optimization():
    sc = SimplicialComplex("ricci_test")
    v0 = sc.add_simplex(0, "x", "var")
    v1 = sc.add_simplex(0, "y", "var")
    v2 = sc.add_simplex(0, "z", "var")
    sc.add_edge(v0, v1, "e01")
    sc.add_edge(v1, v2, "e12")
    sc.add_edge(v0, v2, "e02") # triangle cluster

    optimizer = SimplicialRicciOptimizer(sc)
    diag = optimizer.analyze_curvatures()
    assert diag.num_nodes == 3
    assert diag.num_edges == 3
    assert diag.mean_curvature >= 0.0

def test_symplectic_memory_tracking():
    pass_mem = SimplecticMemoryPass()
    diag = pass_mem.analyze_variables({"acc": 100, "temp": 5, "unused": 0})
    assert diag.total_buffers_tracked == 3
    assert diag.zero_leak_verified is True
    assert diag.recycled_buffer_count >= 2

def test_dspace_concurrency():
    dspace = DSpaceConcurrencyPass([1.0, 1.0])
    # Mutex on [0.2, 0.6] for both processes
    dspace.add_mutex_exclusion({0: (0.2, 0.6), 1: (0.2, 0.6)})
    report = dspace.verify_deadlock_free()
    assert report.is_race_free is True
    assert report.dipath_to_exit_exists is True

def test_profunctor_optics():
    lens_a = record_lens("config")
    lens_b = record_lens("threads")
    composed = lens_a @ lens_b

    state = {"config": {"threads": 4, "mode": "native"}}
    assert composed.get(state) == 4

    new_state = composed.put(state, 8)
    assert composed.get(new_state) == 8
    assert new_state["config"]["mode"] == "native"

    arr_lens = array_lens(1)
    lst = [10, 20, 30]
    assert arr_lens.get(lst) == 20
    assert arr_lens.put(lst, 99) == [10, 99, 30]

def test_topojit_with_extended_optimizations():
    @topojit
    def compute_kernel(n: int) -> float:
        s = 0.0
        for i in range(n):
            s = s + (i * 0.5)
        return s

    res = compute_kernel(100)
    expected = sum(i * 0.5 for i in range(100))
    assert abs(res - expected) < 1e-6
