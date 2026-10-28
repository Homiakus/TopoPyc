"""
Ricci Curvature Optimization Pass for AST Simplicial Complexes.
Ported and adapted from Tao Science: all-math/ricci-curvature-refactoring.

Calculates Ollivier-Ricci curvature on the 1-skeleton of the program simplicial complex.
- Positive curvature identifies dense dataflow clusters (Kernel Fusion candidates).
- Negative curvature detects architectural bottlenecks and synchronization points.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
import scipy.optimize as opt
from .simplicial import SimplicialComplex

@dataclass
class RicciPassDiagnostic:
    num_nodes: int
    num_edges: int
    mean_curvature: float
    bottlenecks_detected: int
    fusion_clusters_detected: int
    fusion_candidates: List[Tuple[str, str, float]]

class SimplicialRicciOptimizer:
    """
    Computes discrete Ollivier-Ricci curvature on SimplicialComplex.
    kappa(u, v) = 1 - W_1(m_u, m_v) / d(u, v)
    """
    def __init__(self, complex: SimplicialComplex, alpha: float = 0.5):
        self.complex = complex
        self.alpha = alpha

    def compute_shortest_paths(self) -> np.ndarray:
        n = self.complex.num_vertices
        dist = np.full((n, n), np.inf, dtype=np.float64)
        np.fill_diagonal(dist, 0.0)

        for u in range(n):
            for v in self.complex.adj.get(u, set()):
                dist[u, v] = 1.0
                dist[v, u] = 1.0

        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        return dist

    def compute_neighborhood_measure(self, u: int) -> Dict[int, float]:
        neighbors = list(self.complex.adj.get(u, set()))
        deg = len(neighbors)
        if deg == 0:
            return {u: 1.0}

        measure = {u: self.alpha}
        neighbor_prob = (1.0 - self.alpha) / deg
        for v in neighbors:
            measure[v] = neighbor_prob
        return measure

    def compute_wasserstein_distance(
        self,
        u: int,
        v: int,
        m_u: Dict[int, float],
        m_v: Dict[int, float],
        dist_matrix: np.ndarray
    ) -> float:
        supp_u = list(m_u.keys())
        supp_v = list(m_v.keys())
        p_u = np.array([m_u[x] for x in supp_u], dtype=np.float64)
        p_v = np.array([m_v[y] for y in supp_v], dtype=np.float64)

        cost = np.zeros((len(supp_u), len(supp_v)), dtype=np.float64)
        for i, x in enumerate(supp_u):
            for j, y in enumerate(supp_v):
                cost[i, j] = dist_matrix[x, y]

        c = cost.flatten()
        n_u, n_v = len(supp_u), len(supp_v)

        A_eq = np.zeros((n_u + n_v, n_u * n_v), dtype=np.float64)
        b_eq = np.concatenate([p_u, p_v])

        for i in range(n_u):
            for j in range(n_v):
                idx = i * n_v + j
                A_eq[i, idx] = 1.0
                A_eq[n_u + j, idx] = 1.0

        res = opt.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
        if res.success:
            return float(res.fun)
        return float(dist_matrix[u, v])

    def analyze_curvatures(self) -> RicciPassDiagnostic:
        n = self.complex.num_vertices
        if n <= 1:
            return RicciPassDiagnostic(
                num_nodes=n, num_edges=0, mean_curvature=0.0,
                bottlenecks_detected=0, fusion_clusters_detected=0,
                fusion_candidates=[]
            )

        dist = self.compute_shortest_paths()
        curvatures = []
        fusion_candidates = []
        bottlenecks = 0

        # Scan each undirected edge
        visited_edges = set()
        for u in range(n):
            m_u = self.compute_neighborhood_measure(u)
            for v in self.complex.adj.get(u, set()):
                edge_pair = (min(u, v), max(u, v))
                if edge_pair in visited_edges:
                    continue
                visited_edges.add(edge_pair)

                d_uv = dist[u, v]
                if np.isinf(d_uv) or d_uv <= 0.0:
                    continue

                m_v = self.compute_neighborhood_measure(v)
                w1 = self.compute_wasserstein_distance(u, v, m_u, m_v, dist)
                kappa = 1.0 - w1 / d_uv
                curvatures.append(kappa)

                u_sym = self.complex.get_vertex(u).id
                v_sym = self.complex.get_vertex(v).id

                if kappa > 0.2:
                    fusion_candidates.append((u_sym, v_sym, float(kappa)))
                elif kappa < -0.1:
                    bottlenecks += 1

        mean_curv = float(np.mean(curvatures)) if curvatures else 0.0
        return RicciPassDiagnostic(
            num_nodes=n,
            num_edges=len(visited_edges),
            mean_curvature=mean_curv,
            bottlenecks_detected=bottlenecks,
            fusion_clusters_detected=len(fusion_candidates),
            fusion_candidates=fusion_candidates
        )
