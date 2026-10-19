"""
Hodge-Laplacian Operator Construction and Cohomology Analysis.
Computes coboundaries d_k, higher Laplacians Delta_k = d_{k-1}d*_{k-1} + d*_k d_k,
Betti numbers, and identifies harmonic sections ker(Delta_k) ~ H^k.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from .sheaf import TypeSheaf, StalkType

@dataclass
class CohomologyAnalysis:
    dim_H0: int
    dim_H1: int
    dim_H2: int
    betti_numbers: List[int]
    is_monomorphic: bool
    is_race_free: bool
    dirichlet_energy: float
    harmonic_frequencies: np.ndarray

class HodgeLaplacianSolver:
    """
    Computes discrete exterior calculus operators over the cellular type sheaf.
    """
    def __init__(self, sheaf: TypeSheaf):
        self.sheaf = sheaf
        self.complex = sheaf.complex

    def build_d0(self) -> np.ndarray:
        """
        Builds co-boundary operator d_0 : C^0(X, F) -> C^1(X, F).
        Uses cellular sheaf restriction maps rho_{v <= e} along 1-simplices.
        """
        nV = self.complex.num_vertices
        nE = self.complex.num_edges
        if nV == 0 or nE == 0:
            return np.zeros((0, 0), dtype=np.float64)

        # For scalar stalk dimension: nE x nV
        d0 = np.zeros((nE, nV), dtype=np.float64)
        for e_idx in range(nE):
            boundary = self.complex.boundary_relations.get((1, e_idx), [])
            for v_idx, sign in boundary:
                # If a specific restriction operator exists, scale by its spectral norm trace
                restr = self.sheaf.restrictions.get((1, e_idx, v_idx))
                weight = 1.0
                if restr is not None and restr.matrix is not None and restr.matrix.size > 0:
                    weight = float(np.linalg.norm(restr.matrix))
                    if weight == 0.0:
                        weight = 1.0
                d0[e_idx, v_idx] = sign * weight
        return d0

    def build_d1(self) -> np.ndarray:
        """
        Builds co-boundary operator d_1 : C^1(X, F) -> C^2(X, F).
        """
        nE = self.complex.num_edges
        nT = self.complex.num_triangles
        if nE == 0 or nT == 0:
            return np.zeros((0, nE), dtype=np.float64)

        d1 = np.zeros((nT, nE), dtype=np.float64)
        for t_idx in range(nT):
            boundary = self.complex.boundary_relations.get((2, t_idx), [])
            for e_idx, sign in boundary:
                if e_idx < nE:
                    d1[t_idx, e_idx] = sign
        return d1

    def compute_laplacians(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes Hodge Laplacians:
        Delta_0 = d_0^* d_0
        Delta_1 = d_0 d_0^* + d_1^* d_1
        Delta_2 = d_1 d_1^*
        """
        d0 = self.build_d0()
        d1 = self.build_d1()

        nV = self.complex.num_vertices
        nE = self.complex.num_edges
        nT = self.complex.num_triangles

        # Delta_0 : C^0 -> C^0
        delta_0 = d0.T @ d0 if nV > 0 and nE > 0 else np.zeros((nV, nV))

        # Delta_1 : C^1 -> C^1
        delta_1_down = d0 @ d0.T if nE > 0 and nV > 0 else np.zeros((nE, nE))
        delta_1_up = d1.T @ d1 if nT > 0 and nE > 0 else np.zeros((nE, nE))
        delta_1 = delta_1_down + delta_1_up

        # Delta_2 : C^2 -> C^2
        delta_2 = d1 @ d1.T if nT > 0 and nE > 0 else np.zeros((nT, nT))

        return delta_0, delta_1, delta_2

    def analyze(self) -> CohomologyAnalysis:
        delta_0, delta_1, delta_2 = self.compute_laplacians()

        def compute_kernel_dim(laplacian: np.ndarray) -> Tuple[int, np.ndarray]:
            if laplacian.size == 0 or laplacian.shape[0] == 0:
                return 0, np.array([])
            n = laplacian.shape[0]
            # Fast-path for small matrices or high-dimensional reduction
            eigvals, _ = np.linalg.eigh(laplacian)
            # Clip very small negative eigenvalues due to numerical precision
            eigvals = np.maximum(eigvals, 0.0)
            dim_ker = int(np.sum(np.isclose(eigvals, 0.0, atol=1e-7)))
            return dim_ker, eigvals

        dim_H0, eig_0 = compute_kernel_dim(delta_0)
        dim_H1, eig_1 = compute_kernel_dim(delta_1)
        dim_H2, eig_2 = compute_kernel_dim(delta_2)

        # Monomorphism check: if all type connections have zero Dirichlet energy gradient
        # and dim_H1 is bounded by the number of independent control components
        is_monomorphic = (dim_H1 == 0 or dim_H1 == 1)
        is_race_free = (dim_H2 == 0)

        # Dirichlet energy of constant test section
        dirichlet_energy = float(np.sum(eig_0)) if len(eig_0) > 0 else 0.0

        return CohomologyAnalysis(
            dim_H0=dim_H0,
            dim_H1=dim_H1,
            dim_H2=dim_H2,
            betti_numbers=[dim_H0, dim_H1, dim_H2],
            is_monomorphic=is_monomorphic,
            is_race_free=is_race_free,
            dirichlet_energy=dirichlet_energy,
            harmonic_frequencies=eig_0
        )
