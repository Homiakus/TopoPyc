"""
Type Sheaf Construction over Simplicial Complex.
Assigns typed stalks (vector spaces) F(sigma) to each cell sigma
and restriction morphisms rho_{sigma <= tau} : F(sigma) -> F(tau).
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from .simplicial import SimplicialComplex, Simplex

class StalkType(Enum):
    UNKNOWN = auto()
    INT64 = auto()
    FLOAT64 = auto()
    BOOL = auto()
    STRING = auto()
    LIST_FLOAT = auto()
    LIST_INT = auto()
    VOID = auto()

    def c_type(self) -> str:
        mapping = {
            StalkType.INT64: "int64_t",
            StalkType.FLOAT64: "double",
            StalkType.BOOL: "bool",
            StalkType.STRING: "const char*",
            StalkType.LIST_FLOAT: "double*",
            StalkType.LIST_INT: "int64_t*",
            StalkType.VOID: "void",
            StalkType.UNKNOWN: "PyObject*"
        }
        return mapping.get(self, "PyObject*")

    def dimension(self) -> int:
        if self in (StalkType.INT64, StalkType.FLOAT64, StalkType.BOOL):
            return 1
        elif self in (StalkType.LIST_FLOAT, StalkType.LIST_INT):
            return 2  # (ptr, length)
        return 4

@dataclass
class RestrictionMap:
    src_simplex: Simplex
    dst_simplex: Simplex
    # Transfer matrix representation in basis of stalk vector spaces
    matrix: np.ndarray
    coercion_safe: bool = True

class TypeSheaf:
    """
    Cellular Sheaf F over simplicial complex X.
    Attaches type stalk vector spaces and restriction operators.
    """
    def __init__(self, complex: SimplicialComplex):
        self.complex = complex
        # Simplex -> StalkType
        self.stalks: Dict[Simplex, StalkType] = {}
        # (k, tau_idx, sigma_idx) -> RestrictionMap
        self.restrictions: Dict[Tuple[int, int, int], RestrictionMap] = {}

    def assign_stalk(self, simplex: Simplex, stype: StalkType):
        self.stalks[simplex] = stype

    def get_stalk(self, simplex: Simplex) -> StalkType:
        return self.stalks.get(simplex, StalkType.UNKNOWN)

    def set_restriction(self, dim: int, src_idx: int, dst_idx: int, matrix: Optional[np.ndarray] = None):
        src_s = self.complex.simplices[dim - 1][src_idx] if dim > 0 else self.complex.simplices[0][src_idx]
        dst_s = self.complex.simplices[dim][dst_idx]
        src_dim = self.get_stalk(src_s).dimension()
        dst_dim = self.get_stalk(dst_s).dimension()
        
        if matrix is None:
            if src_dim == dst_dim:
                matrix = np.eye(src_dim, dtype=np.float64)
            else:
                matrix = np.zeros((dst_dim, src_dim), dtype=np.float64)
                min_d = min(src_dim, dst_dim)
                matrix[:min_d, :min_d] = np.eye(min_d)

        self.restrictions[(dim, dst_idx, src_idx)] = RestrictionMap(
            src_simplex=src_s,
            dst_simplex=dst_s,
            matrix=matrix,
            coercion_safe=True
        )

    def infer_coercion(self, t1: StalkType, t2: StalkType) -> StalkType:
        """Computes minimal upper bound (supremum) in type poset."""
        if t1 == t2:
            return t1
        if {t1, t2} == {StalkType.INT64, StalkType.FLOAT64}:
            return StalkType.FLOAT64
        if {t1, t2} == {StalkType.BOOL, StalkType.INT64}:
            return StalkType.INT64
        return StalkType.UNKNOWN
