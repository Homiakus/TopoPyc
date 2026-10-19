"""
Topological Sheaf Python Compiler (TopoPyc)
Production-grade AOT & JIT Native Python Compiler based on Multidimensional Topological Holography.
"""

__version__ = "0.1.0"

from .simplicial import SimplicialComplex, Simplex
from .sheaf import TypeSheaf, StalkType, RestrictionMap
from .hodge import HodgeLaplacianSolver, CohomologyAnalysis
from .codegen import NativeCCodeGenerator
from .compiler import TopoCompiler, topojit

__all__ = [
    "SimplicialComplex",
    "Simplex",
    "TypeSheaf",
    "StalkType",
    "RestrictionMap",
    "HodgeLaplacianSolver",
    "CohomologyAnalysis",
    "NativeCCodeGenerator",
    "TopoCompiler",
    "topojit",
]
