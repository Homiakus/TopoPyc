"""
Topological Sheaf Python Compiler (TopoPyc)
Production-grade AOT & JIT Native Python Compiler based on Multidimensional Topological Holography.
"""

__version__ = "0.1.0"

from .simplicial import SimplicialComplex, Simplex
from .sheaf import TypeSheaf, StalkType, RestrictionMap
from .hodge import HodgeLaplacianSolver, CohomologyAnalysis
from .codegen import NativeCCodeGenerator
from .optimizer import TopologicalOptimizer, OptimizationStats
from .compiler import TopoCompiler, topojit
from .ricci import SimplicialRicciOptimizer, RicciPassDiagnostic
from .symplectic import SimplecticMemoryPass, SymplecticMemoryDiagnostic
from .dspace import DSpaceConcurrencyPass, DeadlockTopologicalReport
from .optics import TopoLens, record_lens, array_lens

__all__ = [
    "SimplicialComplex",
    "Simplex",
    "TypeSheaf",
    "StalkType",
    "RestrictionMap",
    "HodgeLaplacianSolver",
    "CohomologyAnalysis",
    "TopologicalOptimizer",
    "OptimizationStats",
    "NativeCCodeGenerator",
    "TopoCompiler",
    "topojit",
    "SimplicialRicciOptimizer",
    "RicciPassDiagnostic",
    "SimplecticMemoryPass",
    "SymplecticMemoryDiagnostic",
    "DSpaceConcurrencyPass",
    "DeadlockTopologicalReport",
    "TopoLens",
    "record_lens",
    "array_lens",
]
