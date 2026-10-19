"""
Simplicial complex data structures and AST topological mapper.
Constructs graded k-simplexes (0D vertices, 1D dataflow edges, 2D operator faces, 3D concurrency cycles).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
import ast

@dataclass(frozen=True)
class Simplex:
    dim: int
    id: str
    kind: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.dim, self.id))

    def __eq__(self, other):
        if not isinstance(other, Simplex):
            return False
        return self.dim == other.dim and self.id == other.id

class SimplicialComplex:
    """
    Graded Simplicial Complex X = union_k X^(k) representing program syntax,
    dataflow, lexical scope, and concurrency interactions.
    """
    def __init__(self, name: str = "main"):
        self.name = name
        # k -> list of simplexes
        self.simplices: Dict[int, List[Simplex]] = {0: [], 1: [], 2: [], 3: []}
        # simplex -> global index in that dimension
        self.simplex_to_idx: Dict[Simplex, int] = {}
        # Incidence relations: (k, tau) -> list of (sigma, incidence_sign) where sigma in X^(k-1)
        self.boundary_relations: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}
        # Adjacency maps for quick graph traversals
        self.adj: Dict[int, Set[int]] = {}

    def add_simplex(self, dim: int, sim_id: str, kind: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        s = Simplex(dim=dim, id=sim_id, kind=kind, metadata=metadata or {})
        if s in self.simplex_to_idx:
            return self.simplex_to_idx[s]
        
        if dim not in self.simplices:
            self.simplices[dim] = []
        
        idx = len(self.simplices[dim])
        self.simplices[dim].append(s)
        self.simplex_to_idx[s] = idx
        if dim == 0:
            self.adj[idx] = set()
        return idx

    def add_edge(self, u_idx: int, v_idx: int, edge_id: str, kind: str = "dataflow", metadata: Optional[Dict[str, Any]] = None) -> int:
        """Adds a 1-simplex between vertex u and vertex v with orientation u -> v."""
        e_idx = self.add_simplex(dim=1, sim_id=edge_id, kind=kind, metadata=metadata)
        # Boundary: d(e) = v - u
        self.boundary_relations[(1, e_idx)] = [(u_idx, -1.0), (v_idx, 1.0)]
        self.adj[u_idx].add(v_idx)
        return e_idx

    def add_triangle(self, u_idx: int, v_idx: int, w_idx: int, tri_id: str, kind: str = "closure", metadata: Optional[Dict[str, Any]] = None) -> int:
        """Adds a 2-simplex (triangle) bounded by vertices u, v, w."""
        t_idx = self.add_simplex(dim=2, sim_id=tri_id, kind=kind, metadata=metadata)
        # Find or create corresponding edges: (u,v), (v,w), (u,w)
        e0 = self.add_edge(u_idx, v_idx, f"e_{u_idx}_{v_idx}", kind=f"{kind}_e0")
        e1 = self.add_edge(v_idx, w_idx, f"e_{v_idx}_{w_idx}", kind=f"{kind}_e1")
        e2 = self.add_edge(u_idx, w_idx, f"e_{u_idx}_{w_idx}", kind=f"{kind}_e2")
        # Boundary in C^1: e0 + e1 - e2
        self.boundary_relations[(2, t_idx)] = [(e0, 1.0), (e1, 1.0), (e2, -1.0)]
        return t_idx

    def get_vertex(self, idx: int) -> Simplex:
        return self.simplices[0][idx]

    def get_edge(self, idx: int) -> Simplex:
        return self.simplices[1][idx]

    def get_triangle(self, idx: int) -> Simplex:
        return self.simplices[2][idx]

    @property
    def num_vertices(self) -> int:
        return len(self.simplices.get(0, []))

    @property
    def num_edges(self) -> int:
        return len(self.simplices.get(1, []))

    @property
    def num_triangles(self) -> int:
        return len(self.simplices.get(2, []))
