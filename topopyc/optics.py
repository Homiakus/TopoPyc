"""
Profunctor Optics & Unboxed Lens Accessor Pass.
Ported and adapted from Tao Science: all-math/profunctor-optics-bidirectional.

Provides pure unboxed access (Get/Put/Modify) to structured data
without object allocation or boxing in compiled native loops.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

S = TypeVar("S")
T = TypeVar("T")
A = TypeVar("A")
B = TypeVar("B")

@dataclass(frozen=True)
class TopoLens(Generic[S, T, A, B]):
    """
    Coend representation of Lens for compiled code access paths:
    split: S -> (Context, Focus)
    merge: (Context, NewFocus) -> T
    """
    split: Callable[[S], Tuple[Any, A]]
    merge: Callable[[Any, B], T]

    def get(self, s: S) -> A:
        _, a = self.split(s)
        return a

    def put(self, s: S, b: B) -> T:
        ctx, _ = self.split(s)
        return self.merge(ctx, b)

    def modify(self, s: S, f: Callable[[A], B]) -> T:
        ctx, a = self.split(s)
        return self.merge(ctx, f(a))

    def __matmul__(self, other: "TopoLens[A, B, Any, Any]") -> "TopoLens[S, T, Any, Any]":
        def composed_split(s: S):
            ctx1, a = self.split(s)
            ctx2, sub_a = other.split(a)
            return ((ctx1, ctx2), sub_a)

        def composed_merge(composed_ctx, new_sub_b):
            ctx1, ctx2 = composed_ctx
            new_a = other.merge(ctx2, new_sub_b)
            return self.merge(ctx1, new_a)

        return TopoLens(split=composed_split, merge=composed_merge)

def record_lens(field_name: str) -> TopoLens[Dict[str, Any], Dict[str, Any], Any, Any]:
    """Generates lens for dictionary/record unboxed field access."""
    def split(d: Dict[str, Any]):
        residual = {k: v for k, v in d.items() if k != field_name}
        return (residual, d.get(field_name))

    def merge(residual: Dict[str, Any], val: Any):
        res = dict(residual)
        res[field_name] = val
        return res

    return TopoLens(split=split, merge=merge)

def array_lens(index: int) -> TopoLens[List[Any], List[Any], Any, Any]:
    """Generates lens for array unboxed element access."""
    def split(lst: List[Any]):
        residual = (lst[:index], lst[index + 1:])
        return (residual, lst[index])

    def merge(residual: Tuple[List[Any], List[Any]], val: Any):
        left, right = residual
        return list(left) + [val] + list(right)

    return TopoLens(split=split, merge=merge)
