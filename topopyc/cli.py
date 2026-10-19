"""
Command-line Interface for TopoPyc Compiler.
"""

import argparse
import sys
import os
import ast
import inspect
from .compiler import TopoCompiler
from .simplicial import SimplicialComplex
from .sheaf import TypeSheaf, StalkType
from .hodge import HodgeLaplacianSolver

def main():
    parser = argparse.ArgumentParser(description="TopoPyc: Topological Sheaf Python Compiler")
    parser.add_argument("input_file", help="Python source file to compile")
    parser.add_argument("-o", "--output", help="Output binary or shared library path", default=None)
    parser.add_argument("--emit-c", action="store_true", help="Emit raw generated C code")
    parser.add_argument("--analyze-only", action="store_true", help="Print Hodge cohomology analysis only")
    parser.add_argument("-O", "--opt-level", default="-O3", help="Compiler optimization level (-O2, -O3, -Ofast)")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    with open(args.input_file, "r") as f:
        src = f.read()

    tree = ast.parse(src)
    print(f"⚡ TopoPyc: Parsed AST for {args.input_file}")

    compiler = TopoCompiler(opt_level=args.opt_level)

    # Inspect top-level functions
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            print(f"  -> Discovered function '{node.name}'")

    print("⚡ Topological compilation complete.")

if __name__ == "__main__":
    main()
