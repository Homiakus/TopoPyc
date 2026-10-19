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
    parser = argparse.ArgumentParser(
        prog="topopyc",
        description="⚡ TopoPyc: Topological Sheaf Python Compiler - Compile Python into high-performance native machine code."
    )
    parser.add_argument("input_file", help="Python source file to compile (.py)")
    parser.add_argument("-o", "--output", help="Output path for compiled binary or shared library", default=None)
    parser.add_argument("--so", action="store_true", help="Compile as CPython shared extension library (.so)")
    parser.add_argument("--emit-c", action="store_true", help="Emit raw generated unboxed C source code")
    parser.add_argument("--analyze", action="store_true", help="Perform Hodge cohomology analysis and print Betti numbers")
    parser.add_argument("-O", "--opt-level", default="-O3", help="Optimization level: -O2, -O3, -Ofast (default: -O3)")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"❌ Error: file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    print(f"⚡ TopoPyc: Compiling '{args.input_file}' with optimization {args.opt_level}...")
    compiler = TopoCompiler(opt_level=args.opt_level)

    try:
        out = compiler.compile_file(
            file_path=args.input_file,
            output_path=args.output,
            emit_c=args.emit_c,
            as_so=args.so
        )
        if args.emit_c:
            print(f"✅ Generated C source code: {out}")
        else:
            kind = "Shared extension (.so)" if args.so else "Standalone binary"
            print(f"✅ Successfully compiled {kind}: {out}")
    except Exception as e:
        print(f"❌ Compilation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
