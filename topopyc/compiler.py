"""
Production Compiler Driver and JIT Decorator.
Orchestrates:
1. AST parsing and Complex construction
2. Sheaf type deduction and Hodge Cohomology Analysis
3. GCC compilation with -O3 -march=native -ffast-math
4. Dynamic loading into Python runtime or standalone executable compilation.
"""

import ast
import inspect
import os
import subprocess
import sys
import sysconfig
import tempfile
import importlib.util
from typing import Callable, Any, Dict, Optional, Tuple

from .simplicial import SimplicialComplex
from .sheaf import TypeSheaf, StalkType
from .hodge import HodgeLaplacianSolver, CohomologyAnalysis
from .codegen import NativeCCodeGenerator

class TopoCompiler:
    def __init__(self, opt_level: str = "-O3"):
        self.opt_level = opt_level
        self.include_dir = sysconfig.get_paths()["include"]

    def _type_from_annotation(self, ann: Any) -> StalkType:
        if ann in (float, "float"):
            return StalkType.FLOAT64
        elif ann in (int, "int"):
            return StalkType.INT64
        elif ann in (bool, "bool"):
            return StalkType.BOOL
        return StalkType.FLOAT64

    def analyze_function(self, fn: Callable) -> Tuple[SimplicialComplex, TypeSheaf, CohomologyAnalysis, ast.FunctionDef, StalkType, Dict[str, StalkType]]:
        source = inspect.getsource(fn)
        # Strip leading indentation if any
        lines = source.splitlines()
        first_line = lines[0]
        indent = len(first_line) - len(first_line.lstrip())
        dedented_source = "\n".join(l[indent:] if len(l) >= indent else l for l in lines)
        
        tree = ast.parse(dedented_source)
        fn_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                fn_node = node
                break
        if not fn_node:
            raise ValueError(f"Could not locate FunctionDef in source of {fn}")

        # Build simplicial complex
        complex = SimplicialComplex(name=fn.__name__)
        sheaf = TypeSheaf(complex)

        # Map arguments
        sig = inspect.signature(fn)
        param_types: Dict[str, StalkType] = {}
        for param in sig.parameters.values():
            stype = self._type_from_annotation(param.annotation)
            param_types[param.name] = stype
            v_idx = complex.add_simplex(0, f"arg_{param.name}", "Argument", {"type": stype})
            sheaf.assign_stalk(complex.get_vertex(v_idx), stype)

        # Return type
        ret_type = self._type_from_annotation(sig.return_annotation)

        # Walk AST to construct dataflow edges and operation triangles
        for node in ast.walk(fn_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        v_idx = complex.add_simplex(0, f"var_{target.id}", "Variable")
                        # Infer local type
                        assigned_type = StalkType.FLOAT64
                        if isinstance(node.value, ast.Constant):
                            if isinstance(node.value.value, int):
                                assigned_type = StalkType.INT64
                            elif isinstance(node.value.value, float):
                                assigned_type = StalkType.FLOAT64
                        sheaf.assign_stalk(complex.get_vertex(v_idx), assigned_type)
            elif isinstance(node, ast.BinOp):
                left_id = node.left.id if isinstance(node.left, ast.Name) else f"left_{node.lineno}"
                right_id = node.right.id if isinstance(node.right, ast.Name) else f"right_{node.lineno}"
                v_left = complex.add_simplex(0, f"var_{left_id}", "Operand")
                v_right = complex.add_simplex(0, f"var_{right_id}", "Operand")
                v_op = complex.add_simplex(0, f"op_{type(node.op).__name__}_{node.lineno}", "Operator")
                complex.add_triangle(v_left, v_right, v_op, f"tri_{node.lineno}")

        solver = HodgeLaplacianSolver(sheaf)
        analysis = solver.analyze()
        return complex, sheaf, analysis, fn_node, ret_type, param_types

    def compile_extension(self, fn: Callable, output_dir: Optional[str] = None) -> str:
        """
        Compiles function into a native CPython C extension (.so) and returns shared library path.
        """
        complex, sheaf, analysis, fn_node, ret_type, param_types = self.analyze_function(fn)
        codegen = NativeCCodeGenerator(sheaf, analysis)
        # Populate full type environment from sheaf stalks
        full_env = dict(param_types)
        for s, st in sheaf.stalks.items():
            if s.dim == 0 and s.id.startswith("var_"):
                var_name = s.id[4:]
                full_env[var_name] = st
        codegen.set_type_env(full_env)

        module_name = f"topo_{fn.__name__}"
        c_code = codegen.generate_cpython_extension(module_name, fn_node, ret_type, param_types)

        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="topopyc_")

        c_file_path = os.path.join(output_dir, f"{module_name}.c")
        so_file_path = os.path.join(output_dir, f"{module_name}.so")

        with open(c_file_path, "w") as f:
            f.write(c_code)

        # Invoke GCC
        cmd = [
            "gcc",
            self.opt_level,
            "-march=native",
            "-ffast-math",
            "-shared",
            "-fPIC",
            f"-I{self.include_dir}",
            c_file_path,
            "-lm",
            "-o",
            so_file_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"TopoPyc GCC compilation failed:\n{res.stderr}")

        return so_file_path

    def compile_executable(self, fn: Callable, output_path: str, bench_args: Tuple = ()) -> str:
        """
        Compiles standalone native executable with embedded main benchmark harness.
        """
        complex, sheaf, analysis, fn_node, ret_type, param_types = self.analyze_function(fn)
        codegen = NativeCCodeGenerator(sheaf, analysis)
        full_env = dict(param_types)
        for s, st in sheaf.stalks.items():
            if s.dim == 0 and s.id.startswith("var_"):
                var_name = s.id[4:]
                full_env[var_name] = st
        codegen.set_type_env(full_env)

        fn_c_code = codegen.generate_function(fn_node, ret_type, param_types)
        fn_name = fn_node.name

        call_args_str = ", ".join(str(a) for a in bench_args)

        main_c = f"""
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

{fn_c_code}

int main() {{
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    {ret_type.c_type()} res = topopyc_{fn_name}({call_args_str});
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1000000.0;
    printf("[TopoPyc Standalone] Kernel: {fn_name} | Elapsed: %.3f ms | Result: %.6f\\n", elapsed_ms, (double)res);
    return 0;
}}
"""
        c_src_path = output_path + ".c"
        with open(c_src_path, "w") as f:
            f.write(main_c)

        cmd = [
            "gcc",
            self.opt_level,
            "-march=native",
            "-ffast-math",
            c_src_path,
            "-lm",
            "-o",
            output_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"TopoPyc standalone compilation failed:\n{res.stderr}")

        return output_path

import functools

def topojit(fn: Optional[Callable] = None, *, opt_level: str = "-O3") -> Callable:
    """
    Decorator for high-performance JIT compilation of numerical kernels.
    Transforms Python functions into dynamically linked native kernels.
    Can be used as @topojit or @topojit(opt_level="-Ofast").
    """
    def decorator(target_fn: Callable) -> Callable:
        compiler = TopoCompiler(opt_level=opt_level)
        so_path = compiler.compile_extension(target_fn)
        module_name = f"topo_{target_fn.__name__}"

        spec = importlib.util.spec_from_file_location(module_name, so_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load compiled module from {so_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        native_func = getattr(mod, target_fn.__name__)
        
        @functools.wraps(target_fn)
        def wrapper(*args, **kwargs):
            return native_func(*args, **kwargs)

        wrapper.__native__ = native_func
        wrapper.__so_path__ = so_path
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator
