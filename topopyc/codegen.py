"""
Native C & CPython C-API Extension Generator.
Emits clean, high-performance unboxed C code, supporting both raw CLI executables
and importable CPython shared libraries (.so).
"""

import ast
from typing import Dict, List, Optional, Set, Any
from .sheaf import TypeSheaf, StalkType
from .hodge import CohomologyAnalysis

class NativeCCodeGenerator:
    def __init__(self, sheaf: TypeSheaf, analysis: CohomologyAnalysis):
        self.sheaf = sheaf
        self.analysis = analysis
        self.type_env: Dict[str, StalkType] = {}

    def set_type_env(self, env: Dict[str, StalkType]):
        self.type_env = env

    def _infer_expr_type(self, node: ast.AST) -> StalkType:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int):
                return StalkType.INT64
            elif isinstance(node.value, float):
                return StalkType.FLOAT64
            elif isinstance(node.value, bool):
                return StalkType.BOOL
        elif isinstance(node, ast.Name):
            return self.type_env.get(node.id, StalkType.UNKNOWN)
        elif isinstance(node, ast.BinOp):
            lt = self._infer_expr_type(node.left)
            rt = self._infer_expr_type(node.right)
            if lt == StalkType.FLOAT64 or rt == StalkType.FLOAT64:
                return StalkType.FLOAT64
            if isinstance(node.op, ast.Div):
                return StalkType.FLOAT64
            if lt == StalkType.INT64 and rt == StalkType.INT64:
                return StalkType.INT64
        return StalkType.UNKNOWN

    def map_binop(self, op: ast.operator) -> str:
        if isinstance(op, ast.Add): return "+"
        if isinstance(op, ast.Sub): return "-"
        if isinstance(op, ast.Mult): return "*"
        if isinstance(op, ast.Div): return "/"
        if isinstance(op, ast.FloorDiv): return "/"
        if isinstance(op, ast.Mod): return "%"
        return "+"

    def map_cmpop(self, op: ast.cmpop) -> str:
        if isinstance(op, ast.Eq): return "=="
        if isinstance(op, ast.NotEq): return "!="
        if isinstance(op, ast.Lt): return "<"
        if isinstance(op, ast.LtE): return "<="
        if isinstance(op, ast.Gt): return ">"
        if isinstance(op, ast.GtE): return ">="
        return "=="

    def compile_expr(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "1" if node.value else "0"
            elif isinstance(node.value, float):
                return str(node.value)
            elif isinstance(node.value, int):
                return str(node.value)
            elif isinstance(node.value, str):
                return f'"{node.value}"'
            return "0"
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.BinOp):
            left_str = self.compile_expr(node.left)
            right_str = self.compile_expr(node.right)
            if isinstance(node.op, ast.Mod):
                # If both are known integer types, use native C % for max performance
                left_type = self._infer_expr_type(node.left)
                right_type = self._infer_expr_type(node.right)
                if left_type == StalkType.INT64 and right_type == StalkType.INT64:
                    return f"({left_str} % {right_str})"
                return f"fmod((double)({left_str}), (double)({right_str}))"
            op_str = self.map_binop(node.op)
            return f"({left_str} {op_str} {right_str})"
        elif isinstance(node, ast.UnaryOp):
            operand_str = self.compile_expr(node.operand)
            if isinstance(node.op, ast.USub):
                return f"(-{operand_str})"
            if isinstance(node.op, ast.Not):
                return f"(!{operand_str})"
            return operand_str
        elif isinstance(node, ast.Compare):
            left_str = self.compile_expr(node.left)
            res = left_str
            for op, comparator in zip(node.ops, node.comparators):
                cmp_str = self.map_cmpop(op)
                right_str = self.compile_expr(comparator)
                res = f"({res} {cmp_str} {right_str})"
            return res
        elif isinstance(node, ast.Call):
            # Math built-ins
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            
            args_str = ", ".join(self.compile_expr(a) for a in node.args)
            if func_name in ("sin", "cos", "tan", "sqrt", "exp", "log", "fabs"):
                return f"{func_name}({args_str})"
            elif func_name == "float":
                return f"((double)({args_str}))"
            elif func_name == "int":
                return f"((int64_t)({args_str}))"
            return f"{func_name}({args_str})"
        return "0"

    def compile_stmt(self, node: ast.AST, indent: int = 4) -> List[str]:
        pad = " " * indent
        lines = []
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id
                    val_expr = self.compile_expr(node.value)
                    lines.append(f"{pad}{var_name} = {val_expr};")
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                var_name = node.target.id
                op_str = self.map_binop(node.op)
                val_expr = self.compile_expr(node.value)
                lines.append(f"{pad}{var_name} {op_str}= {val_expr};")
        elif isinstance(node, ast.Return):
            if node.value:
                val_expr = self.compile_expr(node.value)
                lines.append(f"{pad}return {val_expr};")
            else:
                lines.append(f"{pad}return;")
        elif isinstance(node, ast.For):
            # Handle for i in range(...)
            if isinstance(node.iter, ast.Call) and isinstance(node.target, ast.Name):
                func_name = node.iter.func.id if isinstance(node.iter.func, ast.Name) else ""
                var_name = node.target.id
                if func_name == "range":
                    args = node.iter.args
                    if len(args) == 1:
                        stop_expr = self.compile_expr(args[0])
                        lines.append(f"{pad}for (int64_t {var_name} = 0; {var_name} < {stop_expr}; ++{var_name}) {{")
                    elif len(args) == 2:
                        start_expr = self.compile_expr(args[0])
                        stop_expr = self.compile_expr(args[1])
                        lines.append(f"{pad}for (int64_t {var_name} = {start_expr}; {var_name} < {stop_expr}; ++{var_name}) {{")
                    elif len(args) == 3:
                        start_expr = self.compile_expr(args[0])
                        stop_expr = self.compile_expr(args[1])
                        step_expr = self.compile_expr(args[2])
                        lines.append(f"{pad}for (int64_t {var_name} = {start_expr}; {var_name} < {stop_expr}; {var_name} += {step_expr}) {{")
                    for body_stmt in node.body:
                        lines.extend(self.compile_stmt(body_stmt, indent + 4))
                    lines.append(f"{pad}}}")
        elif isinstance(node, ast.While):
            test_expr = self.compile_expr(node.test)
            lines.append(f"{pad}while ({test_expr}) {{")
            for body_stmt in node.body:
                lines.extend(self.compile_stmt(body_stmt, indent + 4))
            lines.append(f"{pad}}}")
        elif isinstance(node, ast.If):
            test_expr = self.compile_expr(node.test)
            lines.append(f"{pad}if ({test_expr}) {{")
            for body_stmt in node.body:
                lines.extend(self.compile_stmt(body_stmt, indent + 4))
            if node.orelse:
                lines.append(f"{pad}}} else {{")
                for else_stmt in node.orelse:
                    lines.extend(self.compile_stmt(else_stmt, indent + 4))
            lines.append(f"{pad}}}")
        elif isinstance(node, ast.Break):
            lines.append(f"{pad}break;")
        elif isinstance(node, ast.Continue):
            lines.append(f"{pad}continue;")
        elif isinstance(node, ast.Pass):
            pass
        return lines

    def generate_function(self, fn_node: ast.FunctionDef, return_type: StalkType, param_types: Dict[str, StalkType]) -> str:
        # Collect local variable declarations
        local_vars: Dict[str, StalkType] = {}
        for sub in ast.walk(fn_node):
            if isinstance(sub, ast.Assign):
                for tgt in sub.targets:
                    if isinstance(tgt, ast.Name) and tgt.id not in param_types:
                        local_vars[tgt.id] = self.type_env.get(tgt.id, StalkType.FLOAT64)
            elif isinstance(sub, ast.For):
                if isinstance(sub.target, ast.Name) and sub.target.id not in param_types:
                    local_vars[sub.target.id] = StalkType.INT64

        params_list = []
        for arg in fn_node.args.args:
            p_type = param_types.get(arg.arg, StalkType.FLOAT64).c_type()
            params_list.append(f"{p_type} {arg.arg}")
        params_str = ", ".join(params_list)

        c_ret = return_type.c_type()
        fn_name = fn_node.name

        body_lines = []
        # Emit variable declarations
        for vname, vtype in local_vars.items():
            if vtype == StalkType.INT64:
                init_val = "0"
            elif vtype == StalkType.BOOL:
                init_val = "false"
            else:
                init_val = "0.0"
            body_lines.append(f"    {vtype.c_type()} {vname} = {init_val};")

        # Emit statements (skipping docstrings or simple pass)
        for stmt in fn_node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                continue
            body_lines.extend(self.compile_stmt(stmt, indent=4))

        decl = f"{c_ret} topopyc_{fn_name}({params_str}) {{"
        full_code = [decl] + body_lines + ["}"]
        return "\n".join(full_code)

    def generate_cpython_extension(self, module_name: str, fn_node: ast.FunctionDef, return_type: StalkType, param_types: Dict[str, StalkType]) -> str:
        fn_name = fn_node.name
        native_fn_code = self.generate_function(fn_node, return_type, param_types)

        # Build PyArg_ParseTuple format
        parse_format = ""
        parse_args = []
        call_args = []
        c_var_decls = []

        for arg in fn_node.args.args:
            st = param_types.get(arg.arg, StalkType.FLOAT64)
            if st == StalkType.INT64:
                parse_format += "l"
                c_var_decls.append(f"    long {arg.arg};")
                parse_args.append(f"&{arg.arg}")
                call_args.append(f"(int64_t){arg.arg}")
            elif st == StalkType.BOOL:
                parse_format += "p"
                c_var_decls.append(f"    int {arg.arg};")
                parse_args.append(f"&{arg.arg}")
                call_args.append(f"(bool){arg.arg}")
            elif st == StalkType.FLOAT64:
                parse_format += "d"
                c_var_decls.append(f"    double {arg.arg};")
                parse_args.append(f"&{arg.arg}")
                call_args.append(f"{arg.arg}")

        parse_args_str = ", ".join(parse_args)
        if parse_args_str:
            parse_args_str = ", " + parse_args_str

        call_args_str = ", ".join(call_args)

        ret_wrapper = ""
        if return_type == StalkType.FLOAT64:
            ret_wrapper = "return PyFloat_FromDouble(res);"
        elif return_type == StalkType.INT64:
            ret_wrapper = "return PyLong_FromLongLong(res);"
        elif return_type == StalkType.BOOL:
            ret_wrapper = "return PyBool_FromLong(res ? 1 : 0);"
        else:
            ret_wrapper = "Py_RETURN_NONE;"

        wrapper_code = f"""
// ============================================================================
// CPYTHON C-API GLUE WRAPPER
// ============================================================================
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

{native_fn_code}

static PyObject* py_{fn_name}(PyObject* self, PyObject* args) {{
{chr(10).join(c_var_decls)}
    if (!PyArg_ParseTuple(args, "{parse_format}"{parse_args_str})) {{
        return NULL;
    }}
    {return_type.c_type()} res = topopyc_{fn_name}({call_args_str});
    {ret_wrapper}
}}

static PyMethodDef TopoMethods[] = {{
    {{"{fn_name}", py_{fn_name}, METH_VARARGS, "Topologically compiled kernel"}},
    {{NULL, NULL, 0, NULL}}
}};

static struct PyModuleDef topomodule = {{
    PyModuleDef_HEAD_INIT,
    "{module_name}",
    "Topologically compiled high-performance Python extension",
    -1,
    TopoMethods
}};

PyMODINIT_FUNC PyInit_{module_name}(void) {{
    return PyModule_Create(&topomodule);
}}
"""
        return wrapper_code
