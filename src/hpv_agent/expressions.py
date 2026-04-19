from __future__ import annotations

import ast
from typing import Any

import numpy as np


_ALLOWED_FUNCS = {
    "abs": abs,
    "min": min,
    "max": max,
    "float": float,
    "int": int,
}


def _to_bool(value: Any) -> bool:
    if hasattr(value, "all"):
        return bool(value.all())
    return bool(value)


def _eval(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id, np.nan)
    if isinstance(node, ast.UnaryOp):
        val = _eval(node.operand, env)
        if isinstance(node.op, ast.Not):
            return ~val if hasattr(val, "__invert__") else (not val)
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return +val
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
    if isinstance(node, ast.BinOp):
        left = _eval(node.left, env)
        right = _eval(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left**right
        if isinstance(node.op, ast.BitAnd):
            return left & right
        if isinstance(node.op, ast.BitOr):
            return left | right
        raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
    if isinstance(node, ast.BoolOp):
        values = [_eval(v, env) for v in node.values]
        if not values:
            return False
        out = values[0]
        for v in values[1:]:
            if isinstance(node.op, ast.And):
                out = out & v if hasattr(out, "__and__") else (_to_bool(out) and _to_bool(v))
            elif isinstance(node.op, ast.Or):
                out = out | v if hasattr(out, "__or__") else (_to_bool(out) or _to_bool(v))
            else:
                raise ValueError(f"Unsupported bool operator: {type(node.op).__name__}")
        return out
    if isinstance(node, ast.Compare):
        left = _eval(node.left, env)
        result = True
        for op, comp in zip(node.ops, node.comparators):
            right = _eval(comp, env)
            if isinstance(op, ast.Gt):
                cur = left > right
            elif isinstance(op, ast.GtE):
                cur = left >= right
            elif isinstance(op, ast.Lt):
                cur = left < right
            elif isinstance(op, ast.LtE):
                cur = left <= right
            elif isinstance(op, ast.Eq):
                cur = left == right
            elif isinstance(op, ast.NotEq):
                cur = left != right
            else:
                raise ValueError(f"Unsupported compare operator: {type(op).__name__}")
            result = result & cur if hasattr(result, "__and__") else (_to_bool(result) and _to_bool(cur))
            left = right
        return result
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only plain function names are allowed")
        fn_name = node.func.id
        if fn_name not in _ALLOWED_FUNCS:
            raise ValueError(f"Function not allowed: {fn_name}")
        fn = _ALLOWED_FUNCS[fn_name]
        args = [_eval(a, env) for a in node.args]
        return fn(*args)
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def safe_eval(expr: str, env: dict[str, Any]) -> Any:
    tree = ast.parse(expr, mode="eval")
    return _eval(tree, env)
