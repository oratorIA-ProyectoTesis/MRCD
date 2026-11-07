"""Static import graph of first-party modules (ast walk; nothing is imported or executed)."""

from __future__ import annotations

import ast
from pathlib import Path


def _module_file(root: Path, dotted: str) -> Path | None:
    base = root.joinpath(*dotted.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def _imports(path: Path, dotted: str) -> tuple[set[str], list[str]]:
    """Returns (absolute dotted names imported, dotted names of attribute calls like ``torch.load``)."""
    tree = ast.parse(path.read_text("utf8"), filename=str(path))
    is_pkg = path.name == "__init__.py"
    pkg = dotted if is_pkg else dotted.rpartition(".")[0]
    names: set[str] = set()
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = pkg.split(".")
                parts = parts[: len(parts) - (node.level - 1)]
                mod = ".".join(parts + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            names.add(mod)
            names.update(f"{mod}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                calls.append(f"{node.func.value.id}.{node.func.attr}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "__import__":
                names.add("__import__")
    return names, calls


def transitive_imports(root: Path, start: str, first_party: str = "mrcd"):
    """Walks ``start`` and every first-party module it (transitively) imports, parent packages included.

    Returns (set of visited first-party modules, set of every imported dotted name, set of attribute calls).
    """
    seen: set[str] = set()
    external: set[str] = set()
    calls: set[str] = set()
    todo = [start]
    while todo:
        mod = todo.pop()
        if mod in seen:
            continue
        path = _module_file(root, mod)
        if path is None:
            continue
        seen.add(mod)
        parts = mod.split(".")
        todo.extend(".".join(parts[:i]) for i in range(1, len(parts)))
        names, cs = _imports(path, mod)
        calls.update(cs)
        for n in names:
            external.add(n)
            if n.split(".")[0] == first_party:
                todo.append(n)
    return seen, external, calls
