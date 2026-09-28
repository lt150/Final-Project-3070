"""Report plumbing shared by the four stage drivers.

Task 9 / Task 10 conventions: run command and UTC timestamp, environment pin
header (interpreter path + versions), gates with PASS/FAIL evidence, notes,
deliverables, STOP line. ASCII-safe output throughout.
"""

from __future__ import annotations

import ast
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .fixtures import IMPORT_MAP, OUT_DIR

PACKAGE_DIR = Path(__file__).resolve().parent


def md_table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    head = "| " + " | ".join(str(h) for h in header) + " |"
    sep = "|" + "|".join(["---"] * len(header)) + "|"
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def env_header(title: str, command: str) -> list[str]:
    import numpy as np
    import pandas as pd
    import pyarrow
    import scipy

    return [
        f"# {title}",
        "",
        "Environment pin: everything evidentiary "
        "runs under the repo venv; the system interpreter is never used.",
        "",
        f"- interpreter: `{sys.executable}`",
        f"- python {sys.version.split()[0]}, numpy {np.__version__}, "
        f"pandas {pd.__version__}, scipy {scipy.__version__}, "
        f"pyarrow {pyarrow.__version__}",
        "",
    ]


def write_report(name: str, lines: Sequence[str]) -> str:
    """Write the report, then enforce the ASCII-safe rule.

    The file lands first so a violation never costs the evidence; the check
    then halts loudly.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines)
    (OUT_DIR / name).write_text(report, encoding="utf-8")
    bad = sorted({c for c in report if ord(c) > 127})
    if bad:
        raise RuntimeError(
            f"{name}: non-ASCII characters in report output "
            f"{[(c, hex(ord(c))) for c in bad]} (Requires ASCII-safe "
            "output) -- STOP"
        )
    return report


# --------------------------------------------------------------------------- #
# Static import audit (Gate A1 evidence)
# --------------------------------------------------------------------------- #
def package_imports() -> tuple[list[tuple[str, str, str, str]], list[str]]:
    """Every import statement in ``src/task12``, parsed from source.

    Returns (rows, upstream_symbols) where each row is
    (file, statement, kind, target) and ``kind`` is one of
    ``upstream`` (a relative import reaching outside the package),
    ``intra`` (a sibling module of ``src.task12``), or
    ``external`` (stdlib / third party).
    """
    rows: list[tuple[str, str, str, str]] = []
    upstream: list[str] = []
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    rows.append((path.name, f"import {alias.name}", "external", alias.name))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = ", ".join(a.name for a in node.names)
                dots = "." * node.level
                stmt = f"from {dots}{module} import {names}"
                if node.level >= 2:
                    target = "src." + module
                    rows.append((path.name, stmt, "upstream", target))
                    upstream += [f"{target}.{a.name}" for a in node.names]
                elif node.level == 1:
                    rows.append((path.name, stmt, "intra", f"src.task12.{module}"))
                else:
                    rows.append((path.name, stmt, "external", module))
    return rows, sorted(set(upstream))


def registered_upstream() -> set[str]:
    return {f"{module}.{name}" for name, module, _ in IMPORT_MAP}


# --------------------------------------------------------------------------- #
# Amendment: the validation fixtures are scoped to Stage A only
# --------------------------------------------------------------------------- #
# Ratified: `fixtures.linear_map` (and the coefficient accessor and
# toy it sits beside) are validation fixtures for the exactness check.
# They must not appear anywhere on the Stage B / Stage C production attribution
# path. `fixtures.py` defines them and `stage_a.py` uses them; every other module
# must be free of them.
STAGE_A_ONLY_FIXTURES = ("linear_map", "linear_coefficients", "toy_function", "fit_coeffs")
PRODUCTION_PATH_MODULES = (
    "engine.py", "probes.py", "metrics.py", "exhibits.py", "stage_b.py", "stage_c.py",
)


def code_referenced_names(path: Path) -> set[str]:
    """Every name a module references IN CODE: bindings, attributes, imports.

    Deliberately structural. String literals and docstrings are ``ast.Constant``
    nodes and contribute nothing, so a module that merely NAMES a symbol in its
    own report prose does not count as referencing it.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.name)
                if alias.asname:
                    names.add(alias.asname)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
                if alias.asname:
                    names.add(alias.asname)
    return names


def stage_a_only_scope() -> list[tuple[str, str, str]]:
    """(module, fixture, verdict) for every production-path module x fixture.
    """
    rows: list[tuple[str, str, str]] = []
    for module in PRODUCTION_PATH_MODULES:
        referenced = code_referenced_names(PACKAGE_DIR / module)
        for name in STAGE_A_ONLY_FIXTURES:
            rows.append(
                (module, name, "absent" if name not in referenced else "**PRESENT**")
            )
    return rows
