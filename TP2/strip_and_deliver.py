#!/usr/bin/env python3
"""Build delivery/src with comments and docstrings stripped, verified to compile."""
import ast, io, tokenize, py_compile, tempfile, shutil, os
from pathlib import Path

FILES = ["vlm_caption.py", "phase1_interrogate.py", "phase2_sampling.py",
         "phase3_render_score.py", "phase4_refine.py", "evaluation.py",
         "phase4_aggregate_seeds.py"]
OUT = Path("delivery/src")
OUT.mkdir(parents=True, exist_ok=True)

def remove_docstrings(src):
    tree = ast.parse(src)
    lines = src.split("\n")
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
               and isinstance(body[0].value.value, str):
                doc = body[0]
                needs_pass = (len(body) == 1) and not isinstance(node, ast.Module)
                spans.append((doc.lineno, doc.end_lineno, needs_pass, doc.col_offset))
    for start, end, needs_pass, indent in sorted(spans, key=lambda x: x[0], reverse=True):
        lines[start - 1:end] = ([" " * indent + "pass"] if needs_pass else [])
    return "\n".join(lines)

def remove_comments(src):
    cuts = {}
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            r, c = tok.start
            cuts[r] = min(c, cuts.get(r, c))
    out = []
    for i, line in enumerate(src.split("\n"), 1):
        if i in cuts:
            line = line[:cuts[i]].rstrip()
        out.append(line)
    return "\n".join(out)

def collapse_blanks(src):
    out, blank = [], 0
    for line in src.split("\n"):
        if line.strip() == "":
            blank += 1
            if blank <= 1:
                out.append("")
        else:
            blank = 0
            out.append(line.rstrip())
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + "\n"

def verify(src):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src); tmp = f.name
    try:
        py_compile.compile(tmp, doraise=True)
        return True
    except py_compile.PyCompileError:
        return False
    finally:
        os.unlink(tmp)

for fn in FILES:
    orig = Path(fn).read_text()
    try:
        stripped = collapse_blanks(remove_comments(remove_docstrings(orig)))
        mode = "comments+docstrings"
        if not verify(stripped):
            raise ValueError("compile failed after docstring removal")
    except Exception as e:
        stripped = collapse_blanks(remove_comments(orig))
        mode = f"comments only (docstring strip skipped: {e})"
        assert verify(stripped), f"{fn} fails to compile even comments-only"
    (OUT / fn).write_text(stripped)
    o, s = orig.count("\n"), stripped.count("\n")
    print(f"{fn:30s} {o:4d} -> {s:4d} lines  [{mode}]")
print("\nwrote", len(FILES), "files to", OUT)
