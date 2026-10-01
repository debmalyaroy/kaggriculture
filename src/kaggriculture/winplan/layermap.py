"""Static layer map of a layered chassis agent.

    python -m kaggriculture.winplan.layermap <agent.py> [--diff <other.py>] [--out FILE.md]

Walks the module's top-level statements in order and finds every layer: an
assignment that captures the current entry point (`_X_PARENT = agent`,
`_X_HOST = [..callable..][-1]`, `_X_PARENT = cha20_entry_agent`, ...), the
wrapper function defined after it, the header comment above it, its knobs
(UPPERCASE literals sharing the prefix) and its telemetry dict. With --diff,
layers are matched by prefix and by the source hash of their functions, so
"same layer", "same name but modified" and "only in A/B" are separated.
Also records which function Kaggle would run (the last callable) and the
decoded exec() payloads it finds (they are NOT expanded: flagged only).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import re

CAPTURE = re.compile(r"^(_[A-Z][A-Z0-9_]*?)_(PARENT|HOST|BASE|INNER|PREV|ORIG[A-Z_]*|ENTRY)$")


def _src(lines, node):
    return "\n".join(lines[node.lineno - 1: node.end_lineno])


def _header(lines, lineno):
    out = []
    i = lineno - 2
    while i >= 0 and (lines[i].startswith("#") or not lines[i].strip()):
        if lines[i].startswith("#"):
            out.append(lines[i].lstrip("# ").rstrip())
        elif out:
            break
        i -= 1
    txt = " ".join(x for x in reversed(out) if x and not set(x) <= set("=-"))
    return txt[:300]


def layers(path):
    text = open(path, encoding="utf-8").read()
    lines = text.splitlines()
    tree = ast.parse(text)
    body = tree.body
    found, knobs, funcs, execs = [], {}, {}, 0
    for i, node in enumerate(body):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and getattr(node.value.func, "id", "") == "exec":
            execs += 1
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            funcs.setdefault(node.name, []).append(hashlib.sha1(_src(lines, node).encode()).hexdigest()[:10])
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.isupper() or (name.startswith("_") and name[1:2].isupper()):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float, bool, str)):
                    knobs[name] = node.value.value
            m = CAPTURE.match(name)
            if m:
                prefix = m.group(1)
                parent = _src(lines, node).split("=", 1)[1].strip()[:80]
                wrapper, wsrc = None, ""
                for nxt in body[i + 1:i + 60]:
                    if isinstance(nxt, ast.FunctionDef) and prefix.lower().strip("_") in _src(lines, nxt).lower() \
                            and ("PARENT" in _src(lines, nxt) or "HOST" in _src(lines, nxt) or m.group(2) in _src(lines, nxt)):
                        if f"{name}(" in _src(lines, nxt):
                            wrapper, wsrc = nxt.name, _src(lines, nxt)
                            break
                found.append({"prefix": prefix, "capture": name, "line": node.lineno, "parent": parent,
                              "wrapper": wrapper, "header": _header(lines, node.lineno),
                              "wrap_hash": hashlib.sha1(wsrc.encode()).hexdigest()[:10] if wsrc else None})
    for L in found:
        p = L["prefix"]
        L["knobs"] = {k: v for k, v in knobs.items() if k.startswith(p + "_") and not k.endswith(("_PARENT", "_HOST", "_REPORT", "_STATE", "_STATS"))}
        L["report"] = next((k for k in knobs if k == p + "_REPORT"), None) or (p + "_REPORT" if f"{p}_REPORT" in text else None)
        L["helpers"] = sorted({n for n in funcs if n.lower().startswith(p.lower().strip("_") + "_") or n.lower().startswith("_" + p.lower().strip("_") + "_")})
        L["helper_hash"] = hashlib.sha1("".join(h for n in L["helpers"] for h in funcs[n]).encode()).hexdigest()[:10]
    ns = {}
    return {"path": path, "lines": len(lines), "layers": found, "exec_payloads": execs}


def render(m, title):
    out = [f"## {title}: {m['path']}", f"{m['lines']} lines, {len(m['layers'])} layer captures, {m['exec_payloads']} exec() payloads", "",
           "| # | line | layer | wraps | wrapper | knobs | header |", "|---|---|---|---|---|---|---|"]
    for i, L in enumerate(m["layers"], 1):
        kn = ", ".join(f"{k.replace(L['prefix'] + '_', '')}={v!r}" for k, v in list(L["knobs"].items())[:6])
        out.append(f"| {i} | {L['line']} | {L['prefix']} | `{L['parent'][:40]}` | {L['wrapper'] or '-'} | {kn[:90]} | {L['header'][:140]} |")
    return "\n".join(out)


def diff(a, b):
    ia = {L["prefix"]: L for L in a["layers"]}
    ib = {L["prefix"]: L for L in b["layers"]}
    rows = ["", "## diff (A -> B)", "| layer | status | detail |", "|---|---|---|"]
    for p in list(ia) + [p for p in ib if p not in ia]:
        if p in ia and p not in ib:
            rows.append(f"| {p} | only in A | {ia[p]['header'][:120]} |")
        elif p in ib and p not in ia:
            rows.append(f"| {p} | **only in B** | {ib[p]['header'][:120]} |")
        else:
            x, y = ia[p], ib[p]
            same = x["helper_hash"] == y["helper_hash"] and x["wrap_hash"] == y["wrap_hash"]
            kd = {k: (x["knobs"].get(k), y["knobs"].get(k)) for k in set(x["knobs"]) | set(y["knobs"]) if x["knobs"].get(k) != y["knobs"].get(k)}
            rows.append(f"| {p} | {'same' if same and not kd else 'MODIFIED'} | {('knobs ' + str(kd)[:140]) if kd else ''}{'' if same else ' code differs'} |")
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("agent")
    ap.add_argument("--diff")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    A = layers(a.agent)
    txt = render(A, "A")
    if a.diff:
        B = layers(a.diff)
        txt += "\n\n" + render(B, "B") + "\n" + diff(A, B)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
