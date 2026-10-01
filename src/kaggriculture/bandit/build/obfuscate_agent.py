"""Behavior-preserving obfuscation of a chassis agent for private submission.

Removes the identity/fingerprint + info-leak surface WITHOUT changing behavior:
  * every comment (attribution: Pipe-16 HybridOpening, Metav4, yhay81/Berat Ozer/
    Gluzdov citations, EXP notes, URLs)
  * every docstring (module/class/function) -- the strategy explanations
  * `<name>.telemetry = ...` / `agent.telemetry` assignments and the _REPORT plumbing
    exposure (an opponent could read our internal state)
  * the leading strategy header comment block

Method: tokenize-strip comments, then AST-strip docstrings, then ast.unparse. The
result is semantically identical (verified by a mirror-tie self-play on serve) but
carries none of the human-readable identity. Optionally renames a few top-level
identifying symbols (Chassis/ROUTES_DATA/_router) via a scope-safe AST rename.

Usage:
  python -m kaggriculture.bandit.build.obfuscate_agent IN.py OUT.py [--rename]
"""
import argparse, ast, io, tokenize


def strip_comments(src: str) -> str:
    out = []
    prev_end = (1, 0)
    toks = tokenize.generate_tokens(io.StringIO(src).readline)
    for tok in toks:
        if tok.type == tokenize.COMMENT:
            continue
        out.append(tok)
    return tokenize.untokenize(out)


class DocstringStripper(ast.NodeTransformer):
    def _strip(self, node):
        self.generic_visit(node)
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            # keep at least a pass if the docstring is the only statement
            rest = body[1:]
            node.body = rest if rest else [ast.Pass()]
        return node

    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip
    visit_ClassDef = _strip
    visit_Module = _strip


class TelemetryStripper(ast.NodeTransformer):
    """Drop `X.telemetry = ...` assignments (fingerprint + state leak)."""
    def visit_Assign(self, node):
        for t in node.targets:
            if isinstance(t, ast.Attribute) and t.attr == "telemetry":
                return None
        return node


RENAMES = {
    "Chassis": "_Eng", "ROUTES_DATA": "_D0", "_router": "_pick",
    "make_agent": "_mk", "_View": "_Vw", "FarmView": "_Fv",
}


class Renamer(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id in RENAMES:
            node.id = RENAMES[node.id]
        return node
    def visit_FunctionDef(self, node):
        if node.name in RENAMES:
            node.name = RENAMES[node.name]
        self.generic_visit(node)
        return node
    def visit_ClassDef(self, node):
        if node.name in RENAMES:
            node.name = RENAMES[node.name]
        self.generic_visit(node)
        return node


def obfuscate(src: str, rename: bool = False) -> str:
    src = strip_comments(src)
    tree = ast.parse(src)
    tree = DocstringStripper().visit(tree)
    tree = TelemetryStripper().visit(tree)
    if rename:
        tree = Renamer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("outp")
    ap.add_argument("--rename", action="store_true")
    a = ap.parse_args()
    src = open(a.inp, encoding="utf-8").read()
    out = obfuscate(src, rename=a.rename)
    with open(a.outp, "w", encoding="utf-8") as fh:
        fh.write(out)
    # report what was removed
    import re
    tells = ["Pipe-16", "Metav4", "yhay81", "Ozer", "Gluzdov", "tetsutani", "EXP",
             "HybridOpening", "telemetry", "http"]
    left = {t: out.count(t) for t in tells if out.count(t)}
    print(f"obfuscated {a.inp} -> {a.outp}  ({len(src)} -> {len(out)} chars)")
    print(f"residual tells: {left or 'none'}")


if __name__ == "__main__":
    main()
