"""Extract the agent source embedded in a public Kaggriculture notebook.

Public notebooks on this competition ship their `main.py` as a base85/base64 +
zlib payload inside a list of string literals, so the readable notebook stays
small. This tool pulls those literals out with a regex, decodes them as **data
only** (no exec, no import) and writes the result next to the notebook.

It also summarises any decoded JSON payload -- several top agents embed a
recorded 720-step action tape rather than a policy, and knowing that changes
what we are competing against.

Usage:
    python -m kaggriculture.data.notebook_extract data/kernels/_src/*.py
    python -m kaggriculture.data.notebook_extract <file> --list          # payloads only
    python -m kaggriculture.data.notebook_extract <file> --json NAME     # summarise a blob
"""
from __future__ import annotations

import argparse
import ast
import base64
import glob
import json
import os
import re
import zlib

DECODERS = (("b85", base64.b85decode), ("b64", base64.b64decode),
            ("a85", base64.a85decode))


def _decode(blob):
    """Try every encoding/compression combination; return (text, how)."""
    for name, dec in DECODERS:
        try:
            raw = dec(blob)
        except Exception:                                    # noqa: BLE001
            continue
        for how, fn in (("zlib", zlib.decompress), ("raw", lambda b: b)):
            try:
                out = fn(raw)
            except Exception:                                # noqa: BLE001
                continue
            try:
                return out.decode("utf-8"), f"{name}+{how}"
            except UnicodeDecodeError:
                continue
    return None, None


def _bare_groups(src):
    """Parenthesised runs of string literals, e.g. inside b85decode((...)).

    Several notebooks pass the payload straight into a call rather than binding
    it to a name, so the assignment scan below never sees it. The name reported
    is the nearest preceding assignment target, which is what a reader would
    call it.
    """
    for m in re.finditer(r"\(\s*\n(\s*['\"][^\n]*\n)+\s*\)", src):
        chunk = m.group(0)
        parts = re.findall(r"'((?:[^'\\]|\\.)*)'", chunk) or \
            re.findall(r'"((?:[^"\\]|\\.)*)"', chunk)
        joined = "".join(parts)
        if len(joined) < 200:
            continue
        before = src[max(0, m.start() - 400):m.start()]
        names = re.findall(r"(\w+)\s*=", before)
        yield (names[-1] if names else "payload"), joined


def _single_line(src):
    """`NAME = '<one very long encoded string>'` on a single line.

    The two scans below both require a *sequence* of literals -- a list, or a
    parenthesised run across lines. A notebook that assigns the whole payload
    as one string matched neither and reported "no encoded payload found",
    which is how three notebooks carrying real agents (including a 2600-Elo
    architecture and a 2.77 MB nested version chain) went undecoded.

    Keyed on the alphabet rather than the length alone, so ordinary long
    strings -- prose, SQL, HTML -- do not trip it.
    """
    # base85 (RFC 1924 / b85encode) plus the base64/ascii85 alphabets.
    alphabet = r"0-9A-Za-z!#$%&()*+\-;<=>?@^_`{|}~/=+"
    pattern = r"(\w+)\s*=\s*(['\"])([" + alphabet + r"]{800,})\2"
    for m in re.finditer(pattern, src):
        yield m.group(1), m.group(3)


def payloads(src):
    """Yield (name, joined_literal) for every list-of-strings assignment."""
    for pair in _bare_groups(src):
        yield pair
    for pair in _single_line(src):
        yield pair
    for m in re.finditer(r"(\w+)\s*=\s*(\[|\(\s*\n)", src):
        name = m.group(1)
        opener = m.group(2).strip()[0]
        closer = "]" if opener == "[" else ")"
        depth, start = 0, m.end(2) - 1
        end = None
        for i in range(start, len(src)):
            ch = src[i]
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            continue
        chunk = src[start:end]
        if chunk.count("'") + chunk.count('"') < 4:
            continue
        try:
            value = ast.literal_eval(chunk)
        except Exception:                                    # noqa: BLE001
            parts = re.findall(r"'([^']*)'", chunk) or re.findall(r'"([^"]*)"', chunk)
            if not parts:
                continue
            value = parts
        if isinstance(value, (list, tuple)) and value and all(
                isinstance(v, str) for v in value):
            joined = "".join(value)
            if len(joined) > 200:
                yield name, joined
        elif isinstance(value, str) and len(value) > 200:
            yield name, value


def summarise_json(text, limit=6):
    try:
        obj = json.loads(text)
    except Exception:                                        # noqa: BLE001
        return None
    if isinstance(obj, list):
        head = ", ".join(json.dumps(o)[:120] for o in obj[:2])
        return (f"JSON list, {len(obj)} entries\n    [0..1] {head}\n"
                f"    [-1]   {json.dumps(obj[-1])[:160]}")
    if isinstance(obj, dict):
        keys = list(obj)[:limit]
        first = json.dumps(obj[keys[0]])[:160] if keys else ""
        return (f"JSON dict, {len(obj)} keys: {keys}\n"
                f"    [{keys[0] if keys else ''}] {first}")
    return f"JSON {type(obj).__name__}"


def process(path, out_dir, want_json=None, list_only=False):
    src = open(path, encoding="utf-8", errors="replace").read()
    base = os.path.splitext(os.path.basename(path))[0]
    found = 0
    for name, blob in payloads(src):
        text, how = _decode(blob)
        if text is None:
            continue
        found += 1
        kind = "python" if "def " in text[:4000] or "import " in text[:400] else "data"
        summary = summarise_json(text)
        print(f"{base}  {name}: {len(blob):,} chars -> {len(text):,} ({how}, {kind})")
        if summary:
            print("    " + summary.replace("\n", "\n    "))
        if list_only or (want_json and name != want_json):
            continue
        ext = ".py" if kind == "python" else ".json"
        dest = os.path.join(out_dir, f"{base}__{name}{ext}")
        os.makedirs(out_dir, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"    -> {dest}")
    if not found:
        print(f"{base}: no encoded payload found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="data/kernels/_agents")
    ap.add_argument("--list", action="store_true", dest="list_only")
    ap.add_argument("--json", default=None, help="only extract this payload name")
    args = ap.parse_args()
    files = []
    for p in args.paths:
        files += sorted(glob.glob(p))
    for f in files:
        process(f, args.out, want_json=args.json, list_only=args.list_only)


if __name__ == "__main__":
    main()
