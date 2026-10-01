"""Canonical repo-root resolver (depth-independent).

Every module that needs the repo root imports ROOT from here instead of
counting os.path.dirname(__file__) levels, so files can live at any depth.
"""
import os

def _find_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(12):
        if os.path.exists(os.path.join(d, 'pyproject.toml')) or \
           os.path.exists(os.path.join(d, 'CLAUDE.md')):
            return d
        p = os.path.dirname(d)
        if p == d:
            break
        d = p
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = _find_root()
