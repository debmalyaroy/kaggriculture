

# ---------------------------------------------------------------------------
# LASTCALL: bind `agent` to the module's LAST callable, i.e. the entry point
# Kaggle's loader would run, before stacking our layers on a base whose
# `agent` name is stale (bases that chain via other names, e.g. ig_agent).
agent = [v for v in list(globals().values()) if callable(v)][-1]
