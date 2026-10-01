"""Model factory: a declarative config in, a submittable agent out.

Every agent this project ships should be reproducible from a file you can read,
diff and put in version control -- not from remembering which flags were passed
to which trainer six hours ago. A config is that file.

    python -m kaggriculture.agentbuild.build_agent configs/v5_ensemble.json
    python -m kaggriculture.agentbuild.build_agent configs/*.json --smoke
    python -m kaggriculture.agentbuild.build_agent --new v6 --base agents/agent_v4_*.py \\
                                --set ensemble_k=12 --set assign_mode=optimal

Config schema (JSON; every field optional except name)
------------------------------------------------------
    {
      "name":        "v5_ensemble",
      "description": "v4 plus a 12-member committee vote",
      "base":        "agents/agent_v4_optimal_20260805_014340.py",
      "params":      {"ensemble_k": 12, "ensemble_consensus": 0.55},
      "asset_bias":  {"from": "agents/agent_vridge_20260804.py"},
      "tags":        ["ensemble", "candidate"],
      "out":         "agents/{name}_{stamp}.py"
    }

`base` supplies the starting PARAMS and may be a glob (newest match wins), so a
config does not go stale every time a trainer produces a new generation.
`params` overrides them. `asset_bias` either inlines a learned bias table or
copies one out of another agent.

Unknown parameter names are a hard error, not a warning. A typo in a knob name
is silent otherwise -- the agent builds, runs, and quietly ignores the thing you
were trying to change, and you spend an afternoon testing a null edit.
"""
from kaggriculture.paths import ROOT
import argparse
import ast
import datetime as dt
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.pipeline.params as paramio  # noqa: E402
import kaggriculture.data.registry as registry  # noqa: E402

SRC = os.path.join(ROOT, "agents", "v1_heuristic.py")
CONFIGS = os.path.join(ROOT, "configs")


class ConfigError(ValueError):
    pass


def _resolve(pattern):
    """A path or a glob; newest match wins. Keeps configs from going stale."""
    if not pattern:
        return None
    p = pattern if os.path.isabs(pattern) else os.path.join(ROOT, pattern)
    if os.path.exists(p):
        return p
    hits = sorted(glob.glob(p))
    if not hits:
        raise ConfigError(f"base not found: {pattern}")
    return hits[-1]


def load_asset_bias(path):
    """Read the ASSET_BIAS block out of an agent without importing it.

    Importing would execute the module; parsing the literal will not, which
    matters when the source is a downloaded opponent rather than our own code.
    """
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if paramio.BIAS_BEGIN not in src:
        return None
    i = src.index(paramio.BIAS_BEGIN)
    j = src.index("ASSET_BIAS", i)
    k = src.index("=", j)
    end = src.index("# --- ASSET_BIAS END", k)
    try:
        return ast.literal_eval(src[k + 1:end].strip())
    except (SyntaxError, ValueError):
        return None


def known_params():
    """The knob names the agent source actually declares."""
    return set(paramio.load(SRC))


def validate(cfg):
    if not cfg.get("name"):
        raise ConfigError("config needs a 'name'")
    bad = sorted(set(cfg.get("params", {})) - known_params())
    if bad:
        near = {}
        for b in bad:
            cands = [k for k in known_params() if k.startswith(b[:4])]
            if cands:
                near[b] = cands[:3]
        hint = ("  did you mean: " + json.dumps(near)) if near else ""
        raise ConfigError(f"unknown parameter(s): {', '.join(bad)}\n{hint}")
    return True


def build(cfg, smoke=False, quiet=False):
    """Materialise one config into an agent file. Returns the repo-relative path."""
    validate(cfg)
    base = _resolve(cfg.get("base")) or os.path.join(ROOT, "agents", "v2_tuned.py")
    P = dict(paramio.load(base))
    P.update(cfg.get("params", {}))

    bias = None
    ab = cfg.get("asset_bias")
    if isinstance(ab, dict) and "from" in ab:
        bias = load_asset_bias(_resolve(ab["from"]))
    elif isinstance(ab, dict):
        bias = ab

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_rel = (cfg.get("out") or "agents/{name}_{stamp}.py").format(
        name=cfg["name"], stamp=stamp)
    out = os.path.join(ROOT, out_rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    diff = {k: (paramio.load(base).get(k), v)
            for k, v in cfg.get("params", {}).items()}
    doc = [cfg.get("description") or cfg["name"], ""]
    doc.append(f"Built by src/kaggriculture/agentbuild/build_agent.py from config '{cfg['name']}'.")
    doc.append(f"Base: {os.path.relpath(base, ROOT)}")
    if diff:
        doc.append("Overrides:")
        for k, (was, now) in sorted(diff.items()):
            doc.append(f"  {k}: {was} -> {now}")
    if bias:
        doc.append(f"ASSET_BIAS: {len(bias)} learned entries")

    paramio.write(SRC, out, P, header=f"config: {cfg['name']}",
                  module_doc="\n".join(doc) + "\n", asset_bias=bias)

    registry.register_model(
        os.path.basename(out), path=out_rel, config=cfg["name"],
        base=os.path.relpath(base, ROOT), overrides=cfg.get("params", {}),
        tags=cfg.get("tags", []), description=cfg.get("description", ""),
        built_by="build_agent")

    if not quiet:
        print(f"built {out_rel}  ({os.path.getsize(out):,} bytes)")
        for k, (was, now) in sorted(diff.items()):
            print(f"   {k}: {was} -> {now}")

    if smoke:
        ok, why = smoke_test(out)
        registry.register_model(os.path.basename(out),
                                tests="smoke ok" if ok else f"smoke FAILED: {why}")
        if not quiet:
            print("   smoke:", "ok" if ok else f"FAILED -- {why}")
        if not ok:
            raise ConfigError(f"{out_rel} failed its smoke test: {why}")
    return out_rel


def smoke_test(path, steps=48):
    """Import it, play a few turns, check the action shape. Seconds, not minutes.

    Catches the failures that would otherwise surface as a wasted submission:
    an import error, a missing agent(), or a malformed action dict.
    """
    try:
        import importlib.util
        sys.path.insert(0, os.path.join(ROOT, "agents"))
        import kaggriculture.engine._vendor as _vendor  # noqa: F401
        from kaggle_environments import make
        spec = importlib.util.spec_from_file_location("cand", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not callable(getattr(mod, "agent", None)):
            return False, "no agent() callable"
        env = make("kaggriculture",
                   configuration={"episodeSteps": steps, "seed": 3,
                                  "actTimeout": 60, "runTimeout": 100000})
        seen = []

        def wrapped(obs, cfg):
            a = mod.agent(obs, cfg)
            seen.append(a)
            return a

        env.run([wrapped, "random"])
        if not seen:
            return False, "agent was never called"
        for a in seen[:8]:
            if not isinstance(a, dict) or "farmer" not in a or "hands" not in a:
                return False, f"malformed action: {str(a)[:80]}"
        return True, ""
    except Exception as exc:                                       # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:120]}"


def new_config(name, base, sets, description="", tags=None):
    params = {}
    for s in sets or []:
        if "=" not in s:
            raise ConfigError(f"--set wants key=value, got {s!r}")
        k, v = s.split("=", 1)
        try:
            params[k.strip()] = json.loads(v)
        except ValueError:
            params[k.strip()] = v.strip()      # bare strings like optimal
    cfg = {"name": name, "description": description, "base": base,
           "params": params, "tags": tags or []}
    validate(cfg)
    os.makedirs(CONFIGS, exist_ok=True)
    path = os.path.join(CONFIGS, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"wrote {os.path.relpath(path, ROOT)}")
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("configs", nargs="*", help="config files or globs")
    ap.add_argument("--smoke", action="store_true",
                    help="import and play a few turns before accepting it")
    ap.add_argument("--list", action="store_true", help="show available configs")
    ap.add_argument("--params", action="store_true",
                    help="print every knob the agent source declares")
    ap.add_argument("--new", metavar="NAME", help="write a new config")
    ap.add_argument("--base", default="agents/agent_v*.py")
    ap.add_argument("--set", action="append", dest="sets", metavar="KEY=VALUE")
    ap.add_argument("--description", default="")
    args = ap.parse_args()

    if args.params:
        for k, v in sorted(paramio.load(SRC).items()):
            print(f"  {k:<28} {v!r}")
        return 0

    if args.list or (not args.configs and not args.new):
        os.makedirs(CONFIGS, exist_ok=True)
        found = sorted(glob.glob(os.path.join(CONFIGS, "*.json")))
        if not found:
            print("no configs yet. Create one:")
            print("  python -m kaggriculture.agentbuild.build_agent --new v5 --set ensemble_k=12")
            return 0
        print(f"{len(found)} config(s) in configs/\n")
        for p in found:
            try:
                with open(p, encoding="utf-8") as f:
                    c = json.load(f)
                print(f"  {os.path.basename(p):<28} {c.get('description', '')[:60]}")
                if c.get("params"):
                    print(f"    {json.dumps(c['params'])[:100]}")
            except ValueError as exc:
                print(f"  {os.path.basename(p):<28} INVALID JSON: {exc}")
        return 0

    if args.new:
        new_config(args.new, args.base, args.sets, args.description)
        return 0

    paths = []
    for pat in args.configs:
        p = pat if os.path.isabs(pat) else os.path.join(ROOT, pat)
        hits = sorted(glob.glob(p)) or ([p] if os.path.exists(p) else [])
        if not hits:
            print(f"no config matched {pat}", file=sys.stderr)
            return 2
        paths.extend(hits)

    rc = 0
    for p in paths:
        with open(p, encoding="utf-8") as f:
            cfg = json.load(f)
        try:
            build(cfg, smoke=args.smoke)
        except ConfigError as exc:
            print(f"{os.path.basename(p)}: {exc}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
