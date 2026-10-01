"""Read/write the PARAMS block of a Kaggriculture agent source file.

The agent keeps its whole configuration in one dict delimited by sentinels:

    # --- PARAMS BEGIN ... ---
    PARAMS = { ... }
    # --- PARAMS END ---

`render` swaps in a new dict while leaving the rest of the file untouched, so a
tuned agent stays a single self-contained file that can be submitted as-is.
"""
import ast
import pprint

BEGIN = "# --- PARAMS BEGIN"
END = "# --- PARAMS END ---"
BIAS_BEGIN = "# --- ASSET_BIAS BEGIN"
BIAS_END = "# --- ASSET_BIAS END ---"


def split(src):
    i = src.index(BEGIN)
    j = src.index(END, i) + len(END)
    return src[:i], src[i:j], src[j:]


def load(path):
    src = open(path, encoding="utf-8").read()
    _head, block, _tail = split(src)
    body = block[block.index("PARAMS = ") + len("PARAMS = "):block.index(END)]
    return ast.literal_eval(body.strip())


def _complete(base_path, params):
    """Fill in knobs the caller's dict is missing from the base file's PARAMS.

    Without this, generating an agent from an older parameter dict silently
    drops any knob added since, and the code's `P.get(name, default)` fallback
    takes over. That is how an A/B test of one change measured a different
    change: a control built from a pre-`pen_reserve_max` dict picked up the code
    default of 8 and reserved eight tiles nobody asked it to.
    """
    base = load(base_path)
    merged = dict(base)
    merged.update(params)
    return merged


def render(base_path, params, header=None, module_doc=None, asset_bias=None):
    """Return the full source of `base_path` with PARAMS replaced.

    `module_doc`, if given, also replaces the module docstring so a generated
    agent does not claim to be the agent it was generated from.
    """
    src = open(base_path, encoding="utf-8").read()
    params = _complete(base_path, params)
    if asset_bias is not None and BIAS_BEGIN in src:
        i = src.index(BIAS_BEGIN)
        j = src.index(BIAS_END, i) + len(BIAS_END)
        body = pprint.pformat(asset_bias, width=96, sort_dicts=True)
        src = (src[:i] + BIAS_BEGIN + " (trainers rewrite this block) ---\n"
               + f"ASSET_BIAS = {body}\n" + BIAS_END + src[j:])
    if module_doc:
        start = src.find('"""')
        end = src.find('"""', start + 3) if start >= 0 else -1
        if start < 0 or end < 0:
            raise ValueError(
                f"{base_path} has no module docstring to replace -- refusing to "
                f"rewrite it (is the file truncated?)")
        src = '"""' + module_doc + '"""' + src[end + 3:]
    head, _block, tail = split(src)
    text = pprint.pformat(params, width=96, sort_dicts=False)
    block = f"{BEGIN} (src/kaggriculture/train/tune.py rewrites this block verbatim) ---\n"
    if header:
        block += "".join(f"# {line}\n" for line in header.splitlines())
    block += f"PARAMS = {text}\n{END}"
    return head + block + tail


def write(base_path, out_path, params, header=None, module_doc=None, asset_bias=None):
    # UTF-8 explicitly. Windows defaults to cp1252, and a single em-dash in a
    # generated module docstring is enough to write a file Python then refuses
    # to parse -- which is how an RL agent that trained fine scored $3,000.
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(render(base_path, params, header, module_doc, asset_bias))
    return out_path
