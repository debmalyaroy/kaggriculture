"""Per-seat selfplay args for a built release stage (the same flags main.py passes agent-stdio)."""
import os, re, sys
RL = os.environ.get("KRL_RL") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def stage_args(stage):
    stage = os.path.abspath(stage)
    src = open(os.path.join(stage, "main.py"), encoding="utf-8").read()
    c = {}
    for k in ["PROFILE", "CLONE_PROFILE", "CLONE_STRICT", "POLICY", "SHIELD", "GROUP", "SHELL", "DISGUISE", "RSHELL", "PREEMPT",
              "GROUP_KNOBS", "GROUP_KNOBS_FOR", "KNOB_OVER", "ENDG", "GT", "DISPATCH", "CHAIN_OFF"]:
        m = re.search(rf"^{k} = (.+?)(\s+#.*)?$", src, re.M)
        c[k] = eval(m.group(1)) if m else None
    j = lambda f: os.path.join(stage, f).replace("\\", "/")
    a = []
    if os.path.exists(j("profiles.json")): a += ["--profiles", j("profiles.json")]
    if c["PROFILE"] is not None: a += ["--pa", str(c["PROFILE"])]
    if c["POLICY"]: a += ["--policy", j(c["POLICY"])]
    if c["SHIELD"]: a += ["--shield", j(c["SHIELD"])]
    if c["GROUP"]: a += ["--group", c["GROUP"]]
    if c["DISGUISE"]: a += ["--disguise"]
    if c["CHAIN_OFF"]: a += ["--chain-off", c["CHAIN_OFF"]]
    if c["SHELL"]: a += ["--shell", j(c["SHELL"])]
    if c["RSHELL"]: a += ["--rshell", j(c["RSHELL"])]
    if c["KNOB_OVER"]: a += ["--knob-over", j(c["KNOB_OVER"])]
    if c["PREEMPT"]: a += ["--preempt", j(c["PREEMPT"])]
    if c["GROUP_KNOBS"]: a += ["--group-knobs", j(c["GROUP_KNOBS"]), "--group-knobs-for", c["GROUP_KNOBS_FOR"]]
    if c["ENDG"]: a += ["--endg", j(c["ENDG"])]
    if c.get("GT"): a += ["--gt", j(c["GT"])]
    if c.get("DISPATCH"): a += ["--dispatch", j(c["DISPATCH"])]
    if c["CLONE_PROFILE"] is not None: a += ["--clone-profile", str(c["CLONE_PROFILE"])]
    if c["CLONE_STRICT"]: a += ["--clone-strict"]
    return j("base"), " ".join(a)

if __name__ == "__main__":
    print(stage_args(sys.argv[1]))
