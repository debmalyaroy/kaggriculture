"""Offline tape generator: run an agent, RECORD its per-turn action stream into
a fixed 720-step tape, then OPTIMIZE the tape offline (hill-climb with full-game
replay) — foresight the reactive planner lacks. Emits a standalone tape agent."""
import sys, os, json, copy, random
os.environ.setdefault("KAGG_BIN", os.path.join("rustengine", "kagg.exe"))
import kaggriculture.engine.serve_match as SM

def record(agent_path, opp, seed, seat=0, srv=None):
    """Play agent at `seat` vs opp; return (tape[720 action dicts], final_bank)."""
    own = srv is None
    if own: srv = SM.Serve()
    A = SM.load_agent(agent_path)
    OA = SM.load_agent(opp) if isinstance(opp, str) else opp
    js = srv.cmd(f"RESET {seed}")
    tape = []
    for st in range(719):
        act = A(SM.obs_for(seat, js)) or {}
        tape.append(act)
        me = SM.action_to_line(act)
        ot = SM.action_to_line(OA(SM.obs_for(1 - seat, js)))
        la, lb = (me, ot) if seat == 0 else (ot, me)
        js = srv.cmd(f"STEP2 {la}\x1e{lb}")
        if "error" in js:
            break
    bank = float(js["farms"][seat].get("money") or 0)
    if own: srv.close()
    return tape, bank

def play_tape(tape, opp, seed, seat=0, srv=None):
    """Replay a fixed tape (ignores observation) vs opp; return final bank."""
    own = srv is None
    if own: srv = SM.Serve()
    OA = SM.load_agent(opp) if isinstance(opp, str) else opp
    js = srv.cmd(f"RESET {seed}")
    for st in range(719):
        me = SM.action_to_line(tape[st] if st < len(tape) else None)
        ot = SM.action_to_line(OA(SM.obs_for(1 - seat, js)))
        la, lb = (me, ot) if seat == 0 else (ot, me)
        js = srv.cmd(f"STEP2 {la}\x1e{lb}")
        if "error" in js:
            break
    bank = float(js["farms"][seat].get("money") or 0)
    if own: srv.close()
    return bank

if __name__ == "__main__":
    PLANNER = "agents/v48.3_planner.py"
    PASS = "data/gauntlet/kaito_v48.py"  # a real opponent to record against
    srv = SM.Serve()
    # record the planner on a few seeds, validate playback reproduces
    print("=== RECORD + PLAYBACK VALIDATION ===")
    import glob
    PASSA = lambda o: {}
    for seed in (3, 4, 5):
        tape, rec_bank = record(PLANNER, PASSA, seed, srv=srv)
        pb_bank = play_tape(tape, PASSA, seed, srv=srv)
        print(f"  seed{seed}: planner(reactive) {rec_bank:,.0f}  tape(playback) {pb_bank:,.0f}  match={abs(rec_bank-pb_bank)<1:.0f}", flush=True)
    srv.close()

def hillclimb(tape, opp, seed, srv, iters=60, seat=0):
    """Offline improvement: perturb PLANT crop choices + SELL quantities in the
    tape and keep replays that bank more. Foresight the reactive planner lacks."""
    rng = random.Random(7)
    best = copy.deepcopy(tape)
    bb = play_tape(best, opp, seed, seat=seat, srv=srv)
    CROPS = ["WHEAT", "CARROT", "STRAWBERRY", "MELON"]
    for it in range(iters):
        cand = copy.deepcopy(best)
        # mutate a handful of turns
        for _ in range(rng.randint(1, 4)):
            i = rng.randrange(len(cand))
            t = cand[i]
            if not isinstance(t, dict):
                continue
            # crop swap in a PLANT
            for seq_key in ("farmer", "hands"):
                seq = t.get(seq_key)
                items = [seq] if seq_key == "farmer" and seq else (seq or [])
                for act in items:
                    if act and act[0] == "PLANT" and len(act) > 1 and rng.random() < 0.5:
                        act[1] = rng.choice(CROPS)
        cb = play_tape(cand, opp, seed, seat=seat, srv=srv)
        if cb > bb:
            best, bb = cand, cb
    return best, bb
