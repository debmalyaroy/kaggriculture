"""The Track P pipeline -- train the closed-loop planner, never submit.

Daily stages (each wrapped: a stage failure degrades the run, never kills
it -- the same discipline as the release cycle):

  recapture        fold new staged replays into trace v2 (forward capture)
  families         rebuild opening-hash families (feeds priors + novelty)
  datasets         rebuild macro decision datasets when traces grew >= 10%
  anchors          weekly: refresh the anchor pool
  projector        weekly: re-validate price projection vs the naive
  insight          weekly: the four-pass correlation study (dated series)
  iql              weekly (GPU): retrain the warm start
  ppo              daily (GPU): league PPO iterations, warm from last best
  exploiter        daily (GPU): one exploiter round = exploitability meter
  search           daily: CMA-ES over PARAMS, seeded from last best
  build            rebuild agents/planner_v0.py (best params + L1 only if
                   the neural head BEAT the hand rules on the anchor set)
  sim2real         serve-vs-official score delta on the same anchors
  judge            paired OFFICIAL games vs the route incumbent ->
                   planner_judgments.jsonl (generation = build stamp)
  panel            weekly: P1.6 regime-stratified kill-switch panel
  generator        weekly (GPU): retrain; daily: sample + funnel
  validity         weekly: P4.1 open/closed ranking correlation
  graduation       report-only: the 5-condition gate

GPU stages run through the llm env's interpreter; engine stages respect the
worker caps. NOTHING here submits to Kaggle, ever.

Usage: python src/trackp/pipeline.py [--stages a,b,c] [--weekly]
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import datetime as dt
import glob
import json
import os
import subprocess
import sys
import time

from kaggriculture.trackp import common  # noqa: E402

LLM_PY = r"C:/ProgramData/anaconda3/envs/llm/python.exe"
PY = sys.executable
SRC = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(common.MODELS, "pipeline_report.json")
STATE = os.path.join(common.MODELS, "pipeline_state.json")


def _load_state() -> dict:
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save_state(st: dict):
    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=1)


def _run(cmd: list, timeout: int = 3600) -> tuple:
    """A stage child's TIMEOUT is a failed child (code 124), never a dead
    pipeline -- same lesson as refresh_cycle.run(), 2026-08-16."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, cwd=common.ROOT)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        return 124, out + f"\n[TIMED OUT after {timeout}s]"


def _week() -> str:
    d = dt.date.today()
    return f"{d.isocalendar().year}-W{d.isocalendar().week:02d}"


class Pipeline:
    def __init__(self, weekly: bool = False):
        self.state = _load_state()
        self.force_weekly = weekly
        self.results = {}

    def _weekly_due(self, key: str) -> bool:
        return self.force_weekly or self.state.get(key) != _week()

    def _mark_weekly(self, key: str):
        self.state[key] = _week()

    # ------------------------------------------------------------ stages --
    def stage_recapture(self):
        code, out = _run([PY, os.path.join(SRC, "recapture.py"),
                          "--jobs", "6"], timeout=5400)
        return {"ok": code == 0, "tail": out.strip().splitlines()[-1:]}

    def stage_families(self):
        code, out = _run([PY, os.path.join(ROOT, "src", "kaggriculture", "data", "families.py"), "--rebuild"])
        return {"ok": code == 0, "tail": out.strip().splitlines()[-6:]}

    def stage_datasets(self):
        n_traces = len(glob.glob(os.path.join(common.TRACES, "*.npz")))
        last = self.state.get("dataset_traces", 0)
        if last and n_traces < last * 1.10:
            return {"ok": True, "skipped": f"traces {n_traces} < +10%"}
        code1, _ = _run([PY, os.path.join(SRC, "macro.py"),
                         "--out", "macro_dataset.npz"], timeout=3600)
        code2, _ = _run([PY, os.path.join(SRC, "macro.py"),
                         "--elite-gap", "8000",
                         "--out", "macro_dataset_elite.npz"], timeout=3600)
        if code1 == 0 and code2 == 0:
            self.state["dataset_traces"] = n_traces
        return {"ok": code1 == 0 and code2 == 0, "traces": n_traces}

    def stage_anchors(self):
        if not self._weekly_due("w_anchors"):
            return {"ok": True, "skipped": "weekly"}
        code, out = _run([PY, os.path.join(SRC, "league.py"),
                          "--build-anchors", "60"], timeout=1800)
        if code == 0:
            self._mark_weekly("w_anchors")
        return {"ok": code == 0, "tail": out.strip().splitlines()[-6:]}

    def stage_projector(self):
        if not self._weekly_due("w_projector"):
            return {"ok": True, "skipped": "weekly"}
        code, out = _run([PY, os.path.join(SRC, "projections.py"),
                          "--limit", "200"], timeout=3600)
        if code == 0:
            self._mark_weekly("w_projector")
        verdict = ""
        for line in out.splitlines():
            if "verdict" in line:
                verdict = line.strip()
        return {"ok": code == 0, "verdict": verdict}

    def stage_insight(self):
        if not self._weekly_due("w_insight"):
            return {"ok": True, "skipped": "weekly"}
        code, out = _run([PY, os.path.join(SRC, "insight.py")], timeout=5400)
        if code == 0:
            self._mark_weekly("w_insight")
        return {"ok": code == 0, "tail": out.strip().splitlines()[-14:]}

    def stage_verdict(self):
        if not self._weekly_due("w_verdict"):
            return {"ok": True, "skipped": "weekly"}
        code, out = _run([PY, os.path.join(SRC, "verdict_model.py")],
                         timeout=5400)
        if code == 0:
            self._mark_weekly("w_verdict")
        return {"ok": code == 0, "tail": out.strip().splitlines()[-3:]}

    def stage_twins(self):
        if not self._weekly_due("w_twins"):
            return {"ok": True, "skipped": "weekly"}
        code, out = _run([PY, os.path.join(SRC, "twins.py")], timeout=5400)
        if code == 0:
            self._mark_weekly("w_twins")
        return {"ok": code == 0, "tail": out.strip().splitlines()[-6:]}

    def stage_winprob(self):
        if not self._weekly_due("w_winprob"):
            return {"ok": True, "skipped": "weekly"}
        code, out = _run([PY, os.path.join(SRC, "winprob.py")], timeout=7200)
        if code == 0:
            self._mark_weekly("w_winprob")
        return {"ok": code == 0, "tail": out.strip().splitlines()[-8:]}

    def stage_regret(self):
        """On our newest locally-available LOSS replays (quota permitting)."""
        try:
            from kaggriculture.trackp import regret as rg
            import glob as _g
            done = {os.path.basename(p).split("_")[0] for p in
                    _g.glob(os.path.join(common.MODELS, "regret", "*.json"))}
            cands = []
            for p in _g.glob(os.path.join(common.ROOT, "data", "ourgames",
                                          "_stage", "*", "*.json")):
                eid = os.path.splitext(os.path.basename(p))[0]
                if eid.isdigit() and eid not in done \
                        and os.path.getsize(p) > 2_000_000:
                    cands.append((os.path.getmtime(p), eid, p))
            cands.sort(reverse=True)
            mined = 0
            for _, eid, p in cands[:3]:
                rep = common.load_replay(p)
                teams = common.replay_teams(rep)
                if "Debmalya" not in teams:
                    continue
                seat = teams.index("Debmalya")
                banks = common.final_banks(rep)
                if banks[seat] >= banks[1 - seat]:
                    continue                        # losses only
                rg.mine_replay(p, seat)
                mined += 1
            return {"ok": True, "mined": mined,
                    "skipped": "no loss replays staged" if not mined else ""}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def stage_iql(self):
        if not self._weekly_due("w_iql"):
            return {"ok": True, "skipped": "weekly"}
        code, out = _run([LLM_PY, os.path.join(SRC, "iql.py"),
                          "--epochs", "30"], timeout=5400)
        if code == 0:
            self._mark_weekly("w_iql")
        return {"ok": code == 0, "tail": out.strip().splitlines()[-4:]}

    def stage_ppo(self):
        init = [] if os.path.exists(os.path.join(
            common.MODELS, "ppo_best.pt")) else ["--init-iql"]
        code, out = _run([LLM_PY, os.path.join(SRC, "ppo.py"),
                          "--iters", "10", "--episodes", "48",
                          "--jobs", "8"] + init, timeout=5400)
        return {"ok": code == 0, "tail": out.strip().splitlines()[-3:]}

    def stage_exploiter(self):
        if not os.path.exists(os.path.join(common.MODELS,
                                           "l1_weights_best.json")):
            return {"ok": True, "skipped": "no frozen best yet"}
        code, out = _run([LLM_PY, os.path.join(SRC, "ppo.py"),
                          "--exploiter", "--iters", "4",
                          "--episodes", "32", "--jobs", "8"], timeout=3600)
        return {"ok": code == 0, "tail": out.strip().splitlines()[-3:]}

    def stage_search(self):
        code, out = _run([PY, os.path.join(SRC, "search.py"),
                          "--gens", "25", "--panel", "8", "--jobs", "8"],
                         timeout=3600)
        return {"ok": code == 0, "tail": out.strip().splitlines()[-2:]}

    def stage_build(self):
        """Best PARAMS always; L1 weights only when the neural head beat the
        hand rules on the anchor set (recorded by ppo logs vs rules probe)."""
        args = [PY, os.path.join(ROOT, "src", "kaggriculture", "agentbuild", "build_agent.py"),
                "--out", os.path.join(common.ROOT, "agents",
                                      "planner_v0.py")]
        pb = os.path.join(common.MODELS, "params_best.json")
        if os.path.exists(pb):
            args += ["--params", pb]
        use_l1 = False
        lp = os.path.join(common.MODELS, "ppo_log.jsonl")
        if os.path.exists(lp):
            rows = [json.loads(x) for x in open(lp, encoding="utf-8")
                    if x.strip()]
            if rows:
                best_anchor = max(r.get("anchor_score", 0) for r in rows)
                rules_score = self.state.get("rules_anchor_score", 0.0)
                use_l1 = best_anchor > max(0.02, rules_score + 0.02)
        if use_l1:
            args += ["--l1", os.path.join(common.MODELS,
                                          "l1_weights_best.json")]
        code, out = _run(args, timeout=600)
        gen = f"trackp-{dt.date.today().isoformat()}"
        self.state["generation"] = gen
        return {"ok": code == 0, "l1_embedded": use_l1, "generation": gen}

    def stage_sim2real(self):
        """Same anchors, serve vs official -- the drift alarm."""
        code, out = _run([PY, os.path.join(SRC, "sim2real.py")],
                         timeout=3600)
        return {"ok": code == 0, "tail": out.strip().splitlines()[-3:]}

    def stage_judge(self, games: int = 3):
        """Paired official games vs the route incumbent -> judgments."""
        try:
            from kaggriculture.trackp import arena, graduation, guard
            incumbent = graduation.route_incumbent()
            planner = os.path.join(common.ROOT, "agents", "planner_v0.py")
            res = arena.paired(planner, incumbent, n=games,
                               seed0=int(time.time()) % 100000 + 11)
            gen = self.state.get("generation", "untagged")
            rows = [{"generation": gen, "seed": r["seed"], "seat": r["seat"],
                     "bank_me": r["bank_a"], "bank_opp": r["bank_b"],
                     "opponent": os.path.basename(incumbent),
                     "ts": time.time()} for r in res["rows"]]
            guard.record(rows)
            verdict = guard.planner_allowed(verbose=False)
            return {"ok": True, "judged_now": len(rows),
                    "guard": {k: verdict[k] for k in
                              ("allowed", "n", "reason")}}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def stage_panel(self):
        if not self._weekly_due("w_panel"):
            return {"ok": True, "skipped": "weekly"}
        try:
            from kaggriculture.trackp import graduation
            incumbent = graduation.route_incumbent()
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)}
        code, out = _run([PY, os.path.join(ROOT, "src", "kaggriculture", "measure", "panel.py"),
                          "--vs", incumbent, "--per-stratum", "2"],
                         timeout=7200)
        if code == 0:
            self._mark_weekly("w_panel")
        return {"ok": code == 0, "tail": out.strip().splitlines()[-12:]}

    def stage_generator(self):
        if self._weekly_due("w_generator"):
            code, out = _run([LLM_PY, os.path.join(SRC, "generator.py"),
                              "--epochs", "10"], timeout=7200)
            if code != 0:
                return {"ok": False, "tail": out.strip().splitlines()[-4:]}
            self._mark_weekly("w_generator")
        code, out = _run([LLM_PY, os.path.join(SRC, "generator.py"),
                          "--sample", "48"], timeout=1800)
        if code != 0:
            return {"ok": False, "tail": out.strip().splitlines()[-4:]}
        code, out = _run([PY, os.path.join(SRC, "gen_funnel.py"),
                          "--top", "4", "--confirm", "2"], timeout=5400)
        return {"ok": code == 0, "tail": out.strip().splitlines()[-8:]}

    def stage_validity(self):
        if not self._weekly_due("w_validity"):
            return {"ok": True, "skipped": "weekly"}
        code, out = _run([PY, os.path.join(SRC, "validity.py"),
                          "--k", "16", "--jobs", "8"], timeout=3600)
        if code == 0:
            self._mark_weekly("w_validity")
        return {"ok": code == 0, "tail": out.strip().splitlines()[-6:]}

    def stage_graduation(self):
        code, out = _run([PY, os.path.join(SRC, "graduation.py"),
                          "--panel-n", "6", "--holdout-n", "3"],
                         timeout=7200)
        return {"ok": code == 0, "tail": out.strip().splitlines()[-16:]}

    def stage_phase0probe(self):
        """The phase-0 currency, refreshed daily: bank ratio vs the route
        incumbent + a takeover-regret probe (search/build/judge already run
        as their own stages, so this only adds the two measurements)."""
        try:
            import datetime as _dt
            import glob as _glob
            from kaggriculture.trackp import arena, graduation
            inc = graduation.route_incumbent()
            res = arena.paired(os.path.join(common.ROOT, "agents",
                                            "planner_v0.py"), inc,
                               n=2, seed0=52011)
            ratio = (sum(r["bank_a"] for r in res["rows"])
                     / max(1.0, sum(r["bank_b"] for r in res["rows"])))
            with open(os.path.join(common.MODELS, "phase0_arena.json"),
                      "w", encoding="utf-8") as fh:
                json.dump({"bank_ratio": round(ratio, 3),
                           "score": res["score_a"],
                           "incumbent": os.path.basename(inc),
                           "when": _dt.datetime.now().isoformat(
                               timespec="seconds")}, fh, indent=1)
            cands = sorted(_glob.glob(os.path.join(
                common.ROOT, "data", "sameday", "_stage", "*", "*.json")),
                key=os.path.getmtime, reverse=True)
            pick = next((c for c in cands
                         if os.path.getsize(c) > 3_000_000), None)
            if pick:
                _run([PY, os.path.join(SRC, "regret.py"),
                      "--replay", pick, "--seat", "0"], timeout=3600)
            return {"ok": True, "bank_ratio": round(ratio, 3)}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def stage_phases(self):
        """Refresh the roadmap gate board (models/trackp/phase_report.json)
        as the run's last act, so 'where are we' is always one file away."""
        code, out = _run([PY, os.path.join(SRC, "phases.py"), "--status"],
                         timeout=600)
        return {"ok": code == 0, "tail": out.strip().splitlines()[-5:]}

    # -------------------------------------------------------------- run --
    ORDER = ["recapture", "families", "datasets", "anchors", "projector",
             "insight", "verdict", "twins", "winprob", "regret",
             "iql", "ppo", "exploiter", "search", "build",
             "sim2real", "judge", "panel", "generator", "validity",
             "graduation", "phase0probe", "phases"]

    def run(self, stages=None):
        stages = stages or self.ORDER
        t0 = time.time()
        for name in stages:
            fn = getattr(self, f"stage_{name}", None)
            if fn is None:
                self.results[name] = {"ok": False, "error": "unknown stage"}
                continue
            ts = time.time()
            try:
                res = fn()
            except Exception as e:  # noqa: BLE001 -- degrade, never die
                res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            res["secs"] = round(time.time() - ts, 1)
            self.results[name] = res
            print(f"[{name}] {'OK' if res.get('ok') else 'FAIL'} "
                  f"({res['secs']}s) "
                  f"{res.get('skipped', '')}{res.get('error', '')}",
                  flush=True)
            _save_state(self.state)
        report = {"date": dt.datetime.now().isoformat(timespec='seconds'),
                  "total_secs": round(time.time() - t0, 1),
                  "stages": self.results}
        with open(REPORT, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        print(json.dumps({"total_secs": report["total_secs"],
                          "failed": [k for k, v in self.results.items()
                                     if not v.get("ok")]}))
        return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="")
    ap.add_argument("--weekly", action="store_true",
                    help="force the weekly stages to run now")
    a = ap.parse_args()
    stages = [s.strip() for s in a.stages.split(",") if s.strip()] or None
    Pipeline(weekly=a.weekly).run(stages)
