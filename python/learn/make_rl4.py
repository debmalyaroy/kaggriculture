"""configs/profiles/rl4.json: the PPO3 action table = rl3 (36 market / timing profiles) + OPTION rows (operator 2026-09-27:
"all these rules must be in PPO"). An option row = a base profile PPO already uses + one project / endgame option:
  projects   v219_min_shops (tomato: 103 = also in PIZZA|FARMERS worlds, 99 = never), v233_on, v231_on, hd2_on, cs_on,
             y_on -- read only at the project's commit point, so a row chosen on another day plays exactly its base
             (a started project always finishes: tape sync)
  daily      ca_margin (carrot-for-wheat swap)
  endgame    term_* (planner start / sims / passes / proposals / on-off), tsell_on
Also prints the --init-map (new row <- base row) for warm starts and the oracle candidate list.

    python python/learn/make_rl4.py [--rl5]
"""
import json
import os

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OPTIONS = [
    (35, "tom_world", {"v219_min_shops": 103}),
    (19, "tom_world", {"v219_min_shops": 103}),
    (34, "tom_world", {"v219_min_shops": 103}),
    (35, "tom_off", {"v219_min_shops": 99}),
    (35, "no_v233", {"v233_on": False}),
    (35, "no_v231", {"v231_on": False}),
    (35, "no_hd2", {"hd2_on": False}),
    (35, "no_cs", {"cs_on": False}),
    (35, "no_y", {"y_on": False}),
    (34, "no_v233", {"v233_on": False}),
    (35, "ca30", {"ca_margin": -30.0}),
    (35, "ca10", {"ca_margin": -10.0}),
    (35, "term700_s1024", {"term_start": 700, "term_sims": 1024, "term_passes": 2, "term_props": 8}),
    (35, "term704_s256", {"term_start": 704, "term_sims": 256}),
    (35, "term_off", {"term_on": False}),
    (34, "term700_s1024", {"term_start": 700, "term_sims": 1024, "term_passes": 2, "term_props": 8}),
    (35, "tsell_off", {"tsell_on": False}),
]

# rl5 (operator 2026-09-27: every rule learned): daily rules, each read only when nothing of its own is in flight
# (R51 tour plan live, carrot tiles to follow up); courier plans are per day, E402 / feed / fert act per turn.
OPTIONS5 = [
    (35, "no_r51", {"r51_input_on": False}),
    (34, "no_r51", {"r51_input_on": False}),
    (35, "no_r85feed", {"r85_feed_on": False}),
    (35, "no_courier", {"courier_on": False}),
    (35, "no_carrot", {"carrot_on": False}),
    (35, "no_e402", {"e402_on": False}),
    (35, "no_v9fert", {"v9_fert_on": False}),
]


def main():
    import sys
    v5 = "--rl5" in sys.argv
    rl3 = json.load(open(os.path.join(RL, "configs", "profiles", "rl3.json"), encoding="utf-8"))
    profs = list(rl3["profiles"])
    n0 = len(profs)
    init_map = []
    for base, tag, kn in OPTIONS + (OPTIONS5 if v5 else []):
        p = dict(profs[base])
        p.update(kn)
        p["name"] = f"{profs[base]['name']}+{tag}"
        init_map.append(f"{len(profs)}:{base}")
        profs.append(p)
    name = "rl5" if v5 else "rl4"
    out = {"version": name, "note": "rl3 + project / endgame OPTION rows for the PPO3 planner (python/learn/make_rl4.py)", "profiles": profs}
    json.dump(out, open(os.path.join(RL, "configs", "profiles", name + ".json"), "w", encoding="utf-8"), indent=1)
    print(f"{name}: {n0} + {len(OPTIONS)} = {len(profs)} profiles")
    print("init-map", ",".join(init_map))
    allo = OPTIONS + (OPTIONS5 if v5 else [])
    proj = [n0 + i for i, (_, t, _) in enumerate(allo) if not t.startswith(("term", "tsell"))]
    endg = [n0 + i for i, (_, t, _) in enumerate(allo) if t.startswith(("term", "tsell"))]
    print("oracle projects cands", ",".join(map(str, [35, 34, 31, 19] + proj)))
    print("oracle endgame cands", ",".join(map(str, [35, 34, 31, 19, 0, 13] + endg)))


if __name__ == "__main__":
    main()
