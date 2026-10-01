import sys, os, json; sys.stdout.reconfigure(encoding="utf-8",errors="replace")
from kaggriculture.trackp import routes_io as R
import kaggriculture.pipeline.sell_search as SS
idx=R.load_index()["routes"]
# strong current frontier: top winning routes by (won, high opp_bank => beat strong opp), fresh
recent=[(v.get('date',''),k,v.get('bank',0),v.get('opp_bank',0),v.get('team','')) for k,v in idx.items()
        if v.get('engine')=='1.32.7' and v.get('won') and str(v.get('date',''))>='2026-09-06' and (v.get('bank') or 0)>=100000 and (v.get('opp_bank') or 0)>=90000]
recent.sort(key=lambda x:-(x[2]+x[3]))
OUT=".local/frontier_panel"; os.makedirs(OUT,exist_ok=True)
seen=set(); tapes=[]
for d,k,b,ob,t in recent:
    if t in seen: continue
    try:
        tapes.append(SS.write_tape(R.load_route(k), os.path.join(OUT,f"{k}.tape"))); seen.add(t)
    except Exception: continue
    if len(tapes)>=8: break
json.dump(tapes, open(os.path.join(OUT,"panel.json"),"w"))
print(f"frontier panel: {len(tapes)} strong current routes (beat >=90k opponents)")
for t in tapes[:8]: print(" ", os.path.basename(t))
