#!/bin/bash
# (2) LEAGUE REFRESH LOOP -- run detached on the box. The box cannot rank agents
# locally (no crown panel / ratings there), so it PULLS the latest top-agents pack
# that you built + uploaded from local (agent_fetch.py -> league_refresh.py --pack
# -> gdrive:kaggriculture/top_agents/), extracts it (binaries keep exec bits), and
# rebuilds the consumable roster every REFRESH_HOURS.
#
#   nohup setsid bash scripts/trackp/league_refresh_loop.sh >/root/kaggriculture/ckpts/league.log 2>&1 </dev/null &
#
# Stop:  pkill -f league_refresh_loop.sh
set -u
cd /root/kaggriculture
export PATH=/venv/main/bin:$PATH
export PYTHONPATH=src:vendor
PY=${PY:-/venv/main/bin/python}
GDRIVE=${GDRIVE:-gdrive:kaggriculture}
REFRESH_HOURS=${REFRESH_HOURS:-6}
TOP=/root/kaggriculture/.local/top_agents
mkdir -p "$TOP"
while true; do
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) league pull+refresh ==="
  # pull any new top_agents_*.tar.gz packs from gdrive
  rclone copy "$GDRIVE/top_agents" "$TOP" --include "top_agents_*.tar.gz" || echo "(pull skipped)"
  NEWEST=$(ls -1t "$TOP"/top_agents_*.tar.gz 2>/dev/null | head -1)
  if [ -n "${NEWEST:-}" ]; then
    $PY scripts/trackp/league_refresh.py --from-tar "$NEWEST" || echo "(extract failed -- retry next cycle)"
  else
    echo "(no top_agents pack in $GDRIVE/top_agents yet -- upload one from local)"
  fi
  sleep "$((REFRESH_HOURS * 3600))"
done
