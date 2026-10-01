#!/usr/bin/env bash
# Hourly: refresh the live pair's board files and download their new games (no parsing).
#   bash scripts/live_board_loop.sh            # runs now, then at the top of every hour
# Each run appends one LIVE_BOARD line to .local/live_board/loop.log. The board itself is written
# from those files (.local/live_board/live.json, status_patch.json) by the session's hourly push.
cd "$(dirname "$0")/.." || exit 1
mkdir -p .local/live_board
while true; do
  python -m kaggriculture.pipeline.live_board "$@" >> .local/live_board/loop.log 2>&1
  now=$(date +%s)
  sleep $(( 3600 - now % 3600 + 60 ))   # one minute past the next hour
done
