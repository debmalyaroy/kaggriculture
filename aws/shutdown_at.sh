#!/usr/bin/env bash
# Power the box off at a UTC time (default: before the submission lock), so it can never run past the budget.
#   bash aws/shutdown_at.sh "2026-09-29 22:00"      # replaces any earlier schedule
set -euo pipefail
WHEN=${1:-2026-09-29 22:00}
sudo systemctl stop krl-shutdown.timer 2>/dev/null || true
sudo tee /etc/systemd/system/krl-shutdown.service >/dev/null <<EOF
[Unit]
Description=Power off the RL box before the lock
[Service]
Type=oneshot
ExecStart=/usr/bin/systemctl poweroff
EOF
sudo tee /etc/systemd/system/krl-shutdown.timer >/dev/null <<EOF
[Unit]
Description=Power off the RL box at $WHEN UTC
[Timer]
OnCalendar=$WHEN UTC
Persistent=false
[Install]
WantedBy=timers.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now krl-shutdown.timer
systemctl list-timers krl-shutdown.timer --no-pager | head -3
