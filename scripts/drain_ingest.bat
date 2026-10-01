@echo off
rem Drain the 1.32.7 staged-replay ingest backlog (2026-08-20). ~15.5k raw
rem 1.32.7 replays were downloaded but never ingested, leaving the index (and
rem thus the crown pool + models) blind to the current-engine meta while the
rem ladder plays 1.32.7. jobs=3 to respect the machine-wide load ceiling with
rem the other project's RL job running (BSOD history).
cd /d D:\codebase\kaggriculture
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
C:\ProgramData\anaconda3\python.exe -X utf8 -u src\kaggriculture\data\backfill_ingest.py --jobs 3 >> data\logs\drain_ingest.log 2>&1
echo ===== drain_ingest done %DATE% %TIME% ===== >> data\logs\drain_ingest.log
