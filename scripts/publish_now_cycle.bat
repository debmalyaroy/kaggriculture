@echo off
rem Operator "run and publish now" (2026-08-20), stage 1: build + crown with
rem the hinge quota live. Serve substrate = one Rust process (machine-wide
rem 3-engine cap; other projects' jobs are running + BSOD history).
cd /d D:\codebase\kaggriculture
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set KAGG_RELEASE=1
C:\ProgramData\anaconda3\python.exe -X utf8 -u src\kaggriculture\pipeline\refresh_cycle.py --no-fetch --include-foreign >> data\logs\publish_now.log 2>&1
echo ===== cycle stage done %DATE% %TIME% ===== >> data\logs\publish_now.log
