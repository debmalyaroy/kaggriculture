@echo off
rem RESTORE (2026-08-21): revert the failed v32 bet by re-submitting our best
rem proven agents as v33.0 -- v25.0_route (peak 2286) + v24.1_bandit (peak
rem 2339). --resume-publish: skip fetch+cycle, go straight to gate ->
rem notebooks -> upload of newest_pair (v33.0).
cd /d D:\codebase\kaggriculture
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "KAGGLE_CONFIG_DIR=C:\Users\debma\.kaggle"
echo ===== publish_pair start %DATE% %TIME% ===== >> data\logs\daily_release.log
"C:\ProgramData\anaconda3\python.exe" -X utf8 scripts\daily_release.py --resume-publish >> data\logs\daily_release.log 2>&1
echo ===== publish_pair exit %ERRORLEVEL% at %DATE% %TIME% ===== >> data\logs\daily_release.log
