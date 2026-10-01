@echo off
rem Unattended launcher for the daily release (KaggricultureRefreshCycle).
rem
rem PUBLISHING ENABLED (operator order 2026-08-14 night): the run uploads the
rem daily pair through the standard sha-verified path. The rails that make
rem unattended publishing sane are all live in daily_release.py:
rem   * gate() -- a red submission gate aborts the upload, never bypassed;
rem   * already_live() -- an unchanged agent is never re-uploaded (that would
rem     evict a developed rating for a provisional copy of identical code);
rem   * quota floor -- uploads hold when the daily reset is <3h away;
rem   * sha-verified kernel output -- never submits a stale main.py;
rem   * LAB-stamped builds are refused outright by submit.py.
rem To park the ladder again, re-add --no-submit to the line below.
rem Fully self-contained: absolute python path (no PATH dependence), quoted
rem env (Task Scheduler's inline `set X=1 &&` once passed "1 " with a trailing
rem space -> fatal PYTHONUTF8 error), explicit Kaggle credential dir (so the
rem task also works if run as SYSTEM), append-only log with per-run headers.
cd /d D:\codebase\kaggriculture
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "KAGGLE_CONFIG_DIR=C:\Users\debma\.kaggle"
if not exist data\logs mkdir data\logs
set "LOG=data\logs\daily_release.log"
echo ===== daily_release start %DATE% %TIME% ===== >> "%LOG%"
"C:\ProgramData\anaconda3\python.exe" -X utf8 scripts\daily_release.py >> "%LOG%" 2>&1
echo ===== daily_release exit %ERRORLEVEL% at %DATE% %TIME% ===== >> "%LOG%"
