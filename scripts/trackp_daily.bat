@echo off
rem Track P daily training pipeline -- NEVER submits (the pipeline has no
rem submit path at all; graduation is report-only + operator go).
rem Scheduled 13:30 daily as KaggricultureTrackP: off the 04:00 autopilot /
rem 04:30 release critical path, and the hourly fetch is concurrent-safe
rem against our recapture scan.

cd /d D:\codebase\kaggriculture
echo ===== trackp %date% %time% ===== >> data\logs\trackp_daily.log
python src\kaggriculture\trackp\pipeline.py >> data\logs\trackp_daily.log 2>&1
echo ===== trackp exit %errorlevel% %time% ===== >> data\logs\trackp_daily.log
