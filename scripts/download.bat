@echo off
REM Double-click me, or run from a terminal.
cd /d "%~dp0.."
echo Kaggriculture data download
echo.
python scripts\download_data.py %*
echo.
pause
