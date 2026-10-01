@echo off
REM Double-click me, or run from a terminal.
REM   dashboard.bat              locked submit (default)
REM   dashboard.bat -AllowSubmit arms the submit button
cd /d "%~dp0.."
echo Starting the Kaggriculture control panel...
echo.
powershell -ExecutionPolicy Bypass -File "%~dp0dashboard.ps1" %*
echo.
pause
