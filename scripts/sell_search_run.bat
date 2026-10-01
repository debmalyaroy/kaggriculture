@echo off
rem Refined sell-schedule search (docs/history/pair-improvement-plan.md, lever L5).
rem Scheduled AFTER the 04:30 cycle so kagg batch never overlaps the
rem tournament (machine-wide 3-wide engine cap, BSOD rule).
cd /d D:\codebase\kaggriculture
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
C:\ProgramData\anaconda3\python.exe -X utf8 -u src\kaggriculture\pipeline\sell_search.py --iters 300 --seeds 16 --holdout-seeds 48 --patience 60 --resume --out D:\codebase\kaggriculture\agents\factory_sell_v29.py >> data\logs\sell_search.log 2>&1
