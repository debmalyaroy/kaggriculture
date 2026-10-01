@echo off
rem One-off paired A/B: the crown-HELD +6pp candidate vs the incumbent
rem rebuild (docs/history/pair-improvement-plan.md 2026-08-20 decision point).
cd /d D:\codebase\kaggriculture
C:\ProgramData\anaconda3\python.exe -u src\kaggriculture\measure\ab_test.py data\refresh\2026-08-20\cand\c_episode-91583525-replay_s0.py agents\v31.0_route.py -n 16 >> data\logs\ab_crown_check.log 2>&1
