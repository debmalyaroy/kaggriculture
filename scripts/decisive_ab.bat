@echo off
rem Decisive A/B (2026-08-20 publish-now): the tournament winner
rem (91664668_s1, +6pp, held by the +10pp margin gate) vs the LIVE incumbent
rem route. Significant edge => ship it (operator override of the margin gate,
rem per the explicit publish order). Not significant => hold, do not ship.
cd /d D:\codebase\kaggriculture
C:\ProgramData\anaconda3\python.exe -u src\kaggriculture\measure\ab_test.py data\refresh\2026-08-20\cand\c_episode-91664668-replay_s1.py agents\v29.0_route.py -n 16 >> data\logs\decisive_ab.log 2>&1
echo ===== decisive_ab done %DATE% %TIME% ===== >> data\logs\decisive_ab.log
