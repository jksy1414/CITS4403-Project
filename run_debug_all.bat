@echo off
setlocal

rem ===============================================
rem  One-time debug run for all micro/macro toggles
rem  Each variant runs once (runs=1)
rem ===============================================

rem --- Phase 1: Baseline & Micro toggles
python -m src.ca.run --scheme sync   --runs 1 --label BL
python -m src.ca.run --scheme async  --runs 1 --label AS
python -m src.ca.run --scheme sync   --micro refractory              --runs 1 --label REF
python -m src.ca.run --scheme sync   --micro misclass --eta 0.02     --runs 1 --label MIS
python -m src.ca.run --scheme async  --micro refractory              --runs 1 --label AS_REF
python -m src.ca.run --scheme async  --micro misclass --eta 0.02     --runs 1 --label AS_MIS
python -m src.ca.run --scheme sync   --micro refractory,misclass --eta 0.02 --runs 1 --label REF_MIS
python -m src.ca.run --scheme async  --micro refractory,misclass --eta 0.02 --runs 1 --label AS_REF_MIS

rem --- Phase 2: Macro toggles
python -m src.ca.run --macro hetero                         --hetero-sd 0.20 --runs 1 --label HET
python -m src.ca.run --macro spatial                        --spatial-strength 0.35 --runs 1 --label SPA
python -m src.ca.run --macro hetero,spatial                 --hetero-sd 0.20 --spatial-strength 0.35 --runs 1 --label HET_SPA

rem --- Phase 3: Combined micro + macro (full feature model)
python -m src.ca.run --scheme async --micro refractory,misclass --eta 0.02 ^
  --macro hetero,spatial --hetero-sd 0.20 --spatial-strength 0.35 ^
  --runs 1 --label FULL

echo.
echo [✓] All debug runs completed.
endlocal
pause
