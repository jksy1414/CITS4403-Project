@echo off
setlocal

rem =====================================================
rem  CITS4403 — Full CA Experiment, Phases 0..3
rem  20 runs per variant (for stable averages)
rem =====================================================

echo [START] Running all phases (0..3) with 20 simulations each...
echo.

rem ---------- PHASE 0: Baseline ----------
python -m src.ca.run --scheme sync   --runs 20 --label BL

rem ---------- PHASE 1: Micro toggles ----------
python -m src.ca.run --scheme async                                --runs 20 --label AS
python -m src.ca.run --scheme sync   --micro refractory            --runs 20 --label REF
python -m src.ca.run --scheme sync   --micro misclass --eta 0.02   --runs 20 --label MIS
python -m src.ca.run --scheme async  --micro refractory            --runs 20 --label AS_REF
python -m src.ca.run --scheme async  --micro misclass --eta 0.02   --runs 20 --label AS_MIS
python -m src.ca.run --scheme sync   --micro refractory,misclass --eta 0.02 --runs 20 --label REF_MIS
python -m src.ca.run --scheme async  --micro refractory,misclass --eta 0.02 --runs 20 --label AS_REF_MIS

rem ---------- PHASE 2: Macro toggles ----------
python -m src.ca.run --macro hetero                         --hetero-sd 0.20          --runs 20 --label HET
python -m src.ca.run --macro spatial                        --spatial-strength 0.35    --runs 20 --label SPA
python -m src.ca.run --macro hetero,spatial                 --hetero-sd 0.20 --spatial-strength 0.35 --runs 20 --label HET_SPA

rem ---------- PHASE 3: Combined (Micro + Macro + Async) ----------
python -m src.ca.run --scheme async --micro refractory,misclass --eta 0.02 ^
  --macro hetero,spatial --hetero-sd 0.20 --spatial-strength 0.35 ^
  --runs 20 --label FULL

echo.
echo [✓] All phases completed.
echo Results in: data\runs\ca\<label>_<timestamp>\ (each has summary.csv and summary.json)
pause
endlocal
