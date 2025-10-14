@echo off
echo =======================================================
echo Running CITS4403 CA Experiments Batch
echo =======================================================
echo.

REM ---------- Baseline ----------
echo [1/12] Running baseline (sync)
python -m src.ca.run --scheme sync --runs 20

REM ---------- Micro (async + variants) ----------
echo [2/12] Running async only
python -m src.ca.run --scheme async --runs 20 --label micro_async

echo [3/12] Running refractory only
python -m src.ca.run --scheme sync --micro refractory --runs 20

echo [4/12] Running misclass only
python -m src.ca.run --scheme sync --micro misclass --eta 0.02 --runs 20

echo [5/12] Running async + refractory
python -m src.ca.run --scheme async --micro refractory --runs 20 --label async_refractory

echo [6/12] Running async + misclass
python -m src.ca.run --scheme async --micro misclass --eta 0.02 --runs 20 --label async_misclass

echo [7/12] Running refractory + misclass
python -m src.ca.run --scheme sync --micro refractory,misclass --eta 0.02 --runs 20

echo [8/12] Running async + refractory + misclass
python -m src.ca.run --scheme async --micro refractory,misclass --eta 0.02 --runs 20 --label async_refractory_misclass

REM ---------- Macro (hetero / spatial) ----------
echo [9/12] Running heterogeneity only
python -m src.ca.run --macro hetero --hetero-sd 0.20 --runs 20

echo [10/12] Running spatial only
python -m src.ca.run --macro spatial --spatial-strength 0.35 --runs 20

echo [11/12] Running hetero + spatial
python -m src.ca.run --macro hetero,spatial --hetero-sd 0.20 --spatial-strength 0.35 --runs 20

REM ---------- Full model ----------
echo [12/12] Running async + refractory + misclass + hetero + spatial
python -m src.ca.run --scheme async --micro refractory,misclass --eta 0.02 --macro hetero,spatial --hetero-sd 0.20 --spatial-strength 0.35 --runs 20 --label async_refractory_misclass_hetero_spatial

echo.
echo =======================================================
echo All simulations complete!
echo Output saved under: data\runs\ca\
echo =======================================================
pause
