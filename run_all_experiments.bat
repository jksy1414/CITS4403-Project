@echo off
echo === Starting All CA Experiments ===

REM BASELINE & MICRO
python -m src.ca.run
python -m src.ca.run --scheme async --micro async
python -m src.ca.run --micro refractory
python -m src.ca.run --micro misclass --eta 0.02
python -m src.ca.run --scheme async --micro async,refractory
python -m src.ca.run --scheme async --micro async,misclass --eta 0.02
python -m src.ca.run --micro refractory,misclass --eta 0.02
python -m src.ca.run --scheme async --micro async,refractory,misclass --eta 0.02

REM MACRO (HETEROGENEITY)
python -m src.ca.run --macro hetero --hetero-sd 0.20
python -m src.ca.run --scheme async --micro async --macro hetero --hetero-sd 0.20
python -m src.ca.run --micro refractory --macro hetero --hetero-sd 0.20
python -m src.ca.run --micro misclass --eta 0.02 --macro hetero --hetero-sd 0.20
python -m src.ca.run --scheme async --micro async,refractory --macro hetero --hetero-sd 0.20
python -m src.ca.run --scheme async --micro async,misclass --eta 0.02 --macro hetero --hetero-sd 0.20
python -m src.ca.run --micro refractory,misclass --eta 0.02 --macro hetero --hetero-sd 0.20
python -m src.ca.run --scheme async --micro async,refractory,misclass --eta 0.02 --macro hetero --hetero-sd 0.20

echo === All Experiments Completed ===
pause
