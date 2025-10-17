@echo off
echo === Starting All CA Experiments ===

REM BASELINE
python -m src.ca.run --scheme sync 


REM MICRO
python -m src.ca.run --scheme async --label micro_async
python -m src.ca.run --scheme sync  --micro refractory              
python -m src.ca.run --scheme sync  --micro misclass   --eta 0.02   
python -m src.ca.run --scheme async --micro refractory    --label async_refractory
python -m src.ca.run --scheme async --micro misclass   --eta 0.02    --label async_misclass
python -m src.ca.run --scheme sync  --micro refractory,misclass     --eta 0.02 
python -m src.ca.run --scheme async --micro refractory,misclass     --eta 0.02  --label async_refractory_misclass


REM MACRO (HETEROGENEITY)
python -m src.ca.run --macro hetero               --hetero-sd 0.20          
python -m src.ca.run --macro spatial              --spatial-strength 0.35    
python -m src.ca.run --macro hetero,spatial       --hetero-sd 0.20 --spatial-strength 0.35 

REM All 
python -m src.ca.run --scheme async --micro refractory,misclass --eta 0.02 --macro hetero,spatial --hetero-sd 0.20 --spatial-strength 0.35  --label async_refractory_misclass_hetero_spatial

REM Base line batch fun 
python -m src.ca.run --scheme sync --runs 20


echo === All Experiments Completed ===
pause
