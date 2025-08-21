@echo off
echo Activating Python environment...

where conda 

REM Activate the base environment
CALL conda activate base

echo Running Python script...
cd /d "%~dp0"
cd "src"
python "main.py"

REM Deactivate Conda environment
CALL conda deactivate

pause