@echo off
setlocal

where conda
:: Prompt the user to enter the Anaconda installation path
echo "Please enter the same information again"
set /p CONDA_PATH="Enter the full path to your Anaconda installation (e.g., C:\Users\YourName\anaconda3): "

:: Add the Anaconda directories to PATH without truncating the existing PATH
setx PATH "%CONDA_PATH%;%CONDA_PATH%\Scripts;%CONDA_PATH%\Library\bin"

echo Conda has been added to PATH. Please restart your terminal.
pause