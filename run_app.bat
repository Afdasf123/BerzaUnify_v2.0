@echo off
title BerzaUnify v2.0 Server

:: Ensure the script runs from its own directory (project root)
cd /d "%~dp0"

:: 3) Check for the venv directory
if not exist "venv" (
    echo ERROR: venv not found. Please run 'python -m venv venv' first.
    pause
    exit /b 1
)

:: 4) Activate the virtual environment for this session
call "venv\Scripts\activate.bat"

:: 5) Run the Python server
echo Starting BerzaUnify v2.0 server...
python app.py

:: Report exit code and wait so user can read final messages
echo.
echo Server process exited with exit code %ERRORLEVEL%.

:: 6) Pause on exit
pause
