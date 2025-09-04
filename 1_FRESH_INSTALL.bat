@echo off
title BerzaUnify v2.0 - Fresh Environment Setup

echo.
echo =================================================================
echo.
echo  WARNING: This will DELETE your current 'venv' and '__pycache__'
echo  folders and reinstall all Python packages.
echo.
echo =================================================================
echo.
pause

echo.
echo --- Deleting old virtual environment...
if exist "venv" (
    rmdir /s /q venv
    echo 'venv' folder deleted.
) else (
    echo No 'venv' folder found.
)

echo.
echo --- Deleting Python cache...
if exist "__pycache__" (
    rmdir /s /q __pycache__
    echo '__pycache__' folder deleted.
) else (
    echo No '__pycache__' folder found.
)

echo.
echo --- Creating new virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create venv. Is Python installed and in your PATH?
    pause
    exit /b 1
)

echo.
echo --- Activating environment and installing packages...
call ".\venv\Scripts\activate.bat"
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install packages. Check your internet connection and requirements.txt.
    pause
    exit /b 1
)

echo.
echo --- Fresh Install Complete! ---
echo You can now run the '2_RUN_APP.bat' script.
echo.
pause