@echo off
title BerzaUnify v2.0 - Application Launcher

echo.
echo --- Activating virtual environment...
call ".\venv\Scripts\activate.bat"

echo.
echo --- Launching AutoHotkey Sentry in the background...
start "" "sentry.ahk"

echo.
echo --- Launching Flask Web Server...
echo (To stop the server, press Ctrl+C in this window)
echo.
python app.py

echo.
echo --- Server has been stopped. ---
echo.
pause