@echo off
title BerzaUnify v2.0 - Cache Cleaner

echo.
echo --- Deleting Python cache...
if exist "__pycache__" (
    rmdir /s /q __pycache__
    echo '__pycache__' folder has been successfully deleted.
) else (
    echo No '__pycache__' folder found to delete.
)

echo.
pause