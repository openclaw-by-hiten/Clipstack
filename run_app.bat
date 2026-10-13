@echo off
title Neural Clip Studio — AI Viral Clipper
cd /d "%~dp0"

echo =========================================================
echo    NEURAL CLIP STUDIO (TRIBE v2 + Faster-Whisper)
echo    100%% Local Video Clipping System
echo =========================================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in %~dp0venv
    pause
    exit /b 1
)

echo [1/2] Starting Local Server on http://localhost:8000 ...
start "" http://localhost:8000
"%~dp0venv\Scripts\python.exe" server.py

pause
