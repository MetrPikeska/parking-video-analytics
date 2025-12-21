@echo off
title Backend - FastAPI Server
cd backend
call .\venv\Scripts\activate
echo ========================================
echo   Backend Server - FastAPI + YOLO
echo ========================================
echo.
echo Spoustim na http://localhost:8000
echo.
python main.py
pause
