@echo off
title Parking Video Analytics - Launcher
echo ========================================
echo   Parking Video Analytics - Spousteni
echo ========================================
echo.

REM Spustit backend v novem okne
echo [1/2] Spoustim backend server...
start "Backend - FastAPI" cmd /k "cd backend && .\venv\Scripts\activate && python main.py"

REM Pockej 3 sekundy aby backend stihl najet
timeout /t 3 /nobreak >nul

REM Spustit frontend v novem okne
echo [2/2] Spoustim frontend server...
start "Frontend - Vite" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================
echo   Servery se spousteji...
echo ========================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo.
echo Aplikace se otevre v prohlizeci za chvili.
echo Tato okna muzete zavrit.
echo.

REM Pockej 8 sekund a otevri prohlizec
timeout /t 8 /nobreak >nul
start http://localhost:5173

echo Hotovo! Pouzivejte aplikaci.
echo.
pause
