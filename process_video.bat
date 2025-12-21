@echo off
title Video Processor - Parking Analytics
echo ========================================
echo   Video Processor - Parking Analytics
echo ========================================
echo.
echo Umistete MP4 video do slozky: backend\input\
echo.
pause
echo.
echo Spoustim analyzu...
echo.

cd backend
call .\venv\Scripts\activate
python process_video.py

echo.
echo ========================================
echo Vysledek najdete v: backend\output\
echo ========================================
pause
