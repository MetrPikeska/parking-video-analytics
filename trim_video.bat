@echo off
title Video Trimmer - 5min -> 2min
echo ========================================
echo   Video Trimmer: 5min -^> 2min
echo ========================================
echo.
echo Zkrati video ze slozky backend\input\
echo na prvnich 2 minuty.
echo.
pause
echo.

cd backend
call .\venv\Scripts\activate
python trim_video.py

echo.
echo ========================================
echo Vysledek: backend\output\*_2min.mp4
echo ========================================
pause
