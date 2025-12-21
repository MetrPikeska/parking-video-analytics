@echo off
REM Instalační skript pro Windows

echo 🚀 Parking Video Analytics - Instalace
echo =======================================

REM Backend
echo 📦 Instaluji backend dependencies...
cd backend
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
cd ..

REM Frontend
echo 📦 Instaluji frontend dependencies...
cd frontend
call npm install
cd ..

echo.
echo ✅ Instalace dokončena!
echo.
echo Spuštění:
echo 1. Backend:  cd backend ^&^& venv\Scripts\activate ^&^& python main.py
echo 2. Frontend: cd frontend ^&^& npm run dev
echo 3. Otevřít:  http://localhost:5173
pause
