#!/bin/bash
# Instalační skript pro Linux/Mac

echo "🚀 Parking Video Analytics - Instalace"
echo "======================================="

# Backend
echo "📦 Instaluji backend dependencies..."
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
cd ..

# Frontend
echo "📦 Instaluji frontend dependencies..."
cd frontend
npm install
cd ..

echo ""
echo "✅ Instalace dokončena!"
echo ""
echo "Spuštění:"
echo "1. Backend:  cd backend && source venv/bin/activate && python main.py"
echo "2. Frontend: cd frontend && npm run dev"
echo "3. Otevřít:  http://localhost:5173"
