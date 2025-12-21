# Quick Start Guide

## Windows - Rychlý start

### 1. Instalace dependencies

**Backend:**
```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**Frontend:**
```powershell
cd frontend
npm install
```

### 2. Spuštění

**Terminál 1 - Backend:**
```powershell
cd backend
.\venv\Scripts\activate
python main.py
```

**Terminál 2 - Frontend:**
```powershell
cd frontend
npm run dev
```

### 3. Otevřít aplikaci
Jděte na: http://localhost:5173

---

## Testování

### Test GPU dostupnosti
```powershell
cd backend
.\venv\Scripts\activate
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### Test YOLO modelu
```powershell
cd backend
.\venv\Scripts\activate
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('Model loaded successfully')"
```

---

## Tipy pro optimální výkon

### GTX 1030 (3GB VRAM)
- Model: `yolov8n.pt`
- FPS Sampling: 5
- Confidence: 0.4
- Očekávaná rychlost: ~0.5x real-time (5min video = 2-3min analýzy)

### RTX 3060+ (8GB+ VRAM)
- Model: `yolov8s.pt` nebo `yolov8m.pt`
- FPS Sampling: 10
- Confidence: 0.5
- Očekávaná rychlost: ~1-2x real-time

### CPU only (fallback)
V `backend/video_analyzer.py` změnit:
```python
def __init__(self, model_name: str = "yolov8n.pt", device: str = "cpu"):
```
- Očekávaná rychlost: ~0.1x real-time (velmi pomalé)

---

## Běžné problémy

### "CUDA not available"
1. Zkontrolujte NVIDIA ovladače
2. Nainstalujte CUDA Toolkit 11.8 nebo 12.x
3. Reinstalujte PyTorch:
   ```powershell
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### "ModuleNotFoundError"
```powershell
# Ujistěte se, že je venv aktivován
.\venv\Scripts\activate
# Reinstalujte dependencies
pip install -r requirements.txt
```

### Frontend nenačte API
- Zkontrolujte, že backend běží na portu 8000
- Zkontrolujte firewall
- Zkontrolujte CORS v konzoli prohlížeče

---

## Pro produkční nasazení

### Backend (produkce)
```powershell
cd backend
.\venv\Scripts\activate
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Frontend (build)
```powershell
cd frontend
npm run build
# Výstup v dist/ složce
```

Pak použijte nginx nebo jiný web server pro serving `dist/` složky.
