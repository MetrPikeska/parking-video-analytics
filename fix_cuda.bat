@echo off
echo ========================================
echo   Instalace PyTorch s CUDA podporou
echo ========================================
echo.

cd backend
call .\venv\Scripts\activate

echo Odstranuji stavajici PyTorch...
pip uninstall -y torch torchvision

echo.
echo Instaluji PyTorch s CUDA 11.8...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo.
echo ========================================
echo Testovani CUDA...
echo ========================================
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo Hotovo! Zkuste znovu spustit aplikaci.
pause
