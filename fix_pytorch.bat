@echo off
echo ========================================
echo   Fix PyTorch DLL Error
echo ========================================
echo.

cd backend
call .\venv\Scripts\activate

echo [1/3] Odstranuji problematickou instalaci...
pip uninstall -y torch torchvision torchaudio

echo.
echo [2/3] Instaluji PyTorch znovu (CPU verze)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo.
echo [3/3] Testovani...
python -c "import torch; print('PyTorch version:', torch.__version__); print('Import OK!')"

echo.
echo ========================================
echo Hotovo! Zkuste znovu spustit aplikaci.
echo ========================================
echo.
echo POZNAMKA: Pokud chcete CUDA podporu, spustte fix_cuda.bat
echo ale MUSI mit nainstalovany:
echo 1. NVIDIA ovladace
echo 2. CUDA Toolkit 11.8
echo 3. Visual C++ Redistributable 2015-2022
echo.
pause
