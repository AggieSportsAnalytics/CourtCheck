@echo off
echo ========================================
echo    Installing CourtCheck Dependencies
echo ========================================
echo.

echo [1/4] Installing PyTorch and base dependencies...
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
echo.

echo [2/4] Installing other Python packages (with NumPy 1.x)...
pip install "numpy<2.0" opencv-python==4.8.1.78 scipy==1.11.2 pandas==2.0.3
pip install Pillow==10.0.0 matplotlib==3.7.2 tqdm==4.66.1
pip install catboost fastapi python-multipart "uvicorn[standard]"
echo.

echo [3/4] Installing Detectron2 (this may take a few minutes)...
pip install "git+https://github.com/facebookresearch/detectron2.git"
echo.

echo [4/4] Installing frontend dependencies...
cd frontend
call npm install
cd ..
echo.

echo ========================================
echo    Installation Complete!
echo ========================================
echo.
echo Next step: python run_local.py
echo.
pause
