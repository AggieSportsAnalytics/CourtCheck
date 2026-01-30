@echo off
echo ========================================
echo    Fixing NumPy Version Conflict
echo ========================================
echo.

echo Uninstalling NumPy 2.x...
pip uninstall -y numpy

echo.
echo Installing NumPy 1.x (compatible version)...
pip install "numpy<2.0"

echo.
echo Reinstalling packages that depend on NumPy...
pip install --force-reinstall --no-deps opencv-python==4.8.1.78
pip install --force-reinstall --no-deps scipy==1.11.2
pip install --force-reinstall --no-deps pandas==2.0.3
pip install --force-reinstall --no-deps matplotlib==3.7.2

echo.
echo ========================================
echo    NumPy Fix Complete!
echo ========================================
echo.
echo NumPy version installed:
python -c "import numpy; print(f'NumPy {numpy.__version__}')"
echo.
echo Now try: python run_local.py
echo.
pause
