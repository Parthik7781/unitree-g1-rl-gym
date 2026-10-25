@echo off
REM Script to train and test G1 robot walking

echo.
echo ========================================
echo Unitree G1 Robot Walking Setup
echo ========================================
echo.

REM Check if Python is available
python --version > nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

echo.
echo [1/3] Installing package in development mode...
pip install -e .

echo.
echo [2/3] Starting G1 Robot Training...
echo This will train a walking policy for the G1 robot.
echo Press Ctrl+C to stop training at any time.
echo.
python legged_gym/scripts/train.py --task=g1

pause
