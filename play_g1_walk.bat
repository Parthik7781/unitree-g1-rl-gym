@echo off
REM Script to test/play the trained G1 walking policy

echo.
echo ========================================
echo Playing Trained G1 Walking Policy
echo ========================================
echo.
echo Make sure you have completed training first!
echo.

python legged_gym/scripts/play.py --task=g1

pause
