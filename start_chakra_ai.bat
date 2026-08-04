@echo off
title Chakra AI
cd /d "%~dp0"
echo.
python -m chakra.cli --preset laptop
echo.
pause
