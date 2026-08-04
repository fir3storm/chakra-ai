@echo off
title Chakra AI - Agentic Code Terminal
cd /d "%~dp0"
echo.
echo =======================================================================
echo  Chakra AI - Agentic Code Terminal
echo  github.com/fir3storm/chakra-ai
echo.
echo  Engine C (default): Qwen2.5-Coder-1.5B, ~2.5 GB RAM, any OS
echo  Engine B:           PyTorch Kimi K3, ~8 GB RAM, Windows
echo  Engine A:           kimi-k3-in-c, 8.24 GB RAM, Linux
echo.
echo  Commands: /help /status /team /persona /audit /scan-vuln /context /tree
echo =======================================================================
echo.
python -m kimipy.cli --preset laptop
echo.
echo =======================================================================
echo Session ended. Press any key to exit.
echo =======================================================================
pause
