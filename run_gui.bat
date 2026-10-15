@echo off
title Audio Cascader - S2S Voice Converter
cd /d "%~dp0"
set PYTHONPATH=%~dp0
echo ==============================================================
echo Launching Audio Cascader (Indian to American S2S Converter)
echo ==============================================================
".venv\Scripts\python.exe" gui.py
pause
