@echo off
title Audio Cascader Web POC
cd /d "%~dp0"
set PYTHONPATH=%~dp0
".venv\Scripts\python.exe" app.py
pause
