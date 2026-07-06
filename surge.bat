@echo off
REM Wrapper script to run SURGE CLI from Windows PowerShell/CMD
REM Usage: surge run config.yaml
REM        surge analyze config.yaml
REM        surge viz runs/my_run/

setlocal enabledelayedexpansion

REM Activate venv if it exists
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

REM Run surge CLI via Python module
python -m surge.cli %*
