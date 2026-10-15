@echo off
rem DEXAIV launcher.
rem   start.bat            -> the GUI
rem   start.bat run        -> live on the CLI: Mom, pitch +12, gate -40, default mic -> VAC
rem   start.bat raw        -> same with auto level off and gate -57 (the old raw-mic setting)
rem   start.bat <args...>  -> any vox command, e.g. start.bat devices
rem Runs the newest code: this folder once it has the live-quality work (src\vox\audio\
rem conditioning.py), otherwise the DEXAIV-live worktree next to it. Python: this
rem folder's .venv, else the main checkout's.
setlocal
set "CODE=%~dp0"
if not exist "%CODE%src\vox\audio\conditioning.py" set "CODE=%~dp0..\DEXAIV-live\"
if not exist "%CODE%src\vox\audio\conditioning.py" (
    echo Could not find the DEXAIV code with the live-quality fixes.
    pause
    exit /b 1
)
cd /d "%CODE%"
set "PYTHONPATH=%CODE%src"
set "PYTHONUTF8=1"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=C:\Users\Asarii\Desktop\DEXAIV\.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo No Python venv found. Run scripts\setup_env.ps1 first.
    pause
    exit /b 1
)

if "%~1"=="" (
    "%PY%" -m vox.cli gui
) else if /i "%~1"=="run" (
    "%PY%" -m vox.cli run --model Mom --pitch 12 --gate on --gate-threshold -40 --output "Line 1" %2 %3 %4 %5 %6 %7 %8 %9
) else if /i "%~1"=="raw" (
    "%PY%" -m vox.cli run --model Mom --pitch 12 --gate on --gate-threshold -57 --auto-level off --output "Line 1" %2 %3 %4 %5 %6 %7 %8 %9
) else (
    "%PY%" -m vox.cli %*
)
if errorlevel 1 pause
endlocal
