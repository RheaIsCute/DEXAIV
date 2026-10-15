@echo off
rem DEXAIV launcher. Runs this folder's code (src\) with this folder's .venv, or with the
rem main checkout's venv when this folder has none (a git worktree).
rem   start.bat            -> the GUI
rem   start.bat run        -> live on the CLI: Mom, pitch +12, gate -40, default mic -> VAC
rem   start.bat raw        -> same with auto level off and gate -57 (the old raw-mic setting)
rem   start.bat <args...>  -> any vox command, e.g. start.bat devices
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"
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
