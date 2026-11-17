@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup_demo.cmd first.
  pause
  exit /b 1
)
cd /d "%~dp0tools\classroom_engagement\backend"
"%~dp0.venv\Scripts\python.exe" main.py %*
set "eng_exit=%errorlevel%"
if not "%eng_exit%"=="0" pause
exit /b %eng_exit%
