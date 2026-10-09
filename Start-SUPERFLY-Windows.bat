@echo off
rem SUPERFLY one-click launcher for Windows. Double-click me.
rem First run: sets up a private Python environment, installs the packages,
rem downloads the fly's connectome and its language model, then opens the Lab.
cd /d "%~dp0"
title SUPERFLY
set PY=
where py >nul 2>nul && set PY=py -3
if "%PY%"=="" where python >nul 2>nul && set PY=python
if "%PY%"=="" (
  echo.
  echo  SUPERFLY needs Python 3.10 or newer.
  echo  Get it from https://www.python.org/downloads/  ^(tick "Add python.exe to PATH"^),
  echo  then double-click this file again.
  echo.
  start https://www.python.org/downloads/
  pause
  exit /b 1
)
if not exist .venv\Scripts\python.exe (
  echo Setting up SUPERFLY's Python environment ^(first run only^)...
  %PY% -m venv .venv || goto :fail
)
call .venv\Scripts\activate.bat
if not exist .venv\superfly_installed (
  python -m pip install --upgrade pip
  python -m pip install -r requirements-superfly.txt || goto :fail
  echo ok> .venv\superfly_installed
)
python -m superfly.quickstart %*
pause
exit /b 0
:fail
echo.
echo Something went wrong above. Copy the messages into an issue on GitHub and we will help.
pause
exit /b 1
