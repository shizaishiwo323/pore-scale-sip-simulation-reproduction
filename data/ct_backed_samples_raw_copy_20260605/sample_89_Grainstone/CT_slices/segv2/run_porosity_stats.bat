@echo off
setlocal
cd /d "%~dp0"

set "PROJECT=%~1"
if "%PROJECT%"=="" (
    echo Paste an ilastik .ilp path, or press Enter to use:
    echo %~dp04-class.ilp
    echo.
    set /p "PROJECT=Project path: "
)
if "%PROJECT%"=="" set "PROJECT=%~dp04-class.ilp"
set "PROJECT=%PROJECT:"=%"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_ilastik_porosity.ps1" -SkipExport -ProjectFile "%PROJECT%"
echo.
pause
