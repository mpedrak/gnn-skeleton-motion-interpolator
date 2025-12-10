@echo off
set cfg=%1

if "%cfg%"=="" (
    echo Usage: predict_multiple_files.bat cfg_name
    exit /b 1
)

set "files=aiming dance fight ground jumps run walk"

setlocal enabledelayedexpansion

for %%s in (%files%) do (
    set "name=%%~s"
    echo.
    python predict.py %cfg% !name! 70
)

for %%s in (%files%) do (
    set "name=%%~s"
    echo.
    python predict.py %cfg% !name!_pred 270
)

for %%s in (%files%) do (
    set "name=%%~s"
    echo.
    python predict.py %cfg% !name!_pred_pred 470
)

endlocal
echo Done