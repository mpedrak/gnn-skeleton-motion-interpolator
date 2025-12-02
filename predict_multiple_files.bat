@echo off
set cfg=%1

if "%cfg%"=="" (
    echo Usage: predict_multiple_files.bat cfg_name
    exit /b 1
)

for /L %%i in (1,1,7) do (
    echo.
    python predict.py %cfg% test_%%i 70
)

for /L %%i in (1,1,7) do (
    echo.
    python predict.py %cfg% test_%%i_pred 170
)

echo Done
