@echo off
REM Batch file to run Kaggle leaderboard checker
REM Usage: check_leaderboard.bat [competition-name]

setlocal

REM Set default competition name
set COMPETITION_NAME=brain-to-text-25

REM Use command line argument if provided
if not "%1"=="" set COMPETITION_NAME=%1

echo Running Kaggle Leaderboard Checker for competition: %COMPETITION_NAME%

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0

REM Run the Python script
python "%SCRIPT_DIR%check_kaggle_leaderboard.py" %COMPETITION_NAME%

REM Check exit code
if %ERRORLEVEL% equ 0 (
    echo.
    echo Leaderboard check completed successfully!
) else (
    echo.
    echo ERROR: Leaderboard check failed with exit code: %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

pause
