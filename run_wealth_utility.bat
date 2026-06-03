@echo off
REM Wealth Utility Production Runner
REM Local fallback runner. Production scheduling is handled by GitHub Actions.
REM Runs daily at 5 PM CT via Windows Task Scheduler when configured locally.
REM Only executes on the last trading day of the month

echo ========================================
echo Wealth Utility Production Scheduler
echo ========================================
echo Run Time: %date% %time%
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Run the Python script
python wealth_utility_production.py

REM Check if the script ran successfully
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Script completed successfully
    echo ========================================
) else (
    echo.
    echo ========================================
    echo Script failed with error code %ERRORLEVEL%
    echo ========================================
)

REM Keep window open for 10 seconds if running manually
timeout /t 10

exit /b %ERRORLEVEL%
