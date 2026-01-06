@echo off
echo ========================================
echo Quick Git Push
echo ========================================
echo.

cd /d "%~dp0"

echo Current changes:
git status
echo.

set /p message="Enter commit message: "

echo.
echo Adding files...
git add .

echo Committing...
git commit -m "%message%"

echo Pushing to GitHub...
git push

echo.
echo ========================================
echo Done! Check Railway for deployment.
echo ========================================
pause
