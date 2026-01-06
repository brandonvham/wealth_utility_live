# PowerShell script to create Windows Task Scheduler task
# Run this as Administrator to set up the automatic scheduler

$TaskName = "Wealth Utility - Monthly Allocation"
$TaskDescription = "Runs Wealth Utility allocation calculator on the last trading day of each month at 5 PM CT"
$ScriptPath = "C:\Users\BrandonVanLandingham\OneDrive - Perissos Private Wealth Management\1 Perissos Private Wealth Management\5 Python\Wealth Utility\run_wealth_utility.bat"

# Create the action (what to run)
$Action = New-ScheduledTaskAction -Execute $ScriptPath

# Create the trigger (when to run: daily at 5 PM Central Time)
# Note: Windows Task Scheduler doesn't have timezone awareness, so adjust for your local time
# If you're in Central Time, use 5 PM (17:00)
$Trigger = New-ScheduledTaskTrigger -Daily -At "17:00"

# Create the principal (run whether user is logged on or not)
$Principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Highest

# Create settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

# Check if task already exists
$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($ExistingTask) {
    Write-Host "Task '$TaskName' already exists. Updating..." -ForegroundColor Yellow
    Set-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings
    Write-Host "Task updated successfully!" -ForegroundColor Green
} else {
    Write-Host "Creating new task '$TaskName'..." -ForegroundColor Cyan
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Description $TaskDescription `
        -Action $Action `
        -Trigger $Trigger `
        -Principal $Principal `
        -Settings $Settings
    Write-Host "Task created successfully!" -ForegroundColor Green
}

Write-Host ""
Write-Host "Task Details:" -ForegroundColor Cyan
Write-Host "  Name: $TaskName"
Write-Host "  Trigger: Daily at 5:00 PM"
Write-Host "  Action: $ScriptPath"
Write-Host ""
Write-Host "The script will check if today is the last trading day before running." -ForegroundColor Yellow
Write-Host "To test the task immediately, run:" -ForegroundColor Cyan
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host ""
