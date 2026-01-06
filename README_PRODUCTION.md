# Wealth Utility - Production Deployment Guide

## Overview

This setup converts the Wealth Utility backtest notebook into an automated production system that runs on the **last trading day of every month at 5 PM Central Time** and outputs the current month's allocation percentages.

## Files Created

1. **wealth_utility_production.py** - Main Python script containing all strategy logic
2. **run_wealth_utility.bat** - Windows batch file to run the script
3. **setup_scheduler.ps1** - PowerShell script to automatically configure Windows Task Scheduler
4. **README_PRODUCTION.md** - This file

## How It Works

### Daily Execution
- The script is scheduled to run **every day at 5 PM Central Time**
- It automatically checks if today is the last trading day of the month
- If yes, it calculates and outputs allocations
- If no, it exits silently

### Last Trading Day Detection
The script uses two methods to determine if today is the last trading day:
1. **Primary**: Checks FMP API for market status
2. **Fallback**: Uses pandas business day logic

### Output
When run on the last trading day, the script outputs:
- Current allocation percentages for each asset
- Total equity vs. non-equity weights
- Saves results to `current_allocation.txt`

## Installation Instructions

### Method 1: Automatic Setup (Recommended)

1. **Open PowerShell as Administrator**
   - Press `Win + X` and select "Windows PowerShell (Admin)"

2. **Navigate to the directory**
   ```powershell
   cd "C:\Users\BrandonVanLandingham\OneDrive - Perissos Private Wealth Management\1 Perissos Private Wealth Management\5 Python\Wealth Utility"
   ```

3. **Allow script execution** (if needed)
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
   ```

4. **Run the setup script**
   ```powershell
   .\setup_scheduler.ps1
   ```

5. **Verify the task was created**
   ```powershell
   Get-ScheduledTask -TaskName "Wealth Utility - Monthly Allocation"
   ```

### Method 2: Manual Setup

1. **Open Task Scheduler**
   - Press `Win + R`, type `taskschd.msc`, press Enter

2. **Create a New Task** (not Basic Task)
   - Click "Create Task..." in the right panel

3. **General Tab**
   - Name: `Wealth Utility - Monthly Allocation`
   - Description: `Runs Wealth Utility allocation calculator on the last trading day of each month at 5 PM CT`
   - Select: "Run whether user is logged on or not"
   - Check: "Run with highest privileges"

4. **Triggers Tab**
   - Click "New..."
   - Begin the task: "On a schedule"
   - Settings: "Daily" at "5:00 PM"
   - Click "OK"

5. **Actions Tab**
   - Click "New..."
   - Action: "Start a program"
   - Program/script: Browse to `run_wealth_utility.bat`
   - Start in: `C:\Users\BrandonVanLandingham\OneDrive - Perissos Private Wealth Management\1 Perissos Private Wealth Management\5 Python\Wealth Utility`
   - Click "OK"

6. **Conditions Tab**
   - Check: "Start only if the following network connection is available: Any connection"
   - Uncheck: "Stop if the computer switches to battery power"

7. **Settings Tab**
   - Check: "Allow task to be run on demand"
   - Check: "Run task as soon as possible after a scheduled start is missed"
   - If the task fails, restart every: "10 minutes" for up to "3" attempts
   - Execution time limit: "2 hours"

8. **Click "OK"** to save the task

## Testing

### Test the Script Manually
To test without waiting for the scheduled time:

```powershell
cd "C:\Users\BrandonVanLandingham\OneDrive - Perissos Private Wealth Management\1 Perissos Private Wealth Management\5 Python\Wealth Utility"

# Set environment variable to force run regardless of day/time
$env:FORCE_RUN = "true"
python wealth_utility_production.py
```

Or simply run the batch file:
```batch
run_wealth_utility.bat
```

### Test the Scheduled Task
```powershell
Start-ScheduledTask -TaskName "Wealth Utility - Monthly Allocation"
```

### View Task History
1. Open Task Scheduler
2. Find "Wealth Utility - Monthly Allocation" in the task list
3. Click the "History" tab at the bottom

## Configuration

### Modify Strategy Parameters
Edit the parameters at the top of `wealth_utility_production.py`:

```python
# Equity tickers
EQUITY_TICKER = ["^GSPC","^IXIC","MSCIWORLD","GCUSD"]
NON_EQUITY_TICKER = "ZNUSD"

# Sleeve method
EQUITY_SLEEVE_METHOD = "max_sharpe"

# Dials & bounds
VALUE_DIAL_FRAC = 25
MOM_BUMP_FRAC = 75
BAND_MODE = "absolute"
BAND_ABS = 1.0
```

### Change Execution Time
To run at a different time:
1. Open Task Scheduler
2. Find the task and double-click it
3. Go to the "Triggers" tab
4. Edit the trigger and change the time
5. Click "OK" to save

## Output Files

When the script runs on the last trading day, it creates:

### current_allocation.txt
```
================================================================================
WEALTH UTILITY - ALLOCATION FOR 2026-01-31
Generated: 2026-01-31 05:00 PM
================================================================================

EQUITY SLEEVE:
--------------------------------------------------------------------------------
  ^GSPC                 35.50%
  ^IXIC                 20.25%
  MSCIWORLD             15.75%
  GCUSD                  8.50%

NON-EQUITY:
--------------------------------------------------------------------------------
  ZNUSD                 20.00%

================================================================================
SUMMARY:
--------------------------------------------------------------------------------
  Total Equity:         80.00%
  Total Non-Equity:     20.00%
  Total:               100.00%
================================================================================
```

## Troubleshooting

### Script doesn't run
- Check Task Scheduler history for error messages
- Verify Python is in your system PATH
- Ensure all dependencies are installed: `pip install -r requirements.txt`

### API errors
- Verify your FMP_KEY and FRED_API_KEY environment variables or hardcoded values
- Check your internet connection
- Ensure API rate limits aren't exceeded

### Wrong timezone
- The script uses Central Time (America/Chicago) via pytz
- Adjust `central = pytz.timezone('America/Chicago')` if needed

### Force run for testing
Set the environment variable before running:
```powershell
$env:FORCE_RUN = "true"
python wealth_utility_production.py
```

Or in batch/cmd:
```batch
set FORCE_RUN=true
python wealth_utility_production.py
```

## Dependencies

Ensure these Python packages are installed:
```
numpy
pandas
scipy
requests
urllib3
openpyxl
pytz
scikit-learn
```

Install all at once:
```powershell
pip install numpy pandas scipy requests urllib3 openpyxl pytz scikit-learn
```

## Maintenance

### Check Task Status
```powershell
Get-ScheduledTask -TaskName "Wealth Utility - Monthly Allocation" | Select-Object TaskName, State, LastRunTime, NextRunTime
```

### Disable Task Temporarily
```powershell
Disable-ScheduledTask -TaskName "Wealth Utility - Monthly Allocation"
```

### Enable Task
```powershell
Enable-ScheduledTask -TaskName "Wealth Utility - Monthly Allocation"
```

### Remove Task
```powershell
Unregister-ScheduledTask -TaskName "Wealth Utility - Monthly Allocation" -Confirm:$false
```

## Security Notes

- API keys are currently hardcoded in the script
- Consider using environment variables for production:
  ```powershell
  [System.Environment]::SetEnvironmentVariable('FMP_KEY', 'your_key_here', 'User')
  [System.Environment]::SetEnvironmentVariable('FRED_API_KEY', 'your_key_here', 'User')
  ```

## Support

For issues or questions:
1. Check the Task Scheduler history
2. Review `current_allocation.txt` for the last successful run
3. Run manually with `FORCE_RUN=true` to see detailed error messages
