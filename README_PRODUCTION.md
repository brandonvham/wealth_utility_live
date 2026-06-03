# Wealth Utility API Production Scheduling

Production hosting is Railway. Production scheduling is GitHub Actions.

## Deployment Flow

1. Edit files in this `Wealth Utility API` repo.
2. Commit and push to `main`.
3. Railway auto-deploys the Flask API from GitHub.
4. GitHub Actions runs `scheduled_refresh.py` on schedule and calls the deployed `/allocations/refresh` endpoint.

## Scheduled Workflow

Workflow file:

```text
.github/workflows/monthly-allocation-refresh.yml
```

Schedule:

```text
5 23 * * 1-5
```

GitHub cron is UTC. 23:05 UTC is after 5 PM Central in both standard and daylight time. The Python scheduler guard prevents refreshes except on the last NYSE trading day after 5 PM CT.

## Required GitHub Secrets

```text
FMP_KEY
FMP_API_KEY
FRED_API_KEY
```

## Recommended GitHub Variable

```text
WEALTH_UTILITY_API_BASE_URL=https://wealthutilitylive-production.up.railway.app
```

## Manual Refresh

1. Open GitHub.
2. Go to Actions.
3. Select `Monthly Allocation Refresh`.
4. Select `Run workflow`.
5. Set `force_refresh` to `true`.
6. Run the workflow.

## Local Fallback

Use this only if GitHub Actions is unavailable:

```powershell
.\setup_scheduler.ps1
```

That registers `run_wealth_utility.bat` with Windows Task Scheduler on this machine. It does not replace the GitHub Actions production scheduler.
