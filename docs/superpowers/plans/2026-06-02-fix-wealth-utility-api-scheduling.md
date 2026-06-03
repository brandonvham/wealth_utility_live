# Wealth Utility API Scheduling Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Wealth Utility API` the clear production source of truth, add GitHub Actions as the primary scheduled monthly refresh mechanism, and remove stale references that still point scheduled runs at the research folder.

**Architecture:** Keep Railway as the web API host and GitHub as the deployment trigger. Add a small, testable scheduler/refresh layer in the API repo: one module decides whether today is the last NYSE trading day after 5 PM Central, one script calls the live Railway `/allocations/refresh` endpoint, and one GitHub Actions workflow runs the script on schedule or manually. Windows Task Scheduler remains only a local fallback.

**Tech Stack:** Python 3.8-compatible syntax, Flask API on Railway, GitHub Actions, `requests`, `pandas`, `pandas-market-calendars`, `unittest`.

---

## Current Findings

The local `Wealth Utility API` repo is on `main`, and local `HEAD` matches `origin/main`.

There is currently no `.github` directory and no tracked `.github/workflows/*` file in `Wealth Utility API`. Local git history also shows no workflow history under `.github`.

The production docs currently describe Railway auto-deploy and Windows Task Scheduler. `README.md`, `README_PRODUCTION.md`, `START_HERE.md`, `run_wealth_utility.bat`, and `setup_scheduler.ps1` still describe or execute local scheduled runs. Worse, `Wealth Utility API/run_wealth_utility.bat` and `Wealth Utility API/setup_scheduler.ps1` point at a hard-coded `5 Python\Wealth Utility` path instead of using the API repo directory.

Conclusion: the API repo does not currently use GitHub Actions for scheduled runs. The fix is to add GitHub Actions scheduling and make the docs/scripts consistent with that model.

## File Structure

- Create: `Wealth Utility API/scheduling.py`
  - Owns all time-window and last-trading-day scheduling decisions.
  - Uses the NYSE calendar instead of the current FMP plus weekday heuristic.

- Create: `Wealth Utility API/scheduled_refresh.py`
  - Entry point for GitHub Actions.
  - Checks `should_run_now()` unless `--force` is passed.
  - Calls the live Railway API's `/allocations/refresh` endpoint.
  - Writes `current_allocation.json` for workflow artifacts.

- Modify: `Wealth Utility API/.gitignore`
  - Keeps `current_allocation.json` as a workflow/local verification artifact, not a tracked source file.

- Create: `Wealth Utility API/tests/test_scheduling.py`
  - Unit tests for cutoff time, normal last trading day, non-last trading day, and NYSE holiday handling.

- Create: `Wealth Utility API/tests/test_scheduled_refresh.py`
  - Unit tests for skip behavior, forced refresh, HTTP success, and HTTP failure.

- Create: `Wealth Utility API/.github/workflows/monthly-allocation-refresh.yml`
  - Runs weekdays after market close.
  - Supports manual `workflow_dispatch` with a force option.
  - Installs dependencies, runs tests, then calls `scheduled_refresh.py`.

- Modify: `Wealth Utility API/wealth_utility_production.py`
  - Keep compatibility functions `is_last_trading_day_of_month()` and `should_run_now()`.
  - Delegate their logic to `scheduling.py`.

- Modify: `Wealth Utility API/requirements_api.txt`
  - Add `pandas-market-calendars`.

- Modify: `Wealth Utility API/run_wealth_utility.bat`
  - Make it run from its own directory with `%~dp0`, not a hard-coded `Wealth Utility` path.

- Modify: `Wealth Utility API/setup_scheduler.ps1`
  - Make it use `$PSScriptRoot`.
  - Label it as a local fallback, not the production scheduler.

- Modify docs:
  - `Wealth Utility API/START_HERE.md`
  - `Wealth Utility API/README.md`
  - `Wealth Utility API/README_PRODUCTION.md`
  - `Wealth Utility API/GITHUB_SETUP.md`
  - `Wealth Utility API/DEPLOYMENT_SUMMARY.md`
  - `Wealth Utility API/LOVABLE_INTEGRATION_GUIDE.md`
  - `Wealth Utility API/test_api.py`

---

### Task 1: Add Testable Scheduling Logic

**Files:**
- Create: `Wealth Utility API/scheduling.py`
- Create: `Wealth Utility API/tests/test_scheduling.py`
- Modify: `Wealth Utility API/requirements_api.txt`

- [ ] **Step 1: Add the failing scheduling tests**

Create `tests/test_scheduling.py`:

```python
from datetime import datetime
import unittest

import pytz

from scheduling import is_after_cutoff, is_last_nyse_trading_day, should_run_now


CENTRAL = pytz.timezone("America/Chicago")


def ct(year, month, day, hour, minute=0):
    return CENTRAL.localize(datetime(year, month, day, hour, minute))


class SchedulingTests(unittest.TestCase):
    def test_is_after_cutoff_false_before_5pm_central(self):
        self.assertFalse(is_after_cutoff(ct(2026, 6, 30, 16, 59)))

    def test_is_after_cutoff_true_at_5pm_central(self):
        self.assertTrue(is_after_cutoff(ct(2026, 6, 30, 17, 0)))

    def test_last_nyse_trading_day_true_for_normal_month_end(self):
        self.assertTrue(is_last_nyse_trading_day(ct(2026, 6, 30, 17, 0)))

    def test_last_nyse_trading_day_false_for_prior_business_day(self):
        self.assertFalse(is_last_nyse_trading_day(ct(2026, 6, 29, 17, 0)))

    def test_last_nyse_trading_day_handles_good_friday_month_end(self):
        # NYSE was closed Friday 2024-03-29 for Good Friday, so Thursday
        # 2024-03-28 was the final NYSE trading day of March 2024.
        self.assertTrue(is_last_nyse_trading_day(ct(2024, 3, 28, 17, 0)))

    def test_should_run_now_requires_last_trading_day_and_cutoff(self):
        self.assertTrue(should_run_now(ct(2026, 6, 30, 17, 1)))
        self.assertFalse(should_run_now(ct(2026, 6, 30, 16, 59)))
        self.assertFalse(should_run_now(ct(2026, 6, 29, 17, 1)))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```powershell
cd "Wealth Utility API"
python -m unittest tests.test_scheduling -v
```

Expected: FAIL because `scheduling.py` does not exist.

- [ ] **Step 3: Add the NYSE-calendar scheduling module**

Create `scheduling.py`:

```python
from datetime import datetime, time
from typing import Optional

import pandas as pd
import pandas_market_calendars as mcal
import pytz


CENTRAL_TZ = pytz.timezone("America/Chicago")
NYSE_CALENDAR_NAME = "NYSE"
RUN_CUTOFF_CT = time(17, 0)


def to_central(dt: Optional[datetime] = None) -> datetime:
    if dt is None:
        return datetime.now(CENTRAL_TZ)
    if dt.tzinfo is None:
        return CENTRAL_TZ.localize(dt)
    return dt.astimezone(CENTRAL_TZ)


def is_after_cutoff(dt: Optional[datetime] = None, cutoff: time = RUN_CUTOFF_CT) -> bool:
    dt_ct = to_central(dt)
    return dt_ct.time() >= cutoff


def is_last_nyse_trading_day(dt: Optional[datetime] = None) -> bool:
    dt_ct = to_central(dt)
    today = pd.Timestamp(dt_ct.date())
    month_start = today.replace(day=1)
    month_end = today + pd.offsets.MonthEnd(0)

    calendar = mcal.get_calendar(NYSE_CALENDAR_NAME)
    schedule = calendar.schedule(start_date=month_start.date(), end_date=month_end.date())
    if schedule.empty:
        return False

    trading_dates = [idx.date() for idx in schedule.index]
    return dt_ct.date() == trading_dates[-1]


def should_run_now(dt: Optional[datetime] = None) -> bool:
    dt_ct = to_central(dt)
    return is_after_cutoff(dt_ct) and is_last_nyse_trading_day(dt_ct)
```

- [ ] **Step 4: Add the dependency**

Append this line to `requirements_api.txt`:

```text
pandas-market-calendars>=4.4.0
```

- [ ] **Step 5: Run the scheduling tests**

Run:

```powershell
cd "Wealth Utility API"
python -m pip install -r requirements_api.txt
python -m unittest tests.test_scheduling -v
```

Expected: PASS for all six tests.

- [ ] **Step 6: Commit**

Run:

```powershell
cd "Wealth Utility API"
git add scheduling.py tests/test_scheduling.py requirements_api.txt
git commit -m "feat: add NYSE calendar scheduling guard"
```

---

### Task 2: Route Existing Production Scheduler Functions Through `scheduling.py`

**Files:**
- Modify: `Wealth Utility API/wealth_utility_production.py`
- Create: `Wealth Utility API/tests/test_production_scheduler_compat.py`

- [ ] **Step 1: Add compatibility tests**

Create `tests/test_production_scheduler_compat.py`:

```python
import os
import unittest
from unittest.mock import patch

# `wealth_utility_production.py` validates these at import time. These dummy
# values keep the compatibility test isolated from local or CI secrets.
os.environ.setdefault("FMP_KEY", "test-fmp-key")
os.environ.setdefault("FRED_API_KEY", "test-fred-key")

import wealth_utility_production as production


class ProductionSchedulerCompatTests(unittest.TestCase):
    def test_last_trading_day_delegates_to_scheduling_module(self):
        with patch.object(production.scheduling, "is_last_nyse_trading_day", return_value=True) as guard:
            self.assertTrue(production.is_last_trading_day_of_month())

        guard.assert_called_once_with()

    def test_should_run_now_delegates_and_returns_true(self):
        with patch.object(production.scheduling, "should_run_now", return_value=True) as guard:
            self.assertTrue(production.should_run_now())

        guard.assert_called_once_with()

    def test_should_run_now_delegates_and_returns_false(self):
        with patch.object(production.scheduling, "should_run_now", return_value=False) as guard:
            self.assertFalse(production.should_run_now())

        guard.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the compatibility tests**

Run:

```powershell
cd "Wealth Utility API"
python -m unittest tests.test_production_scheduler_compat -v
```

Expected: FAIL before the change because `wealth_utility_production` does not yet expose the `scheduling` module; PASS after the refactor. This protects the current public function names and proves they delegate to the new scheduler.

- [ ] **Step 3: Modify `wealth_utility_production.py` imports**

Add near the existing imports:

```python
import scheduling
```

- [ ] **Step 4: Replace only the bodies of the existing scheduler functions**

Keep the function names and signatures, but replace their bodies:

```python
def is_last_trading_day_of_month() -> bool:
    """
    Determine if today is the last NYSE trading day of the current month.
    Kept for backward compatibility with local scheduler scripts.
    """
    return scheduling.is_last_nyse_trading_day()


def should_run_now() -> bool:
    """
    Check if the script should run right now.
    Must be 5 PM CT or later, and must be the last NYSE trading day.
    """
    if not scheduling.should_run_now():
        print("Current time is before 5:00 PM CT or today is not the last NYSE trading day. Skipping.")
        return False
    return True
```

- [ ] **Step 5: Run tests**

Run:

```powershell
cd "Wealth Utility API"
python -m unittest tests.test_scheduling tests.test_production_scheduler_compat -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
cd "Wealth Utility API"
git add wealth_utility_production.py tests/test_production_scheduler_compat.py
git commit -m "refactor: reuse calendar scheduler in production runner"
```

---

### Task 3: Add the GitHub Actions Refresh Entry Point

**Files:**
- Create: `Wealth Utility API/scheduled_refresh.py`
- Create: `Wealth Utility API/tests/test_scheduled_refresh.py`
- Modify: `Wealth Utility API/.gitignore`

- [ ] **Step 1: Add failing tests for the refresh entry point**

Create `tests/test_scheduled_refresh.py`:

```python
import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import scheduled_refresh


class ScheduledRefreshTests(unittest.TestCase):
    def test_main_skips_when_scheduler_says_not_to_run(self):
        with patch("scheduled_refresh.should_run_now", return_value=False), \
             patch("scheduled_refresh.refresh_allocations") as refresh:
            exit_code = scheduled_refresh.main([])
        self.assertEqual(exit_code, 0)
        refresh.assert_not_called()

    def test_main_force_refresh_bypasses_schedule_guard(self):
        with patch("scheduled_refresh.should_run_now", return_value=False), \
             patch("scheduled_refresh.refresh_allocations", return_value={"success": True}) as refresh:
            exit_code = scheduled_refresh.main(["--force"])
        self.assertEqual(exit_code, 0)
        refresh.assert_called_once()

    def test_refresh_allocations_posts_to_refresh_endpoint_and_writes_artifact(self):
        response = Mock()
        response.json.return_value = {"success": True, "allocation_date": "2026-06-30"}
        response.raise_for_status.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir, \
             patch("scheduled_refresh.requests.post", return_value=response):
            artifact_path = os.path.join(tmpdir, "current_allocation.json")
            payload = scheduled_refresh.refresh_allocations(
                base_url="https://wealthutilitylive-production.up.railway.app",
                artifact_path=artifact_path,
            )

            self.assertEqual(payload["allocation_date"], "2026-06-30")
            with open(artifact_path, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
            self.assertEqual(saved["allocation_date"], "2026-06-30")

    def test_refresh_allocations_fails_when_api_reports_failure(self):
        response = Mock()
        response.json.return_value = {"success": False, "error": "bad data"}
        response.raise_for_status.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir, \
             patch("scheduled_refresh.requests.post", return_value=response), \
             self.assertRaises(RuntimeError):
            scheduled_refresh.refresh_allocations(
                base_url="https://wealthutilitylive-production.up.railway.app",
                artifact_path=os.path.join(tmpdir, "current_allocation.json"),
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
cd "Wealth Utility API"
python -m unittest tests.test_scheduled_refresh -v
```

Expected: FAIL because `scheduled_refresh.py` does not exist.

- [ ] **Step 3: Add `scheduled_refresh.py`**

Create `scheduled_refresh.py`:

```python
import argparse
import json
import os
import sys
from typing import List, Optional

import requests

from scheduling import should_run_now


DEFAULT_API_BASE_URL = "https://wealthutilitylive-production.up.railway.app"
DEFAULT_ARTIFACT_PATH = "current_allocation.json"


def refresh_allocations(
    base_url: str,
    artifact_path: str = DEFAULT_ARTIFACT_PATH,
    timeout_seconds: int = 300,
) -> dict:
    refresh_url = base_url.rstrip("/") + "/allocations/refresh"
    response = requests.post(refresh_url, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()

    if not payload.get("success", False):
        raise RuntimeError(f"Allocation refresh failed: {payload}")

    with open(artifact_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"Refreshed allocations at {refresh_url}")
    print(f"Saved refresh artifact to {artifact_path}")
    return payload


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh Wealth Utility API allocations on schedule.")
    parser.add_argument("--force", action="store_true", help="Refresh even when the schedule guard would skip.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("WEALTH_UTILITY_API_BASE_URL", DEFAULT_API_BASE_URL),
        help="Base URL for the deployed Wealth Utility API.",
    )
    parser.add_argument(
        "--artifact-path",
        default=os.getenv("WEALTH_UTILITY_ARTIFACT_PATH", DEFAULT_ARTIFACT_PATH),
        help="Path where the JSON response should be written.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if not args.force and not should_run_now():
        print("Skipping refresh: not after 5 PM CT on the last NYSE trading day.")
        return 0

    try:
        refresh_allocations(base_url=args.base_url, artifact_path=args.artifact_path)
    except Exception as exc:
        print(f"Refresh failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests**

Run:

```powershell
cd "Wealth Utility API"
python -m unittest tests.test_scheduled_refresh -v
```

Expected: PASS.

- [ ] **Step 5: Ignore the refresh artifact**

Append this line to `.gitignore` near `current_allocation.txt`:

```text
current_allocation.json
```

- [ ] **Step 6: Commit**

Run:

```powershell
cd "Wealth Utility API"
git add scheduled_refresh.py tests/test_scheduled_refresh.py .gitignore
git commit -m "feat: add scheduled allocation refresh runner"
```

---

### Task 4: Add GitHub Actions Scheduled Workflow

**Files:**
- Create: `Wealth Utility API/.github/workflows/monthly-allocation-refresh.yml`

- [ ] **Step 1: Create the workflow file**

Create `.github/workflows/monthly-allocation-refresh.yml`:

```yaml
name: Monthly Allocation Refresh

on:
  schedule:
    # GitHub cron is UTC. 23:05 UTC is 5:05 PM CT during standard time
    # and 6:05 PM CT during daylight time, which keeps the run after market close.
    - cron: "5 23 * * 1-5"
  workflow_dispatch:
    inputs:
      force_refresh:
        description: "Bypass the last-trading-day schedule guard"
        required: true
        default: "false"
        type: choice
        options:
          - "false"
          - "true"

jobs:
  refresh-allocations:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    env:
      FMP_KEY: ${{ secrets.FMP_KEY }}
      FMP_API_KEY: ${{ secrets.FMP_API_KEY }}
      FRED_API_KEY: ${{ secrets.FRED_API_KEY }}
      WEALTH_UTILITY_API_BASE_URL: ${{ vars.WEALTH_UTILITY_API_BASE_URL || 'https://wealthutilitylive-production.up.railway.app' }}
    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: "pip"

      - name: Install dependencies
        run: python -m pip install -r requirements_api.txt

      - name: Run tests
        run: python -m unittest discover -s tests -v

      - name: Refresh allocations
        shell: bash
        run: |
          if [ "${{ github.event.inputs.force_refresh }}" = "true" ]; then
            python scheduled_refresh.py --force
          else
            python scheduled_refresh.py
          fi

      - name: Upload allocation artifact
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: current-allocation
          path: current_allocation.json
          if-no-files-found: ignore
```

- [ ] **Step 2: Validate workflow syntax locally**

Run:

```powershell
cd "Wealth Utility API"
python -c "from pathlib import Path; text = Path('.github/workflows/monthly-allocation-refresh.yml').read_text(encoding='utf-8'); assert 'workflow_dispatch' in text; assert 'scheduled_refresh.py' in text; assert 'unittest discover -s tests' in text; assert '5 23 * * 1-5' in text; print('workflow file sanity checks passed')"
```

Expected: `workflow file sanity checks passed`.

- [ ] **Step 3: Commit**

Run:

```powershell
cd "Wealth Utility API"
git add .github/workflows/monthly-allocation-refresh.yml
git commit -m "ci: add monthly allocation refresh workflow"
```

---

### Task 5: Fix Stale Local Scheduler Scripts

**Files:**
- Modify: `Wealth Utility API/run_wealth_utility.bat`
- Modify: `Wealth Utility API/setup_scheduler.ps1`

- [ ] **Step 1: Update the batch runner to use its own directory**

Replace the hard-coded `cd /d` line in `run_wealth_utility.bat`:

```bat
cd /d "%~dp0"
```

Keep this execution line:

```bat
python wealth_utility_production.py
```

- [ ] **Step 2: Update the PowerShell scheduler path**

Replace the hard-coded `$ScriptPath` line in `setup_scheduler.ps1`:

```powershell
$ScriptPath = Join-Path $PSScriptRoot "run_wealth_utility.bat"
```

Add this comment near the top:

```powershell
# Local fallback only. Production scheduling is handled by GitHub Actions.
```

- [ ] **Step 3: Sanity-check the paths**

Run:

```powershell
cd "Wealth Utility API"
Select-String -LiteralPath run_wealth_utility.bat, setup_scheduler.ps1 -Pattern "5 Python\\Wealth Utility"
```

Expected: no matches.

- [ ] **Step 4: Commit**

Run:

```powershell
cd "Wealth Utility API"
git add run_wealth_utility.bat setup_scheduler.ps1
git commit -m "fix: make local scheduler scripts repo-relative"
```

---

### Task 6: Update Documentation So Scheduling Is Unambiguous

**Files:**
- Modify: `Wealth Utility API/START_HERE.md`
- Modify: `Wealth Utility API/README.md`
- Modify: `Wealth Utility API/README_PRODUCTION.md`
- Modify: `Wealth Utility API/GITHUB_SETUP.md`
- Modify: `Wealth Utility API/DEPLOYMENT_SUMMARY.md`
- Modify: `Wealth Utility API/LOVABLE_INTEGRATION_GUIDE.md`
- Modify: `Wealth Utility API/test_api.py`

- [ ] **Step 1: Update the production summary**

In `START_HERE.md`, replace the "Schedule Monthly Runs" section with:

```markdown
### Schedule Monthly Runs

Production scheduled runs are handled by GitHub Actions:

- Workflow: `.github/workflows/monthly-allocation-refresh.yml`
- Schedule: weekdays at 23:05 UTC, with an internal guard for last NYSE trading day after 5 PM CT
- Manual run: GitHub Actions -> Monthly Allocation Refresh -> Run workflow -> `force_refresh=true`

`setup_scheduler.ps1` is a local Windows fallback only.
```

- [ ] **Step 2: Update README scheduling section**

In `README.md`, replace the Windows Task Scheduler-first text with:

```markdown
## Scheduled Production Runs

The production monthly refresh is handled by GitHub Actions. The workflow runs on weekdays after market close and only refreshes allocations when `scheduling.should_run_now()` confirms it is after 5 PM Central on the last NYSE trading day of the month.

Required GitHub repository secrets:

- `FMP_KEY`
- `FMP_API_KEY`
- `FRED_API_KEY`

Recommended GitHub repository variable:

- `WEALTH_UTILITY_API_BASE_URL=https://wealthutilitylive-production.up.railway.app`

Windows Task Scheduler scripts remain in the repo as a local fallback only.
```

- [ ] **Step 3: Replace `README_PRODUCTION.md` with the GitHub Actions operating guide**

Use this document body:

````markdown
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
````

- [ ] **Step 4: Update `GITHUB_SETUP.md` for the API repo and Actions scheduler**

Replace every command path that points at:

```text
C:\Users\BrandonVanLandingham\OneDrive - Perissos Private Wealth Management\1 Perissos Private Wealth Management\5 Python\Wealth Utility
```

with:

```text
C:\Users\BrandonVanLandingham\OneDrive - Perissos Private Wealth Management\1 Perissos Private Wealth Management\5 Python\01_Active_Strategies\Wealth Utility API
```

Replace the repository-name instruction with:

```markdown
3. Repository name: `wealth_utility_live`
```

Replace the remote example with:

```bash
git remote add origin https://github.com/YOUR_USERNAME/wealth_utility_live.git
```

Replace this initial-commit bullet:

```markdown
- Automated production scheduler
```

with:

```markdown
- GitHub Actions production scheduler
```

Add this section after the Railway deployment steps:

```markdown
## Production Scheduler

GitHub Actions is the production scheduler for monthly allocation refreshes.

- Workflow: `.github/workflows/monthly-allocation-refresh.yml`
- Schedule: weekdays at 23:05 UTC, guarded by `scheduling.should_run_now()`
- Manual run: Actions -> Monthly Allocation Refresh -> Run workflow -> `force_refresh=true`

Required GitHub repository secrets:

- `FMP_KEY`
- `FMP_API_KEY`
- `FRED_API_KEY`

Recommended GitHub repository variable:

- `WEALTH_UTILITY_API_BASE_URL=https://wealthutilitylive-production.up.railway.app`
```

- [ ] **Step 5: Update `DEPLOYMENT_SUMMARY.md` scheduler status**

Add this bullet under `### GitHub Repository`:

```markdown
- GitHub Actions workflow `.github/workflows/monthly-allocation-refresh.yml` handles the production monthly allocation refresh
```

Add this block after the `wealth_utility_api.py` architecture description:

```markdown
**`scheduled_refresh.py`** (Scheduled Refresh Runner)
- Called by GitHub Actions, not Railway
- Checks the NYSE last-trading-day guard unless `--force` is used
- Calls the deployed Railway `/allocations/refresh` endpoint
- Saves `current_allocation.json` as a workflow artifact

**`.github/workflows/monthly-allocation-refresh.yml`** (Production Scheduler)
- Runs weekdays after market close
- Installs API dependencies and runs the test suite
- Supports manual `force_refresh=true` runs from GitHub Actions
```

Add this block after the current data-flow diagram:

````markdown
Scheduled refresh flow:

```text
GitHub Actions
    ->
scheduled_refresh.py
    ->
Railway /allocations/refresh
    ->
current_allocation.json workflow artifact
```
````

Replace this documentation reference row:

```markdown
| Automate monthly runs | `README_PRODUCTION.md` |
```

with:

```markdown
| Operate production scheduling | `README_PRODUCTION.md` |
```

- [ ] **Step 6: Update `LOVABLE_INTEGRATION_GUIDE.md` path and refresh guidance**

Replace every command path that points at:

```text
C:\Users\BrandonVanLandingham\OneDrive - Perissos Private Wealth Management\1 Perissos Private Wealth Management\5 Python\Wealth Utility
```

with:

```text
C:\Users\BrandonVanLandingham\OneDrive - Perissos Private Wealth Management\1 Perissos Private Wealth Management\5 Python\01_Active_Strategies\Wealth Utility API
```

Replace the entire `## Automated Updates` section with:

```markdown
## Automated Updates

Production allocation refreshes are handled by GitHub Actions in this repository.

- Workflow: `.github/workflows/monthly-allocation-refresh.yml`
- Schedule: weekdays at 23:05 UTC
- Guard: `scheduling.should_run_now()` only allows refreshes after 5 PM Central on the last NYSE trading day
- Manual refresh: GitHub Actions -> Monthly Allocation Refresh -> Run workflow -> `force_refresh=true`

Lovable clients should read allocations from the API:

- `GET https://wealthutilitylive-production.up.railway.app/allocations`
- `GET https://wealthutilitylive-production.up.railway.app/allocations?profile=moderate`
```

- [ ] **Step 7: Update stale command in `test_api.py`**

Replace the printed `cd` path with:

```python
print("  cd \"C:\\Users\\BrandonVanLandingham\\OneDrive - Perissos Private Wealth Management\\1 Perissos Private Wealth Management\\5 Python\\01_Active_Strategies\\Wealth Utility API\"")
```

- [ ] **Step 8: Search for stale scheduler claims and stale research-folder commands**

Run:

```powershell
cd "Wealth Utility API"
Select-String -Path *.md, *.py, *.bat, *.ps1 -Pattern "Windows Task Scheduler|Task Scheduler|setup_scheduler|GitHub Actions|scheduled runs|5 Python\\Wealth Utility|Wealth Utility\\run_wealth_utility" -CaseSensitive:$false
```

Expected: references to Windows Task Scheduler should clearly say "local fallback only"; production scheduling references should point to GitHub Actions; no executable command should point at `5 Python\Wealth Utility`.

- [ ] **Step 9: Commit**

Run:

```powershell
cd "Wealth Utility API"
git add START_HERE.md README.md README_PRODUCTION.md GITHUB_SETUP.md DEPLOYMENT_SUMMARY.md LOVABLE_INTEGRATION_GUIDE.md test_api.py
git commit -m "docs: document GitHub Actions production scheduler"
```

---

### Task 7: Verify End-to-End Locally

**Files:**
- No new files.

- [ ] **Step 1: Run unit tests**

Run:

```powershell
cd "Wealth Utility API"
python -m unittest discover -s tests -v
```

Expected: all scheduler and refresh tests pass.

- [ ] **Step 2: Run the standalone calculation smoke test**

Run:

```powershell
cd "Wealth Utility API"
python test_api_standalone.py
```

Expected: allocation percentages print without exceptions.

- [ ] **Step 3: Test the refresh runner against the live Railway API**

Run:

```powershell
cd "Wealth Utility API"
python scheduled_refresh.py --force --base-url "https://wealthutilitylive-production.up.railway.app"
```

Expected:

```text
Refreshed allocations at https://wealthutilitylive-production.up.railway.app/allocations/refresh
Saved refresh artifact to current_allocation.json
```

- [ ] **Step 4: Confirm artifact contents**

Run:

```powershell
cd "Wealth Utility API"
python -c "import json; from pathlib import Path; payload = json.loads(Path('current_allocation.json').read_text(encoding='utf-8')); assert payload['success'] is True; assert 'allocation_date' in payload; assert 'profiles' in payload or 'allocations' in payload; print(payload['allocation_date'])"
```

Expected: prints the allocation date.

- [ ] **Step 5: Confirm the refresh artifact remains ignored**

Run:

```powershell
cd "Wealth Utility API"
git status --short
```

Expected: `current_allocation.json` does not appear because Task 3 added it to `.gitignore`; only intentional source, workflow, test, and documentation files are modified.

---

### Task 8: Push and Verify GitHub Actions

**Files:**
- No new files.

- [ ] **Step 1: Push the branch**

Run:

```powershell
cd "Wealth Utility API"
git push origin main
```

Expected: push succeeds and Railway starts an auto-deploy.

- [ ] **Step 2: Configure GitHub secrets and variable**

In GitHub repo `brandonvham/wealth_utility_live`, configure repository secrets:

```text
FMP_KEY
FMP_API_KEY
FRED_API_KEY
```

Configure repository variable:

```text
WEALTH_UTILITY_API_BASE_URL=https://wealthutilitylive-production.up.railway.app
```

- [ ] **Step 3: Manually run the workflow**

In GitHub:

```text
Actions -> Monthly Allocation Refresh -> Run workflow -> force_refresh=true
```

Expected: the workflow passes, uploads `current-allocation`, and the artifact contains `current_allocation.json`.

- [ ] **Step 4: Verify live API health**

Run:

```powershell
curl.exe https://wealthutilitylive-production.up.railway.app/health
curl.exe https://wealthutilitylive-production.up.railway.app/allocations
```

Expected: `/health` returns healthy status and `/allocations` returns a successful allocation payload.

---

## Self-Review

Spec coverage:

- Explains whether GitHub Actions currently exists: yes, under Current Findings.
- Adds GitHub Actions scheduling: Task 4.
- Runs all scheduler, refresh, and production compatibility tests in GitHub Actions: Task 4.
- Keeps Railway as API host: architecture and Task 3/4.
- Fixes stale `Wealth Utility` hard-coded paths: Task 5 and exact documentation replacements in Task 6.
- Updates docs so future edits happen in the API repo: Task 6.
- Adds tests for scheduling and refresh behavior: Tasks 1 through 3.
- Keeps `current_allocation.json` as an ignored workflow/local artifact: Task 3 and Task 7.
- Uses Python 3.8-compatible type hints instead of `datetime | None` or `list[str] | None`: Tasks 1 and 3.

Placeholder scan:

- No task says "TBD", "TODO", "add appropriate error handling", or "write tests" without concrete tests.
- Every code-changing task includes code or exact replacement text.

Type consistency:

- `should_run_now()` is defined in `scheduling.py`, imported by `scheduled_refresh.py`, and wrapped through `import scheduling` in `wealth_utility_production.py`.
- `refresh_allocations(base_url, artifact_path, timeout_seconds)` is used consistently by tests and `main()`.

## Execution Options

Plan complete and saved to `docs/superpowers/plans/2026-06-02-fix-wealth-utility-api-scheduling.md`.

Two execution options:

1. Subagent-Driven (recommended) - dispatch a fresh subagent per task, review between tasks, fast iteration.
2. Inline Execution - execute tasks in this session using executing-plans, batch execution with checkpoints.
