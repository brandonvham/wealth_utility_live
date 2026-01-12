# ✅ Wealth Utility API - Production Deployment Summary

**Current Version:** v2.5.0
**Last Updated:** January 12, 2026
**Status:** ✅ Deployed to Railway
**URL:** https://wealth-utility-live-production.up.railway.app

---

## 📁 Project Architecture

### 🔬 Research & Development
**Location:** `5 Python/Wealth Utility/` & `5 Python/Merged Wealth Utility and FI/`
- Jupyter notebooks for strategy development
- `merged_wealth_utility_fi.py` - Merged Wealth Utility + Dynamic FI strategy
- `Merged_wealth_fi.ipynb` - Notebook with quantstats reporting
- Historical backtesting and analysis
- **Stays local** - not deployed

### 🚀 Production API (Railway)
**Location:** `5 Python/Wealth Utility API/` ← **THIS FOLDER**
- `wealth_utility_production.py` - Core calculation engine (~1,250 lines)
- `wealth_utility_api.py` - Flask REST API wrapper
- Multi-profile risk allocations (5 profiles)
- Dynamic Fixed Income sleeve for allocations
- **Deployed to Railway** via GitHub auto-deploy

---

## 🚀 Recent Updates & Version History

### v2.5.0 (January 12, 2026) - Latest
**Commit:** `e8c6872`
- ✅ **Separate ticker lists**: Allocations and backtest now independent
  - `ALLOCATIONS_EQUITY_TICKERS` - Used only for `/allocations` display
  - `EQUITY_TICKER` (production.py) - Used for `/backtest` default
- ✅ **User-configurable backtest tickers**: Accept `equity_tickers` query parameter
  - Example: `/backtest?equity_tickers=SPY,QQQ,VTI&baseline_w=0.6`
- ✅ **Timeout fixes**: Increased gunicorn timeout (120s → 300s)
- ✅ **Longer cache**: Allocations 4hr, Backtest 8hr (reduced API load)
- ✅ **Updated ticker lists**:
  - Allocations: ACWI, COWG, COWZ, EDIV, IWR, JIVE, JMEE, JQUA, MGK, PDBC, REZ
  - FI: PFFD, FMHI, VWOB, SRLN, ANGL, ICVT, TLT, BIL

### v2.4.0 (January 12, 2026)
**Commit:** `605ae03`
- ✅ **Dynamic Fixed Income allocation** for `/allocations` endpoint
- ✅ Replaces single BIL with 8-ticker FI sleeve (moving average signals)
- ✅ `/backtest` unchanged (still uses BIL for safe asset)

### v2.3.0 (January 12, 2026)
**Commit:** `01ffa72`
- ✅ **Multi-profile risk allocations**: 5 risk profiles in single API response
  - all_equity (baseline 1.0)
  - moderate_aggressive (baseline 0.80)
  - moderate (baseline 0.60)
  - moderate_conservative (baseline 0.50)
  - conservative (baseline 0.40)
- ✅ Dynamic f_max calculation (baseline_w + 15%)
- ✅ Optional profile filter: `/allocations?profile=moderate`

### v2.2.0 (Earlier)
- ✅ Dynamic f_max based on baseline_w
- ✅ f_max = min(baseline_w + 15%, 100%)

### v2.1.0 (Earlier)
- ✅ User-configurable baseline_w parameter via `/backtest` endpoint

### v2.0.0 (Earlier)
- ✅ CAGR calculation fix (properly accounts for starting NAV)
- ✅ Initial Railway deployment

---

## 📦 Current Configuration

### Production Files
- [x] `wealth_utility_production.py` - Core calculation engine (1,252 lines)
- [x] `wealth_utility_api.py` - Flask REST API (777 lines)
- [x] `Procfile` - Railway deployment config (gunicorn, 300s timeout)
- [x] `requirements.txt` - Python dependencies
- [x] `.env.example` - Environment variable template
- [x] `ecy4.xlsx` - Valuation data (CAPE, earnings yield)

### Documentation
- [x] `prompt.md` - Lovable integration guide for v2.5.0
- [x] `DEPLOYMENT_SUMMARY.md` - This file
- [x] `LOVABLE_INTEGRATION_GUIDE.md` - Web app integration
- [x] `GITHUB_SETUP.md` - Deployment instructions
- [x] Other guides (START_HERE, QUICK_START, etc.)

### Security & Environment
- [x] API keys secured via Railway environment variables
- [x] `.env` excluded from Git (`.gitignore`)
- [x] CORS enabled for all origins
- [x] No secrets in codebase

### Deployment
- [x] GitHub repository: `wealth_utility_live`
- [x] Railway auto-deploy from `main` branch
- [x] Health check endpoint: `/health`
- [x] Public domain configured

---

## 🌐 API Endpoints (v2.5.0)

**Base URL:** `https://wealth-utility-live-production.up.railway.app`

### GET /health
Health check endpoint
```bash
curl https://wealth-utility-live-production.up.railway.app/health
```

### GET /allocations
Get current portfolio allocations for all 5 risk profiles with Dynamic FI sleeve

**Query Parameters:**
- `profile` (optional) - Filter to single profile: `all_equity`, `moderate_aggressive`, `moderate`, `moderate_conservative`, `conservative`

**Cache:** 4 hours

**Example:**
```bash
# All profiles
GET /allocations

# Single profile
GET /allocations?profile=moderate
```

**Response:** 5 risk profiles, each with detailed equity + FI allocations

### POST /allocations/refresh
Force refresh allocations (bypass 4-hour cache)

### GET /backtest
Run historical backtest with performance metrics

**Query Parameters:**
- `start_date` (optional) - YYYY-MM-DD format
- `end_date` (optional) - YYYY-MM-DD format
- `baseline_w` (optional) - 0.0 to 1.0 (default from config)
- `equity_tickers` (optional) - Comma-separated tickers (e.g., "SPY,QQQ,VTI")
- `force_refresh` (optional) - Set to 'true' to bypass cache

**Cache:** 8 hours

**Example:**
```bash
# Default tickers and baseline
GET /backtest

# Custom configuration
GET /backtest?baseline_w=0.6&equity_tickers=SPY,QQQ,VTI&start_date=2010-01-01
```

**Response:** Full backtest with NAV, drawdowns, rolling metrics, allocation history

### POST /backtest/refresh
Force refresh backtest (bypass 8-hour cache)

Same query parameters as GET /backtest

### GET /config
Get current strategy configuration

### GET /
API documentation and available endpoints

---

## 🎯 Current Strategy Configuration

### Allocations Endpoint (`/allocations`)
**Equity Tickers:** ACWI, COWG, COWZ, EDIV, IWR, JIVE, JMEE, JQUA, MGK, PDBC, REZ (11 tickers)

**Fixed Income Tickers:** PFFD, FMHI, VWOB, SRLN, ANGL, ICVT, TLT, BIL (8 tickers)

**Method:** Max Sharpe optimization with Ledoit-Wolf covariance shrinkage

**Safe Asset:** Dynamic Fixed Income sleeve (moving average signals)

### Backtest Endpoint (`/backtest`)
**Default Equity Tickers:** RPV, RPG, IWR, EFA, QQQ, EEM, VTI, DBC, IYR (from production.py)

**Safe Asset:** BIL (simple)

**Note:** User can override with `equity_tickers` query parameter

### Risk Profiles (All Endpoints)
1. **All Equity** - baseline_w: 1.0, f_max: 1.0
2. **Moderate Aggressive** - baseline_w: 0.80, f_max: 0.95
3. **Moderate** - baseline_w: 0.60, f_max: 0.75
4. **Moderate Conservative** - baseline_w: 0.50, f_max: 0.65
5. **Conservative** - baseline_w: 0.40, f_max: 0.55

---

## 🔧 Performance & Optimization

### Caching Strategy
- **Allocations:** 240 minutes (4 hours)
- **Backtest:** 480 minutes (8 hours)
- **First request:** 2-5 minutes (fetches all data)
- **Cached requests:** < 100ms (instant response)

### Timeout Configuration
- **Gunicorn:** 300 seconds (5 minutes)
- **Railway platform:** ~5 minutes
- Sufficient for 17-ticker calculations

### Data Sources
- **FMP API:** Daily price data for all tickers
- **FRED API:** TIPS 10-year rate (DFII10)
- **Excel file:** CAPE and earnings yield data (ecy4.xlsx)

---

## 🚀 Deployment Status

### GitHub Repository
**URL:** https://github.com/brandonvham/wealth_utility_live
- ✅ Auto-deploy to Railway on push to `main`
- ✅ All commits signed with Claude Code co-authorship
- ✅ Comprehensive commit history with detailed messages

### Railway Deployment
**URL:** https://wealth-utility-live-production.up.railway.app
- ✅ Production environment
- ✅ Environment variables configured (FMP_KEY, FRED_API_KEY)
- ✅ Auto-deploy enabled from GitHub
- ✅ 300s timeout, sufficient for calculations
- ✅ Public domain configured

### Lovable Integration
- ✅ CORS enabled for all origins
- ✅ REST API ready for webapp integration
- ✅ Supabase Edge Function proxy (optional, for additional CORS handling)
- ✅ Comprehensive documentation in `prompt.md`

---

## 🏗️ Architecture Overview

### File Separation
**`wealth_utility_production.py`** (Core Engine)
- All calculation logic and business rules
- `run_backtest()` - Full historical backtest
- `calculate_current_allocations()` - Current month allocation
- Can be run standalone (has `main()` function)
- NOT directly executed by Railway

**`wealth_utility_api.py`** (Web Server)
- Flask REST API wrapper
- Imports functions from `wealth_utility_production.py`
- Adds caching, multi-profile calculations, Dynamic FI sleeve
- **This is what Railway runs** (via Procfile)
- Serves HTTP requests from Lovable webapp

### Data Flow
```
Lovable Webapp
    ↓
[Optional: Supabase Edge Function CORS proxy]
    ↓
Railway: wealth_utility_api.py (Flask server)
    ↓
Check cache (4hr allocations, 8hr backtest)
    ↓
If cache miss: Call wealth_utility_production.py functions
    ↓
Fetch data: FMP API, FRED API, ecy4.xlsx
    ↓
Run calculations: Optimization, signals, allocations
    ↓
Cache results & return JSON
    ↓
Lovable displays to user
```

---

## 📚 Documentation Quick Reference

| Need to... | Read this |
|------------|-----------|
| Understand v2.5.0 changes | `prompt.md` |
| Integrate with Lovable | `LOVABLE_INTEGRATION_GUIDE.md` |
| Review API endpoints | This file (DEPLOYMENT_SUMMARY.md) |
| Deploy to Railway | `GITHUB_SETUP.md` |
| Test locally | `QUICK_START.md` |
| Automate monthly runs | `README_PRODUCTION.md` |

---

## 🔍 Troubleshooting

### "Failed to fetch" errors
**Likely cause:** Timeout, not CORS
- **Solution:** Wait 5 minutes for calculation to complete
- **Cache:** Subsequent requests within 4-8 hours are instant
- **Fixed in v2.5.0:** Increased timeout to 300s, cache to 4-8hr

### "Application not found" on Railway
**Cause:** Deployment in progress or failed
- Check Railway dashboard for deployment status
- View logs for build/startup errors
- Verify environment variables are set (FMP_KEY, FRED_API_KEY)

### Slow first request
**Expected behavior:** 2-5 minutes for first request after cache expires
- Fetching 19 tickers × 4,500 days of data
- Running optimization and calculations
- Subsequent requests are instant (< 100ms)

### Different allocations vs backtest tickers
**Expected behavior:** Different ticker lists since v2.5.0
- `/allocations` uses `ALLOCATIONS_EQUITY_TICKERS` (11 tickers)
- `/backtest` uses `EQUITY_TICKER` from production.py (9 tickers)
- This is intentional - allows independent configurations

---

## ✅ Deployment Checklist

### Infrastructure ✅
- [x] GitHub repository created: `wealth_utility_live`
- [x] Code pushed to GitHub main branch
- [x] Railway project created and linked to GitHub
- [x] Railway environment variables configured
- [x] Auto-deploy enabled from GitHub
- [x] Public domain configured
- [x] Health endpoint responding

### Code & Configuration ✅
- [x] API keys secured via Railway environment variables
- [x] `.env` excluded from Git (`.gitignore`)
- [x] CORS enabled for all origins
- [x] Gunicorn timeout set to 300s
- [x] Cache durations optimized (4hr/8hr)
- [x] Separate ticker lists for allocations/backtest
- [x] Dynamic FI sleeve implemented

### Documentation ✅
- [x] `prompt.md` - Lovable integration guide (v2.5.0)
- [x] `DEPLOYMENT_SUMMARY.md` - This file (updated)
- [x] Version history documented
- [x] API endpoints documented
- [x] Troubleshooting guide included

### Testing ✅
- [x] Health endpoint tested
- [x] Allocations endpoint tested
- [x] Backtest endpoint tested
- [x] Multi-profile response validated
- [x] Dynamic FI allocations verified
- [x] Custom ticker parameters tested
- [x] Cache expiration behavior verified

---

## 🎯 What You Have Now

✅ **v2.5.0 deployed** - Latest features live on Railway
✅ **Multi-profile allocations** - 5 risk levels in one API call
✅ **Dynamic Fixed Income** - 8-ticker FI sleeve with MA signals
✅ **Flexible backtesting** - User-configurable ticker lists
✅ **Optimized performance** - 4-8hr caching, 300s timeout
✅ **Separate configurations** - Allocations and backtest independent
✅ **Production-ready** - Deployed, documented, tested
✅ **Lovable-ready** - CORS enabled, comprehensive API docs

---

## 🚀 Recent Commits

```
e8c6872 - Fix timeout issues (300s timeout, 4-8hr cache)
62d7ad3 - Separate ticker lists, user-configurable backtest
9c8f7d3 - Add prompt.md for Lovable v2.5.0
605ae03 - Add Dynamic FI allocation to /allocations
01ffa72 - Add multi-profile risk allocations
13ac87b - Dynamic f_max calculation
2cc8db6 - Add baseline_w parameter
6ea0711 - Fix CAGR calculation
```

---

## 📞 Support & Resources

**API Documentation:** `GET https://wealth-utility-live-production.up.railway.app/`

**Health Check:** `GET https://wealth-utility-live-production.up.railway.app/health`

**GitHub Repo:** https://github.com/brandonvham/wealth_utility_live

**Railway Dashboard:** Check deployment logs and metrics

---

**Status: ✅ Production Ready & Deployed**

Last updated: January 12, 2026 - v2.5.0
