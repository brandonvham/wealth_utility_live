# 🚀 Wealth Utility API - Start Here

**This is the PRODUCTION version** - ready for deployment and web integration.

For backtesting and research, see the `../Wealth Utility/` folder.

---

## ⚡ Quick Test (30 seconds)

```bash
# 1. Make sure you're in the right folder
cd "C:\Users\BrandonVanLandingham\OneDrive - Perissos Private Wealth Management\1 Perissos Private Wealth Management\5 Python\Wealth Utility API"

# 2. Test the API
python test_api_standalone.py
```

You should see allocation percentages output!

---

## 📖 Full Documentation

| Document | Purpose |
|----------|---------|
| **QUICK_START.md** | 5-minute local setup & test |
| **GITHUB_SETUP.md** | Push to GitHub securely |
| **LOVABLE_INTEGRATION_GUIDE.md** | Integrate with web app |
| **README_PRODUCTION.md** | Automated monthly scheduler |
| **README.md** | Complete overview |

---

## 🎯 Common Tasks

### Test Locally
```bash
python test_api_standalone.py
```

### Start API Server
```bash
python wealth_utility_api.py
# Then open test_api.html in browser
```

### Deploy to Railway
1. Read **GITHUB_SETUP.md**
2. Push to GitHub (API keys secured)
3. Deploy on Railway.app
4. Done!

### Schedule Monthly Runs

Production scheduled runs are handled by GitHub Actions:

- Workflow: `.github/workflows/monthly-allocation-refresh.yml`
- Schedule: weekdays at 23:05 UTC, with an internal guard for last NYSE trading day after 5 PM CT
- Manual run: GitHub Actions -> Monthly Allocation Refresh -> Run workflow -> `force_refresh=true`

`setup_scheduler.ps1` is a local Windows fallback only.

### Integrate with Lovable
1. Deploy API to Railway
2. Copy `lovable_integration.tsx` to your Lovable project
3. Update API URL in the component
4. Done!

---

## 🔐 Security Note

✅ **API keys are secured** - stored in `.env` file (not committed to Git)
✅ **`.env` file exists locally** - for your development
✅ **Railway uses environment variables** - set in dashboard

---

## 📁 Folder Structure

```
Wealth Utility API/         ← YOU ARE HERE (Production)
├── wealth_utility_production.py
├── wealth_utility_api.py
├── lovable_integration.tsx
├── test_api_standalone.py
└── ...

../Wealth Utility/          ← Backtesting & Research
├── Wealth Utility.ipynb
├── Earnings Yield.ipynb
└── ...
```

---

## ✅ Ready to Deploy?

Follow these steps in order:

1. ✅ Test locally: `python test_api_standalone.py`
2. ✅ Read: **GITHUB_SETUP.md**
3. ✅ Push to GitHub
4. ✅ Deploy to Railway
5. ✅ Add to Lovable (use **LOVABLE_INTEGRATION_GUIDE.md**)

---

**Questions?** Check the documentation files above or run tests to verify everything works.
