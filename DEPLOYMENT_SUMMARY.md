# ✅ Wealth Utility API - Production Setup Complete

## 📁 Folder Organization

Your project is now cleanly separated:

### 🔬 Research & Backtesting
**Location:** `5 Python/Wealth Utility/`
- Jupyter notebooks for backtesting
- Strategy development and testing
- Historical analysis
- Parameter optimization
- **Stays local** - not deployed

### 🚀 Production & API
**Location:** `5 Python/Wealth Utility API/` ← **THIS FOLDER**
- Production-ready Python scripts
- Flask REST API
- React components for web apps
- Automated scheduling
- **Ready for GitHub and Railway deployment**

---

## ✅ What's Been Set Up

### Security ✅
- [x] API keys removed from code
- [x] Environment variables configured (`.env` file)
- [x] `.env` excluded from Git (`.gitignore`)
- [x] `.env.example` template included
- [x] `python-dotenv` installed and configured

### Production Files ✅
- [x] `wealth_utility_production.py` - Core calculation engine
- [x] `wealth_utility_api.py` - Flask REST API
- [x] `lovable_integration.tsx` - React component
- [x] Testing tools (standalone, web, full suite)
- [x] Windows Task Scheduler setup
- [x] All dependencies listed

### Documentation ✅
- [x] `START_HERE.md` - Quick orientation
- [x] `QUICK_START.md` - 5-minute local test
- [x] `GITHUB_SETUP.md` - Secure GitHub deployment
- [x] `LOVABLE_INTEGRATION_GUIDE.md` - Web app integration
- [x] `README_PRODUCTION.md` - Automated scheduler
- [x] `README.md` - Complete overview
- [x] `../FOLDER_STRUCTURE.md` - Explains separation

### Testing ✅
- [x] API calculation tested and working (14.3 seconds)
- [x] Environment variables loading correctly
- [x] Returns correct JSON format
- [x] Allocations calculated successfully

---

## 🚀 Next Steps - Deploy to GitHub & Railway

### 1️⃣ Initialize Git (if not already done)

```bash
cd "C:\Users\BrandonVanLandingham\OneDrive - Perissos Private Wealth Management\1 Perissos Private Wealth Management\5 Python\Wealth Utility API"

git init
```

### 2️⃣ Add Files

```bash
git add .

# Verify .env is NOT in the list (should be ignored)
git status
```

### 3️⃣ Create Commit

```bash
git commit -m "Initial commit: Wealth Utility API

Production-ready portfolio allocation API with:
- Adaptive risk management and multi-factor signals
- Flask REST API for web integration
- React components for Lovable.dev
- Automated monthly scheduler
- Comprehensive testing and documentation
- Secured API keys with environment variables
"
```

### 4️⃣ Create GitHub Repository

1. Go to https://github.com/
2. Click "+" → "New repository"
3. Name: `wealth-utility-api`
4. **Private repository** (recommended)
5. **DO NOT** initialize with README
6. Create repository

### 5️⃣ Push to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/wealth-utility-api.git
git branch -M main
git push -u origin main
```

### 6️⃣ Deploy to Railway

1. Go to https://railway.app/
2. Sign in with GitHub
3. "New Project" → "Deploy from GitHub repo"
4. Select `wealth-utility-api`
5. Add environment variables:
   - `FMP_KEY` = your FMP API key
   - `FRED_API_KEY` = your FRED API key
6. Railway auto-deploys!
7. Get your URL (e.g., `https://wealth-utility-api.up.railway.app`)

### 7️⃣ Test Your Deployment

```bash
curl https://your-app.up.railway.app/health
curl https://your-app.up.railway.app/allocations
```

### 8️⃣ Integrate with Lovable

1. Copy `lovable_integration.tsx` to your Lovable project
2. Update `API_BASE_URL` to your Railway URL
3. Import and use the component
4. Done!

---

## 📊 Current Allocation (Test Results)

**As of January 31, 2026:**

**Equity Sleeve (100%):**
- ^GSPC: 25.00%
- ^IXIC: 25.00%
- MSCIWORLD: 25.00%
- GCUSD: 25.00%

**Fixed Income (0%):**
- ZNUSD: 0.00%

**Strategy:**
- Method: max_sharpe
- Band Mode: absolute
- Risk Dial: band
- Value Dial: 25%
- Momentum Dial: 75%

---

## 🔄 Monthly Automation (Optional)

To run automatically on last trading day at 5 PM CT:

```powershell
# Run as Administrator
.\setup_scheduler.ps1
```

This creates a Windows Task Scheduler job.

---

## 📚 Documentation Quick Reference

| Need to... | Read this |
|------------|-----------|
| Test locally | QUICK_START.md |
| Push to GitHub | GITHUB_SETUP.md |
| Deploy to Railway | GITHUB_SETUP.md |
| Integrate with Lovable | LOVABLE_INTEGRATION_GUIDE.md |
| Automate monthly runs | README_PRODUCTION.md |
| Understand folder structure | ../FOLDER_STRUCTURE.md |

---

## ✅ Pre-Deployment Checklist

Before deploying:

- [x] ✅ API keys secured in `.env` file
- [x] ✅ `.env` file in `.gitignore`
- [x] ✅ Code tested locally
- [x] ✅ All documentation complete
- [x] ✅ Dependencies listed in `requirements_api.txt`
- [x] ✅ Excel file (`ecy4.xlsx`) included
- [ ] GitHub repository created
- [ ] Code pushed to GitHub
- [ ] Railway environment variables set
- [ ] API deployed and tested
- [ ] Lovable integration complete

---

## 🎯 What You Have Now

✅ **Separate folders** - Research vs Production
✅ **Secure deployment** - No API keys in code
✅ **REST API** - Ready for web apps
✅ **React component** - Pre-built for Lovable
✅ **Testing tools** - Verify everything works
✅ **Documentation** - Complete guides
✅ **Automation** - Monthly scheduler ready
✅ **GitHub ready** - Secure configuration

---

## 🆘 Need Help?

1. **Local testing issues**: Run `python test_api_standalone.py`
2. **GitHub questions**: Read `GITHUB_SETUP.md`
3. **API not working**: Check `.env` file has your keys
4. **Deployment issues**: Check Railway logs
5. **Lovable integration**: See `LOVABLE_INTEGRATION_GUIDE.md`

---

**You're all set to deploy! 🚀**

Start with `GITHUB_SETUP.md` when you're ready to push to GitHub.
