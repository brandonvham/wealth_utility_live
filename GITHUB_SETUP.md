# GitHub Setup Guide

## ✅ Security Check Complete

Your project is now ready for GitHub! All sensitive data has been secured:

- ✅ API keys removed from code
- ✅ Environment variables configured
- ✅ `.gitignore` updated
- ✅ `.env` file excluded from Git
- ✅ `.env.example` template added
- ✅ Documentation complete

## 🚀 Push to GitHub (Step-by-Step)

### Step 1: Initialize Git Repository

```bash
cd "C:\Users\BrandonVanLandingham\OneDrive - Perissos Private Wealth Management\1 Perissos Private Wealth Management\5 Python\01_Active_Strategies\Wealth Utility API"

# Initialize git (if not already done)
git init

# Check current status
git status
```

### Step 2: Add Files

```bash
# Add all files (respecting .gitignore)
git add .

# Verify what will be committed
git status

# Make sure .env is NOT in the list (should be ignored)
```

### Step 3: Create First Commit

```bash
git commit -m "Initial commit: Wealth Utility adaptive portfolio allocation system

- Core calculation engine with adaptive risk management
- Flask REST API for web integration
- React components for Lovable.dev
- GitHub Actions production scheduler
- Comprehensive documentation
"
```

### Step 4: Create GitHub Repository

1. Go to https://github.com/
2. Click "+" → "New repository"
3. Repository name: `wealth_utility_live`
4. Description: "Adaptive portfolio allocation with REST API"
5. Choose **Private** (recommended for financial data)
6. **DO NOT** initialize with README (we already have one)
7. Click "Create repository"

### Step 5: Connect and Push

GitHub will show you commands like this:

```bash
# Add GitHub as remote
git remote add origin https://github.com/YOUR_USERNAME/wealth_utility_live.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## 📦 What Gets Committed

✅ **Included in GitHub:**
- Python source code (`.py` files)
- Documentation (`.md` files)
- Configuration templates (`.env.example`)
- Requirements files
- Excel data file (`ecy4.xlsx`)
- React components (`.tsx`)
- Test files

❌ **Excluded from GitHub (via .gitignore):**
- `.env` file (contains your actual API keys)
- Output files (`current_allocation.txt`, etc.)
- Python cache (`__pycache__/`)
- Virtual environments
- IDE settings
- Logs

## 🔐 Security Verification

Before pushing, verify no secrets are included:

```bash
# Check if .env is ignored
git status | grep ".env"
# Should show nothing (file is ignored)

# Search for potential API keys in tracked files
git grep -i "api.*key" | grep -v ".env.example" | grep -v "your_.*_key_here"
# Should show only documentation references
```

## 🚢 Deploy to Railway After GitHub Push

Once code is on GitHub:

1. **Go to Railway.app**
   - https://railway.app/
   - Sign in with GitHub

2. **Create New Project**
   - "New Project" → "Deploy from GitHub repo"
   - Select your `wealth_utility_live` repository

3. **Configure Environment Variables**
   - In Railway dashboard, go to "Variables"
   - Add:
     - `FMP_KEY` = your actual FMP API key
     - `FRED_API_KEY` = your actual FRED API key
   - These are read from environment, not from .env file

4. **Railway Auto-Deploys**
   - Detects Python project
   - Installs from `requirements_api.txt`
   - Starts with `gunicorn`
   - Gives you a public URL

5. **Test Your Deployment**
   ```bash
   curl https://your-app.up.railway.app/health
   curl https://your-app.up.railway.app/allocations
   ```

## 🔄 Future Updates

After initial push, to update GitHub:

```bash
# Make your changes to files
# Then:

git add .
git commit -m "Description of changes"
git push
```

Railway will automatically redeploy when you push to GitHub!

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

## 🌿 Branching Strategy (Optional)

For safer development:

```bash
# Create development branch
git checkout -b development

# Make changes and test
# ...

# Commit to development
git add .
git commit -m "New feature"
git push -u origin development

# When ready, merge to main
git checkout main
git merge development
git push
```

## ⚠️ Important Reminders

1. **Never commit .env file** - Already in .gitignore
2. **Private repository recommended** - Contains financial strategy
3. **Environment variables in Railway** - Not in code
4. **Test locally first** - Before pushing
5. **API rate limits** - Be mindful of FMP/FRED limits

## 🆘 Troubleshooting

### "API keys not set" error after deployment
- Solution: Add environment variables in Railway dashboard

### Large files error
- `ecy4.xlsx` should be fine (~1MB typically)
- If too large, add to .gitignore and upload separately

### Git tracking .env file
```bash
# If .env was accidentally committed:
git rm --cached .env
git commit -m "Remove .env from tracking"
```

## ✅ Final Checklist

Before pushing to GitHub:

- [ ] API keys removed from code
- [ ] `.env` file exists locally (for your use)
- [ ] `.env` is in `.gitignore`
- [ ] `.env.example` has template (no real keys)
- [ ] `README.md` is complete
- [ ] All files added: `git add .`
- [ ] First commit created
- [ ] GitHub repository created (private)
- [ ] Remote added: `git remote add origin ...`
- [ ] Pushed: `git push -u origin main`

---

**You're all set! 🎉**

Your code is secure and ready for GitHub and Railway deployment.
