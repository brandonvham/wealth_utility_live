# Wealth Utility API - Quick Start Guide

## ✅ What Was Created

I've created a complete integration package for your Lovable.dev webapp:

### Files
1. **wealth_utility_api.py** - Flask REST API server
2. **requirements_api.txt** - Python dependencies
3. **lovable_integration.tsx** - React component for Lovable
4. **test_api.html** - Simple HTML test page
5. **LOVABLE_INTEGRATION_GUIDE.md** - Complete integration guide
6. **QUICK_START.md** - This file

## 🚀 Quick Test (5 minutes)

### Step 1: Install Dependencies
```bash
cd "C:\Users\BrandonVanLandingham\OneDrive - Perissos Private Wealth Management\1 Perissos Private Wealth Management\5 Python\Wealth Utility"
pip install flask flask-cors
```

### Step 2: Start the API Server
```bash
python wealth_utility_api.py
```

You should see:
```
================================================================================
WEALTH UTILITY API SERVER
================================================================================
Starting Flask development server...
API will be available at: http://localhost:5000
...
```

### Step 3: Test in Browser

**Option A: Interactive Test Page**
1. Open `test_api.html` in your browser (just double-click it)
2. Click "Load Allocations"
3. See your portfolio displayed beautifully!

**Option B: Direct API Test**
1. Open browser to: http://localhost:5000/
2. See API documentation
3. Go to: http://localhost:5000/allocations
4. See JSON response with your allocations

### Step 4: Test with cURL (Optional)
```bash
# Get allocations
curl http://localhost:5000/allocations

# Health check
curl http://localhost:5000/health

# Force refresh
curl -X POST http://localhost:5000/allocations/refresh
```

## 🌐 Deploy to Internet (For Lovable Integration)

### Easiest Option: Railway.app

1. **Go to https://railway.app/**
2. **Sign up** (free tier available)
3. **Create New Project** → "Empty Project"
4. **Add Service** → "GitHub Repo" or "Empty Service"
5. **Upload your files**:
   - wealth_utility_api.py
   - wealth_utility_production.py
   - requirements_api.txt
   - ecy4.xlsx (your Excel file)

6. **Configure the service**:
   - Root Directory: (leave default)
   - Build Command: `pip install -r requirements_api.txt`
   - Start Command: `gunicorn wealth_utility_api:app`

7. **Deploy!**
   - Railway will give you a URL like: `https://wealth-utility-production.up.railway.app`

8. **Test your deployed API**:
   ```bash
   curl https://your-app.up.railway.app/allocations
   ```

### Alternative: Render.com

1. Go to https://render.com/
2. New Web Service
3. Connect GitHub or upload code
4. Build: `pip install -r requirements_api.txt`
5. Start: `gunicorn wealth_utility_api:app`
6. Deploy!

## 📱 Add to Lovable.dev

### Quick Integration (3 Steps)

1. **Open your Lovable.dev project**

2. **Create new component**: `src/components/WealthUtility.tsx`
   - Copy the entire contents of `lovable_integration.tsx`
   - Update line 10: Change `API_BASE_URL` to your deployed URL:
   ```typescript
   const API_BASE_URL = 'https://your-app.up.railway.app';
   ```

3. **Use the component** in your app:
   ```typescript
   import WealthUtilityDashboard from '@/components/WealthUtility';

   function DashboardPage() {
     return <WealthUtilityDashboard />;
   }
   ```

That's it! Your Lovable app will now display live portfolio allocations.

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API documentation |
| `/allocations` | GET | Get current allocations (cached 1hr) |
| `/allocations/refresh` | POST | Force recalculate allocations |
| `/health` | GET | Health check |
| `/config` | GET | Get strategy parameters |

## 🎨 Example API Response

```json
{
  "success": true,
  "allocation_date": "2026-01-31",
  "allocations": [
    {
      "ticker": "^GSPC",
      "asset_class": "equity",
      "weight": 0.25,
      "weight_pct": "25.00%"
    },
    {
      "ticker": "ZNUSD",
      "asset_class": "fixed_income",
      "weight": 0.0,
      "weight_pct": "0.00%"
    }
  ],
  "summary": {
    "total_equity": 1.0,
    "total_equity_pct": "100.00%",
    "total_fixed_income": 0.0,
    "total_fixed_income_pct": "0.00%"
  }
}
```

## 🔧 Troubleshooting

### API won't start
```bash
# Check if port 5000 is already in use
netstat -ano | findstr :5000

# Or use a different port
# Edit wealth_utility_api.py, line 869:
app.run(debug=True, host='0.0.0.0', port=8000)
```

### Excel file not found
- Make sure `ecy4.xlsx` is in the same directory
- Or update the path in `wealth_utility_production.py` line 25

### API keys not working
- Check FMP_KEY and FRED_API_KEY are correct
- Test them separately at:
  - FMP: https://financialmodelingprep.com/developer/docs/
  - FRED: https://fred.stlouisfed.org/docs/api/fred/

## 💡 Pro Tips

1. **Caching**: API caches results for 1 hour to save computation time
2. **Performance**: First request is slow (fetches data), subsequent are fast
3. **Updates**: Call `/allocations/refresh` monthly to update
4. **Security**: Add authentication before making public
5. **Monitoring**: Check Railway/Render logs for errors

## 📚 Next Steps

1. ✅ Test API locally with test_api.html
2. ✅ Deploy to Railway or Render
3. ✅ Copy React component to Lovable
4. ✅ Update API URL in component
5. ✅ Test in your Lovable app
6. ⭐ Customize styling to match your app
7. 🔒 Add authentication if needed

## 🆘 Need Help?

- Check `LOVABLE_INTEGRATION_GUIDE.md` for detailed instructions
- Test endpoints with `test_api.html`
- Review Railway/Render deployment logs
- Verify Excel file and API keys are accessible

## 🎯 What This Solves

✅ **Automated Calculations**: No more manual backtest runs
✅ **Live Data**: Always current allocations
✅ **Web Integration**: Display in any web app
✅ **API Access**: Use data anywhere (mobile app, dashboard, etc.)
✅ **Caching**: Fast responses after first calculation
✅ **Professional**: Production-ready REST API

---

**You now have:**
- ✅ Scheduled monthly production script (Windows Task Scheduler)
- ✅ REST API for web integration (Flask)
- ✅ React component for Lovable.dev
- ✅ Test page for quick validation
- ✅ Deployment guides for Railway/Render
