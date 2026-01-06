# Wealth Utility - Lovable.dev Integration Guide

## Overview

This guide shows you how to integrate the Wealth Utility allocation calculator with your Lovable.dev web application.

## Architecture

```
Lovable.dev Web App (Frontend)
    ↓ HTTP Request
Flask API Server (Backend)
    ↓ Runs calculation
wealth_utility_production.py
    ↓ Returns JSON
Lovable.dev displays allocations
```

## Files Created

1. **wealth_utility_api.py** - Flask REST API server
2. **requirements_api.txt** - Python dependencies for the API
3. **lovable_integration.tsx** - React component for Lovable.dev
4. **test_api.html** - Simple HTML test page

## Quick Start

### Step 1: Test the API Locally

1. Install dependencies:
```bash
cd "C:\Users\BrandonVanLandingham\OneDrive - Perissos Private Wealth Management\1 Perissos Private Wealth Management\5 Python\Wealth Utility"
pip install -r requirements_api.txt
```

2. Start the API server:
```bash
python wealth_utility_api.py
```

3. The API will be available at `http://localhost:5000`

4. Test in your browser:
   - Open: http://localhost:5000/
   - Check allocations: http://localhost:5000/allocations

### Step 2: Deploy the API

You have several deployment options:

#### Option A: Railway.app (Recommended - Easiest)

1. Go to https://railway.app/
2. Sign up/login with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Connect your repository or upload the files
5. Railway will auto-detect Python and install dependencies
6. Set environment variables if needed
7. Get your deployment URL (e.g., `https://your-app.up.railway.app`)

#### Option B: Render.com

1. Go to https://render.com/
2. Sign up/login
3. Click "New +" → "Web Service"
4. Connect GitHub or deploy from repo
5. Configure:
   - Build Command: `pip install -r requirements_api.txt`
   - Start Command: `gunicorn wealth_utility_api:app`
6. Deploy and get your URL

#### Option C: Vercel (Serverless)

1. Install Vercel CLI: `npm install -g vercel`
2. Create `vercel.json`:
```json
{
  "builds": [
    {
      "src": "wealth_utility_api.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "wealth_utility_api.py"
    }
  ]
}
```
3. Run: `vercel --prod`

#### Option D: Keep it Local (Development Only)

If you just want to test locally:
1. Run `python wealth_utility_api.py` on your computer
2. Use ngrok to expose it: `ngrok http 5000`
3. Use the ngrok URL in your Lovable app

### Step 3: Integrate with Lovable.dev

#### Method 1: Copy the React Component

1. Open your Lovable.dev project
2. Create a new component file (e.g., `WealthUtility.tsx`)
3. Copy the code from `lovable_integration.tsx`
4. Update the `API_BASE_URL` to your deployed API URL:
```typescript
const API_BASE_URL = 'https://your-api.up.railway.app'; // Your deployed URL
```
5. Import and use the component in your app:
```typescript
import WealthUtilityDashboard from '@/components/WealthUtility';

function App() {
  return (
    <div>
      <WealthUtilityDashboard />
    </div>
  );
}
```

#### Method 2: Simple Fetch Example

If you prefer a simpler approach, here's minimal code:

```typescript
import { useState, useEffect } from 'react';

export default function SimpleAllocations() {
  const [allocations, setAllocations] = useState(null);

  useEffect(() => {
    fetch('https://your-api.up.railway.app/allocations')
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          setAllocations(data);
        }
      });
  }, []);

  if (!allocations) return <div>Loading...</div>;

  return (
    <div>
      <h1>Portfolio Allocations</h1>
      <div>
        <h2>Total Equity: {allocations.summary.total_equity_pct}</h2>
        <h2>Total Fixed Income: {allocations.summary.total_fixed_income_pct}</h2>
      </div>
      <ul>
        {allocations.allocations.map(item => (
          <li key={item.ticker}>
            {item.ticker}: {item.weight_pct}
          </li>
        ))}
      </ul>
    </div>
  );
}
```

## API Endpoints

### GET /allocations
Returns current portfolio allocations (cached for 1 hour).

**Response:**
```json
{
  "success": true,
  "calculation_date": "2026-01-06T12:30:00",
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
      "weight": 0.00,
      "weight_pct": "0.00%"
    }
  ],
  "summary": {
    "total_equity": 1.0,
    "total_equity_pct": "100.00%",
    "total_fixed_income": 0.0,
    "total_fixed_income_pct": "0.00%"
  },
  "strategy_params": {
    "sleeve_method": "max_sharpe",
    "band_mode": "absolute",
    "risk_dial_mode": "band",
    "value_dial": 25,
    "momentum_dial": 75
  }
}
```

### POST /allocations/refresh
Forces a fresh calculation, bypassing the cache.

### GET /health
Health check endpoint.

### GET /config
Returns current strategy configuration.

## CORS Configuration

The API includes CORS headers, allowing requests from any origin. For production, you may want to restrict this:

```python
# In wealth_utility_api.py, replace:
CORS(app)

# With:
CORS(app, origins=['https://your-lovable-app.lovable.app'])
```

## Caching

The API caches results for 1 hour by default. To change:

```python
# In wealth_utility_api.py
_cache = {
    'cache_duration_minutes': 30  # Change to 30 minutes
}
```

## Testing with cURL

Test your deployed API:

```bash
# Get allocations
curl https://your-api.up.railway.app/allocations

# Refresh allocations
curl -X POST https://your-api.up.railway.app/allocations/refresh

# Health check
curl https://your-api.up.railway.app/health
```

## Environment Variables (Optional)

For better security, set API keys as environment variables:

```bash
# On your deployment platform, set:
FMP_KEY=your_fmp_key_here
FRED_API_KEY=your_fred_key_here
```

Then update `wealth_utility_api.py` to read from environment:

```python
FMP_KEY = os.getenv("FMP_KEY")
FRED_KEY = os.getenv("FRED_API_KEY")
```

## Automated Updates

To automatically refresh allocations on the last trading day:

1. Deploy the API to a platform that supports cron jobs (Railway, Render)
2. Add a cron job that calls `/allocations/refresh` monthly
3. Or use the Windows Task Scheduler approach from the production deployment

Example cron (runs monthly on the 28th at 5 PM):
```bash
0 17 28 * * curl -X POST https://your-api.up.railway.app/allocations/refresh
```

## Troubleshooting

### CORS Errors
- Ensure `flask-cors` is installed
- Check that CORS is enabled in the API
- Verify your frontend URL is allowed

### API Not Responding
- Check if the Excel file path is correct
- Verify API keys are valid
- Check server logs for errors

### Slow First Request
- First calculation takes time (data fetching + processing)
- Subsequent requests use cached data (fast)
- Consider warming up the cache on deployment

## Example Lovable.dev Pages

### Dashboard Page
```typescript
import WealthUtilityDashboard from '@/components/WealthUtility';

export default function DashboardPage() {
  return (
    <div className="container mx-auto py-8">
      <WealthUtilityDashboard />
    </div>
  );
}
```

### Simple Card View
```typescript
import { useQuery } from '@tanstack/react-query';
import { Card } from '@/components/ui/card';

export default function AllocationCard() {
  const { data } = useQuery({
    queryKey: ['allocations'],
    queryFn: () =>
      fetch('https://your-api.up.railway.app/allocations')
        .then(res => res.json()),
    refetchInterval: 3600000, // Refresh every hour
  });

  if (!data?.success) return null;

  return (
    <Card className="p-6">
      <h3 className="text-xl font-bold mb-4">Current Allocation</h3>
      <div className="space-y-2">
        <div className="flex justify-between">
          <span>Equity:</span>
          <span className="font-bold">{data.summary.total_equity_pct}</span>
        </div>
        <div className="flex justify-between">
          <span>Fixed Income:</span>
          <span className="font-bold">{data.summary.total_fixed_income_pct}</span>
        </div>
      </div>
    </Card>
  );
}
```

## Next Steps

1. Deploy the API to your preferred platform
2. Test the API endpoints
3. Copy the React component to your Lovable project
4. Update the API URL in the component
5. Style and customize as needed
6. Add authentication if required

## Security Considerations

- **API Keys**: Don't expose API keys in frontend code
- **Rate Limiting**: Consider adding rate limiting to the API
- **Authentication**: Add authentication for production use
- **HTTPS**: Always use HTTPS in production
- **CORS**: Restrict CORS to your Lovable domain only

## Support

For issues:
1. Check API logs on your deployment platform
2. Test endpoints with cURL
3. Verify data files are accessible
4. Check API key validity
