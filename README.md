# Wealth Utility - Adaptive Portfolio Allocation

An automated portfolio allocation system using adaptive risk management, value signals, and momentum indicators.

## Features

- **Adaptive Risk Management**: Dynamic allocation based on market conditions
- **Multiple Asset Optimization**: Max Sharpe, Risk Parity, Beta Maximization
- **Value & Momentum Signals**: Multi-factor approach
- **REST API**: Flask-based API for web integration
- **Automated Scheduling**: Monthly production runs
- **Web Dashboard**: Beautiful React components for visualization

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd "Wealth Utility"

# Install dependencies
pip install -r requirements_api.txt
```

### 2. Configure API Keys

```bash
# Copy the environment template
cp .env.example .env

# Edit .env and add your API keys
# - FMP_KEY: Get from https://financialmodelingprep.com/
# - FRED_API_KEY: Get from https://fred.stlouisfed.org/
```

### 3. Test the System

```bash
# Test the calculation engine
python test_api_standalone.py

# Start the API server
python wealth_utility_api.py

# View in browser
# Open test_api.html in your browser
```

## Project Structure

```
Wealth Utility/
├── wealth_utility_production.py    # Core calculation engine
├── wealth_utility_api.py            # Flask REST API
├── lovable_integration.tsx          # React component
├── test_api_standalone.py           # Standalone test
├── test_api.html                    # Web test interface
├── ecy4.xlsx                        # Valuation data
├── requirements_api.txt             # Python dependencies
├── .env.example                     # Environment template
└── README.md                        # This file
```

## Deployment

### Railway (Recommended)

1. Push to GitHub
2. Go to [Railway.app](https://railway.app/)
3. Create new project from GitHub repo
4. Add environment variables:
   - `FMP_KEY`
   - `FRED_API_KEY`
5. Deploy!

Railway will automatically:
- Install dependencies from `requirements_api.txt`
- Start the API with `gunicorn`
- Provide a public URL

### Render

1. Go to [Render.com](https://render.com/)
2. New Web Service → Connect GitHub
3. Build Command: `pip install -r requirements_api.txt`
4. Start Command: `gunicorn wealth_utility_api:app`
5. Add environment variables
6. Deploy

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API documentation |
| `/allocations` | GET | Get current allocations (cached 1hr) |
| `/allocations/refresh` | POST | Force recalculate |
| `/health` | GET | Health check |
| `/config` | GET | Strategy parameters |

## Example API Response

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
    }
  ],
  "summary": {
    "total_equity_pct": "100.00%",
    "total_fixed_income_pct": "0.00%"
  }
}
```

## Integration with Lovable.dev

```typescript
// In your Lovable app
const response = await fetch('https://your-api.railway.app/allocations');
const data = await response.json();

// Use the allocations
console.log(data.summary.total_equity_pct);
```

Or use the pre-built React component from `lovable_integration.tsx`.

## Configuration

Edit parameters in `wealth_utility_production.py`:

```python
# Asset selection
EQUITY_TICKER = ["^GSPC","^IXIC","MSCIWORLD","GCUSD"]
NON_EQUITY_TICKER = "ZNUSD"

# Strategy dials
VALUE_DIAL_FRAC = 25      # Value signal strength
MOM_BUMP_FRAC = 75        # Momentum signal strength
BAND_MODE = "absolute"     # Band definition mode
RISK_DIAL_MODE = "band"    # Risk adjustment mode
```

## Scheduled Production Runs

For automated monthly calculations:

### Windows Task Scheduler

```powershell
cd "path/to/Wealth Utility"
.\setup_scheduler.ps1
```

This creates a task that runs daily at 5 PM CT and executes on the last trading day.

## Testing

```bash
# Standalone test (no server needed)
python test_api_standalone.py

# Full API test (requires server running)
python test_api.py

# Or use the web interface
# Open test_api.html in browser
```

## Documentation

- `QUICK_START.md` - 5-minute setup guide
- `LOVABLE_INTEGRATION_GUIDE.md` - Web app integration
- `README_PRODUCTION.md` - Scheduled execution setup

## Strategy Overview

The Wealth Utility strategy combines:

1. **Value Signal**: Risk premium relative to historical anchor
2. **Momentum Signal**: Trend-following based on moving averages
3. **Risk Dial**: Volatility-adjusted position sizing
4. **Equity Sleeve Optimization**: Max Sharpe, Risk Parity, or Beta Max

## Requirements

- Python 3.8+
- pandas, numpy, scipy
- flask, flask-cors
- requests
- openpyxl
- scikit-learn

See `requirements_api.txt` for full list.

## Security

⚠️ **Important:**
- Never commit `.env` file
- Always use environment variables for API keys
- Use `.env.example` as template only

## License

Private/Proprietary

## Support

For issues:
1. Check API logs
2. Verify environment variables are set
3. Test with `test_api_standalone.py`
4. Check data file paths (ecy4.xlsx)

---

**Generated with Claude Code** 🤖
