# Wealth Utility API v2.5.0 - Updates for Lovable Webapp

## API Base URL
```
https://wealth-utility-live-production.up.railway.app
```

## Key Changes in v2.5.0

### 1. Separate Ticker Lists
- **`/allocations` endpoint**: Now uses a separate `ALLOCATIONS_EQUITY_TICKERS` list (independent from backtest)
- **`/backtest` endpoint**: Can accept custom equity tickers via query parameters

### 2. New Query Parameter: `equity_tickers`

The `/backtest` and `/backtest/refresh` endpoints now accept an optional `equity_tickers` parameter.

---

## Updated API Endpoints

### GET /allocations
**No changes to existing usage** - continues to work as before.

Returns all 5 risk profiles with Dynamic Fixed Income allocations.

**Example:**
```
GET https://wealth-utility-live-production.up.railway.app/allocations
```

**Response includes:**
- All 5 risk profiles (all_equity, moderate_aggressive, moderate, moderate_conservative, conservative)
- Each profile shows equity allocations (using ALLOCATIONS_EQUITY_TICKERS)
- Each profile shows detailed FI allocations (PFF, USIG, EMB, ITM, LQD, HYG, TLT, BIL)

---

### GET /backtest (UPDATED)

Run historical backtest with **optional custom equity tickers**.

**Query Parameters:**
- `start_date` (optional): Start date in YYYY-MM-DD format
- `end_date` (optional): End date in YYYY-MM-DD format
- `baseline_w` (optional): Baseline equity weight (0.0 to 1.0)
- **`equity_tickers` (NEW, optional)**: Comma-separated list of equity ticker symbols
- `force_refresh` (optional): Set to 'true' to bypass cache

**Default Behavior (if `equity_tickers` not provided):**
Uses the default EQUITY_TICKER constant: `["RPV","RPG","IWR","EFA","QQQ","EEM","VTI","DBC","IYR"]`

**Example with Custom Tickers:**
```
GET https://wealth-utility-live-production.up.railway.app/backtest?equity_tickers=SPY,QQQ,VTI&baseline_w=0.6&start_date=2010-01-01
```

**Example without Custom Tickers (uses default):**
```
GET https://wealth-utility-live-production.up.railway.app/backtest?baseline_w=0.6
```

**Important Notes:**
- Tickers must be comma-separated (e.g., `SPY,QQQ,VTI`)
- No spaces around commas recommended
- Invalid ticker format returns 400 error
- Must provide at least one ticker if using this parameter

---

### POST /backtest/refresh (UPDATED)

Force refresh backtest, bypassing cache. Same parameters as GET /backtest.

**Query Parameters:**
- `start_date` (optional): Start date in YYYY-MM-DD format
- `end_date` (optional): End date in YYYY-MM-DD format
- `baseline_w` (optional): Baseline equity weight (0.0 to 1.0)
- **`equity_tickers` (NEW, optional)**: Comma-separated list of equity ticker symbols

**Example:**
```
POST https://wealth-utility-live-production.up.railway.app/backtest/refresh?equity_tickers=RPV,QQQ&baseline_w=0.8
```

---

## Implementation Guide for Lovable

### Option 1: Add Ticker Selection UI (Recommended for Advanced Users)

Allow users to customize the ticker universe for backtests:

```typescript
// Add to backtest form
const [customTickers, setCustomTickers] = useState("");

// Build API URL with custom tickers
const buildBacktestUrl = (baselineW: number, customTickers?: string) => {
  const baseUrl = "https://wealth-utility-live-production.up.railway.app/backtest";
  const params = new URLSearchParams();

  params.append("baseline_w", baselineW.toString());

  if (customTickers && customTickers.trim()) {
    // Clean up ticker input: remove spaces, ensure comma-separated
    const cleanedTickers = customTickers.replace(/\s/g, "");
    params.append("equity_tickers", cleanedTickers);
  }
  // If no custom tickers provided, API uses default list

  return `${baseUrl}?${params.toString()}`;
};

// UI Component
<div>
  <label>Custom Equity Tickers (optional):</label>
  <input
    type="text"
    placeholder="SPY,QQQ,VTI (comma-separated)"
    value={customTickers}
    onChange={(e) => setCustomTickers(e.target.value)}
  />
  <small>Leave blank to use default ticker list</small>
</div>
```

### Option 2: Keep Simple (Use Defaults)

If you don't need custom ticker selection, **no changes required**. Just continue calling:

```typescript
const url = `https://wealth-utility-live-production.up.railway.app/backtest?baseline_w=${baselineW}`;
```

The API will automatically use the default ticker list.

---

## Error Handling

### New Error Responses for Invalid Tickers

**400 Bad Request - Empty Ticker List:**
```json
{
  "success": false,
  "error": "equity_tickers must contain at least one ticker symbol",
  "calculation_timestamp": "2026-01-12T..."
}
```

**400 Bad Request - Invalid Format:**
```json
{
  "success": false,
  "error": "Invalid equity_tickers format: ...",
  "calculation_timestamp": "2026-01-12T..."
}
```

**Handle in your code:**
```typescript
const response = await fetch(url);
const data = await response.json();

if (!data.success) {
  // Show error to user
  alert(data.error);
  return;
}

// Process successful backtest results
processBacktestData(data);
```

---

## Summary for Lovable Implementation

### What Changed:
1. `/backtest` endpoint now accepts optional `equity_tickers` parameter
2. Parameter format: comma-separated string (e.g., `"SPY,QQQ,VTI"`)
3. If not provided, API uses default ticker list (backward compatible)

### What Stays the Same:
- `/allocations` endpoint unchanged
- All existing query parameters work as before
- Response format unchanged
- No breaking changes to existing webapp functionality

### Recommended Approach:
- **Phase 1**: No changes needed - existing code continues to work
- **Phase 2** (optional): Add ticker selection UI for advanced users who want to customize backtest universe

### Testing:
```
# Test default behavior (should work with existing code)
GET /backtest?baseline_w=0.6

# Test custom tickers
GET /backtest?baseline_w=0.6&equity_tickers=SPY,QQQ,VTI

# Test with allocation profile
GET /allocations?profile=moderate
```

---

## Questions or Issues?

If the webapp encounters any issues with the new parameters:
1. Check that ticker string is comma-separated with no spaces
2. Verify at least one ticker is provided if using the parameter
3. Check API error response for specific error message
4. Fall back to default behavior by omitting `equity_tickers` parameter
