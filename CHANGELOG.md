# Changelog

All notable changes to the Wealth Utility API project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [2.6.0] - 2026-01-13 - Critical Fixes & Enhancements

### Fixed

#### 🎯 Start Date Alignment Issue
- **Problem**: Benchmark started one month before wealth utility strategy, causing misaligned performance metrics (223 vs 224 months)
- **Root Cause**: Warmup period data was not properly handled across all data sources
- **Solution**:
  - Added automatic warmup period calculation (13 months) for all rolling calculations
  - Fetch all data (equity, non-equity, benchmark, TIPS) from warmup start date
  - Perform all calculations with full warmup data
  - Trim results to user-specified START_DATE after calculations complete
  - Drop first row if execution weight is invalid (NaN or 0) to ensure clean alignment
- **Impact**: Both strategy and benchmark now start on the same date with identical month counts
- **Files Modified**:
  - `wealth_utility_production.py` (both API and non-API versions)
  - `wealth_utility_api.py` (both versions)

#### 🔄 NAV Calculation Fix
- **Problem**: NAVs calculated before trimming caused compounding issues
- **Solution**: Recalculate NAVs from trimmed returns to ensure all start at 1.0 from same date
- **Impact**: Perfect alignment of portfolio, benchmark, and allocation benchmark NAVs
- **Location**: `wealth_utility_production.py` lines 1084-1095

#### 💾 Cache Parameter Issue
- **Problem**: Backtest cache was global - different parameters returned wrong cached data
- **Example**: Request with `start_date=2010` would return cached data from `start_date=2007`
- **Solution**: Implemented parameter-specific cache with unique keys
- **Features**:
  - Cache key includes: start_date, end_date, baseline_w, equity_tickers
  - Maximum 20 cache entries with automatic oldest-entry removal
  - Force refresh now only clears specific parameter cache
- **Impact**: Each unique backtest configuration gets its own cached result
- **Location**: `wealth_utility_api.py` lines 72-119, 628-698, 710-786

### Added

#### 📊 Fixed Income Dual Price Calculation
- **Feature**: FI sleeve now uses unadjusted prices for MA signals and adjusted prices for returns
- **Rationale**:
  - MA signals based on actual market price action (what traders see)
  - Returns based on total return including dividends (accurate performance)
- **Implementation**:
  - New function: `fetch_fmp_daily_unadjusted()` (line 219)
  - Updated: `build_fi_sleeve_monthly()` to fetch both price types (line 242)
  - MA signals use unadjusted prices (line 277)
  - Returns use adjusted prices (line 259-264)
- **Impact**: More realistic technical signals while maintaining accurate return calculations
- **Location**: `wealth_utility_api.py` lines 219-277

#### ⚖️ Minimum Weight Per Asset Configuration
- **Feature**: Added configurable minimum weight per equity ticker for allocations endpoint
- **Separation**: Minimum only applies to allocations endpoint, NOT backtest
- **Configuration**: `ALLOCATIONS_MIN_WEIGHT_PER_ASSET = 0.04` (4% minimum per ticker)
- **Logic**:
  - Weights below minimum are raised to minimum
  - All weights renormalized to sum to 100%
  - Applied after optimization but before execution lag
- **Use Case**: Ensures all tickers in allocations have meaningful positions
- **Backtest Behavior**: Continues using no minimum constraint for historical accuracy
- **Files Modified**:
  - `wealth_utility_production.py` - Added `min_weight_per_asset` parameter (line 465)
  - `wealth_utility_production.py` - Applied constraint logic (lines 565-573)
  - `wealth_utility_api.py` - Added configuration constant (line 156)
  - `wealth_utility_api.py` - Passed to allocations call (line 372)
- **Location**: Both production files lines 565-573, API file lines 156, 372

### Technical Details

#### Data Flow Changes

**Before:**
```
1. Fetch equity data with warmup (internal to equity sleeve)
2. Fetch non-equity & benchmark from START_DATE (no warmup)
3. Calculate momentum/risk dials (insufficient history → NaN)
4. Trim to START_DATE
5. Calculate NAVs
→ Result: Misaligned data, 1-month discrepancy
```

**After:**
```
1. Calculate warmup period (13 months)
2. Fetch ALL data from warmup_start
3. Calculate momentum/risk dials with full history
4. Trim ALL data to START_DATE
5. Drop first row if w_exec invalid
6. Recalculate NAVs from trimmed returns
→ Result: Perfect alignment, identical month counts
```

#### Cache Structure Changes

**Before:**
```python
_backtest_cache = {
    'data': None,
    'timestamp': None,
    'cache_duration_minutes': 480
}
# Single global cache for all requests
```

**After:**
```python
_backtest_cache = {}  # Dictionary of caches
# Each unique parameter combination gets own cache entry
# Cache key: "start|end|baseline|tickers"
# Max 20 entries with LRU removal
```

### Configuration Changes

#### New Constants Added

**`wealth_utility_api.py`:**
```python
# Line 156
ALLOCATIONS_MIN_WEIGHT_PER_ASSET = 0.04  # 4% minimum per ticker

# Lines 75-76
_backtest_cache_duration_minutes = 480
_backtest_cache_max_entries = 20
```

#### Function Signatures Updated

**`build_equity_sleeve_monthly()`:**
```python
# Added parameter (line 465 in both production files)
min_weight_per_asset: float = 0.0
```

### Breaking Changes

⚠️ None - All changes are backward compatible with default parameters

### Migration Notes

- Railway deployment will need restart to clear old global cache
- No code changes required in calling applications
- All new features use sensible defaults
- Existing API calls will work identically

### Performance Impact

- **Cache**: Slight memory increase (max 20 backtest entries vs 1)
- **FI Sleeve**: Additional API calls for unadjusted prices (~50% more FI data requests)
- **Min Weight**: Negligible computational overhead
- **Warmup**: Initial calculation slower (more data fetched), but results in cleaner data

### Testing Checklist

- [x] Start date alignment verified (both strategies start same date)
- [x] Month counts match (224 = 224)
- [x] Parameter-specific cache working (different params = different cache)
- [x] FI sleeve using correct price types (unadj for MA, adj for returns)
- [x] Minimum weight applied correctly (allocations only, not backtest)
- [x] NAVs all start at 1.0 on same date
- [x] No first-month zero allocations
- [x] Force refresh clears correct cache entry

---

## [2.5.0] - Prior to 2026-01-13

Previous stable version with:
- Multi-profile risk allocations endpoint
- Dynamic Fixed Income sleeve with MA signals
- Backtest endpoint with customizable parameters
- 8-hour backtest cache (global)
- 4-hour allocation cache
- Equity sleeve optimization (max sharpe, beta max, optimized, equal weight)
- Value and momentum dials
- Risk-adjusted position sizing

---

## Future Enhancements (Not Yet Implemented)

- [ ] Redis cache for multi-instance deployments
- [ ] WebSocket support for real-time updates
- [ ] Historical allocation download endpoint
- [ ] Configurable warmup period via API parameter
- [ ] Batch backtest endpoint for multiple configurations
- [ ] Dynamic min/max weight constraints via API parameters

---

**Generated with Claude Code** 🤖
