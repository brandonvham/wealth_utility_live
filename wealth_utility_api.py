"""
Wealth Utility - API Server
Flask REST API for serving allocation data to web applications
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import sys
import os

# Import the calculation logic from the production script
# (We'll import the main calculation function)
from wealth_utility_production import (
    calculate_current_allocations,
    run_backtest,
    START_DATE,
    EQUITY_TICKER,
    NON_EQUITY_TICKER,
    EQUITY_SLEEVE_METHOD,
    BASELINE_W,
    VALUE_DIAL_FRAC,
    MOM_BUMP_FRAC,
    BAND_MODE,
    BAND_ABS,
    RISK_DIAL_MODE,
    fetch_fmp_daily,
    monthly_from_daily_price,
    fetch_fred_dfii10,
    read_us_valuation_from_excel,
    build_equity_sleeve_monthly,
    _to_me,
    _to_frac,
    EQUITY_SLEEVE_LOOKBACK_M,
    EQUITY_SLEEVE_WARMUP_M,
    EQUITY_MAX_WEIGHT_PER_ASSET,
    EQUITY_COV_SHRINKAGE,
    EQUITY_RIDGE_LAMBDA,
    EQUITY_CLUSTERS,
    EQUITY_CLUSTER_RISK_BUDGETS,
    BENCHMARK_TICKER,
    MOM_LOOKBACK_M,
    RP_ANCHOR_MODE,
    RP_ANCHOR_FIXED,
    RISK_LOOKBACK_M,
    RISK_REF_METHOD,
    RISK_REF_FIXED,
    RISK_POWER,
    RISK_MULT_MIN,
    RISK_MULT_MAX,
    f_min,
    f_max,
    FMP_KEY,
    FRED_KEY,
    ECY_XLSX_PATH,
    ECY_SHEET,
    CAP_DEV_FRAC,
)
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Cache for storing the last calculation result
_cache = {
    'data': None,
    'timestamp': None,
    'cache_duration_minutes': 240  # Cache for 4 hours (reduced API calls)
}

# Separate cache for backtest results (more expensive computation)
# Now stores multiple cache entries keyed by parameters
_backtest_cache = {}
_backtest_cache_duration_minutes = 480  # Cache for 8 hours (very expensive)
_backtest_cache_max_entries = 20  # Limit cache size to prevent memory bloat


def _get_backtest_cache_key(start_date, end_date, baseline_w, equity_tickers):
    """Generate a unique cache key from backtest parameters."""
    # Use defaults if None
    start = start_date or START_DATE
    end = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    baseline = baseline_w if baseline_w is not None else BASELINE_W
    tickers = tuple(equity_tickers) if equity_tickers else tuple(EQUITY_TICKER if isinstance(EQUITY_TICKER, list) else [EQUITY_TICKER])

    return f"{start}|{end}|{baseline}|{tickers}"


def _get_cached_backtest(cache_key):
    """Retrieve cached backtest data if valid."""
    if cache_key not in _backtest_cache:
        return None

    entry = _backtest_cache[cache_key]
    age_minutes = (datetime.now() - entry['timestamp']).total_seconds() / 60

    if age_minutes < _backtest_cache_duration_minutes:
        print(f"[BACKTEST CACHE] Returning cached data for key: {cache_key[:50]}... (age: {age_minutes:.1f} minutes)")
        return entry['data']
    else:
        # Expired, remove from cache
        del _backtest_cache[cache_key]
        return None


def _set_cached_backtest(cache_key, data):
    """Store backtest data in cache with timestamp."""
    # Limit cache size - remove oldest entries if needed
    if len(_backtest_cache) >= _backtest_cache_max_entries:
        oldest_key = min(_backtest_cache.keys(), key=lambda k: _backtest_cache[k]['timestamp'])
        del _backtest_cache[oldest_key]
        print(f"[BACKTEST CACHE] Removed oldest cache entry to maintain size limit")

    _backtest_cache[cache_key] = {
        'data': data,
        'timestamp': datetime.now()
    }
    print(f"[BACKTEST CACHE] Cached data for key: {cache_key[:50]}... (total cached: {len(_backtest_cache)})")

# Risk profile definitions
RISK_PROFILES = {
    'all_equity': {
        'name': 'All Equity',
        'baseline_w': 1.0,
        'description': 'Maximum equity allocation'
    },
    'aggressive': {
        'name': 'Aggressive',
        'baseline_w': 0.80,
        'description': '80% equity baseline with tactical adjustments'
    },
    'moderate_aggressive': {
        'name': 'Moderate Aggressive',
        'baseline_w': 0.70,
        'description': '70% equity baseline with tactical adjustments'
    },
    'moderate': {
        'name': 'Moderate',
        'baseline_w': 0.60,
        'description': '60% equity baseline with tactical adjustments'
    },
    'moderate_conservative': {
        'name': 'Moderate Conservative',
        'baseline_w': 0.50,
        'description': '50% equity baseline with tactical adjustments'
    },
    'conservative': {
        'name': 'Conservative',
        'baseline_w': 0.40,
        'description': '40% equity baseline with tactical adjustments'
    }
}

# Allocations endpoint equity tickers (separate from backtest)
# This list is used ONLY for /allocations endpoint display
ALLOCATIONS_EQUITY_TICKERS = ["ACWI","COWG","COWZ","EDIV","IWR","JIVE","JMEE","JQUA","MGK","PDBC","REZ"]

# Minimum weight per asset for allocations endpoint (NOT used in backtest)
# Set to 0.0 for no minimum, or e.g. 0.05 for 5% minimum per ticker
ALLOCATIONS_MIN_WEIGHT_PER_ASSET = 0.04  # 5% minimum per ticker

# Dynamic Fixed Income configuration (for current allocations only)
FI_SECURITIES = ["PFFD", "FMHI", "VWOB", "SRLN", "ANGL", "ICVT"]
FI_RESERVES = ["TLT", "BIL"]
FI_TICKERS = FI_SECURITIES + FI_RESERVES
FI_MA_LOOKBACK = 10  # 10-month moving average


def _calculate_profile_weight(panel, baseline_w, rel_value, rp_anchor, r_mult):
    """
    Calculate target weight for a specific baseline_w (risk profile).

    Args:
        panel: DataFrame with market data
        baseline_w: Baseline equity weight for this profile
        rel_value: Relative valuation series
        rp_anchor: Risk premium anchor value
        r_mult: Risk multiplier series

    Returns:
        Series of target weights
    """
    # Normalize dials
    VALUE_DIAL = _to_frac(VALUE_DIAL_FRAC) or 0.0
    MOM_BUMP = _to_frac(MOM_BUMP_FRAC) or 0.0

    # Value center (based on this profile's baseline)
    value_bump = baseline_w * VALUE_DIAL * rel_value
    w_value = (baseline_w + value_bump).clip(0, 1)

    # Momentum bump (based on this profile's baseline)
    mom_bump = baseline_w * MOM_BUMP * panel["MOM_STATE"]
    w_uncapped = w_value + mom_bump

    # Band construction
    if BAND_MODE.lower() == "absolute":
        base_band = pd.Series(float(BAND_ABS), index=panel.index)
        band_width = base_band.copy()
        if RISK_DIAL_MODE.lower() == "band":
            band_width = (base_band * r_mult).clip(0.0, 1.0)
        lo = (w_value - band_width).astype(float)
        hi = (w_value + band_width).astype(float)
    elif BAND_MODE.lower() == "proportional":
        CAP_DEV = _to_frac(CAP_DEV_FRAC) or 0.0
        base_band_prop = (CAP_DEV * w_value).astype(float)
        band_width = base_band_prop.copy()
        if RISK_DIAL_MODE.lower() == "band":
            band_width = (base_band_prop * r_mult).clip(0.0, 1.0)
        lo = (w_value - band_width).astype(float)
        hi = (w_value + band_width).astype(float)

    w_capped = w_uncapped.clip(lower=lo, upper=hi)

    if RISK_DIAL_MODE.lower() == "scale":
        w_capped = (w_capped * r_mult).clip(0.0, 1.0)

    # Final target weight with dynamic f_max
    f_min_frac = _to_frac(f_min) or 0.0
    f_max_dynamic = min((baseline_w + 0.15) * 100, 100)
    f_max_frac = _to_frac(f_max_dynamic) or 1.0

    w_target = w_capped.clip(f_min_frac, f_max_frac).clip(0, 1)

    return w_target


def fetch_fmp_daily_unadjusted(symbol: str, start: str, end: str, apikey: str) -> pd.DataFrame:
    """
    Fetch daily unadjusted close prices from FMP (for MA signals).
    Similar to fetch_fmp_daily but forces use of unadjusted 'close' price.
    """
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
    params = {"from": start, "to": end, "apikey": apikey, "serietype": "line"}
    from wealth_utility_production import _HTTP, ensure_unique_index
    r = _HTTP.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    hist = js.get("historical", [])
    if not hist:
        raise ValueError(f"FMP returned no data for {symbol}")
    df = pd.DataFrame(hist)
    df["date"] = pd.to_datetime(df["date"])
    # Force use of unadjusted 'close' column only
    out = df[["date", "close"]].rename(columns={"close": "price"}).sort_values("date").set_index("date")
    out = out.asfreq("B").ffill()
    out = ensure_unique_index(out)
    return out


def build_fi_sleeve_monthly(fi_tickers, start, end, apikey, ma_lookback=10):
    """
    Build fixed income sleeve using moving average signals.
    Returns DataFrame with columns: mret (monthly return), tri (total return index), and weight columns for each FI ticker.

    This function is used ONLY for current allocations display, not for backtesting.

    Note: Uses UNADJUSTED prices for MA signals (reflects actual market price action),
          but ADJUSTED prices for returns (captures total return with dividends).
    """
    # Download FI data - adjusted prices for returns
    mlist = []
    price_list_adj = []
    price_list_unadj = []

    for s in fi_tickers:
        # Adjusted prices for returns
        d_adj = fetch_fmp_daily(s, start, end, apikey)
        m = monthly_from_daily_price(d_adj)
        mlist.append(m[["mret"]].rename(columns={"mret": s}))
        p_adj = d_adj["price"].resample("M").last().to_frame(name=s)
        p_adj.index = _to_me(p_adj.index)
        price_list_adj.append(p_adj)

        # Unadjusted prices for MA signals
        d_unadj = fetch_fmp_daily_unadjusted(s, start, end, apikey)
        p_unadj = d_unadj["price"].resample("M").last().to_frame(name=s)
        p_unadj.index = _to_me(p_unadj.index)
        price_list_unadj.append(p_unadj)

    R_full = pd.concat(mlist, axis=1).dropna(how="any").sort_index()
    P_full_adj = pd.concat(price_list_adj, axis=1).reindex(R_full.index).ffill()
    P_full_unadj = pd.concat(price_list_unadj, axis=1).reindex(R_full.index).ffill()

    # Calculate moving average signals using UNADJUSTED prices
    movingavg = P_full_unadj - P_full_unadj.rolling(ma_lookback).mean()

    # Determine weights at each rebalance
    weights_list = []
    for date in R_full.index:
        if date not in movingavg.index or movingavg.loc[date].isna().any():
            # Equal weight during warmup
            weights_list.append(np.ones(len(fi_tickers)) / len(fi_tickers))
            continue

        ma_signals = movingavg.loc[date]
        allocation = {}

        # Check securities (exclude TLT, BIL)
        securities = [t for t in fi_tickers if t not in ["TLT", "BIL"]]
        positive_securities = [t for t in securities if ma_signals[t] > 0]

        # Equal weight for positive securities
        wts = 1.0 / len(securities) if len(securities) > 0 else 0.0
        reserve_weight = (len(securities) - len(positive_securities)) / len(securities) if len(securities) > 0 else 1.0

        for ticker in fi_tickers:
            if ticker in positive_securities:
                allocation[ticker] = wts
            elif ticker == "TLT":
                if "TLT" in ma_signals and ma_signals["TLT"] > 0:
                    allocation[ticker] = reserve_weight
                else:
                    allocation[ticker] = 0.0
            elif ticker == "BIL":
                if "TLT" in ma_signals and ma_signals["TLT"] <= 0:
                    allocation[ticker] = reserve_weight
                else:
                    allocation[ticker] = 0.0
            else:
                allocation[ticker] = 0.0

        weights_list.append([allocation.get(t, 0.0) for t in fi_tickers])

    W_target = pd.DataFrame(weights_list, index=R_full.index, columns=fi_tickers)
    W_exec = W_target.shift(1).bfill()

    # Calculate sleeve return
    sleeve_ret = (W_exec * R_full).sum(axis=1)
    tri = (1.0 + sleeve_ret).cumprod()

    result = pd.DataFrame({"mret": sleeve_ret, "tri": tri}, index=R_full.index)
    for col in fi_tickers:
        result[col] = W_exec[col]

    return result


def get_current_allocations_json():
    """
    Calculate current allocations for all risk profiles and return as JSON-friendly dict.
    Uses caching to avoid recalculating on every request.
    """
    now = datetime.now()

    # Check if we have cached data that's still valid
    if _cache['data'] is not None and _cache['timestamp'] is not None:
        age_minutes = (now - _cache['timestamp']).total_seconds() / 60
        if age_minutes < _cache['cache_duration_minutes']:
            print(f"[CACHE] Returning cached data (age: {age_minutes:.1f} minutes)")
            return _cache['data']

    print("[CALC] Calculating fresh allocations for all risk profiles...")

    try:
        # Run the calculation
        start = START_DATE
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

        # Calculate warmup period needed for all rolling calculations
        warmup_months = max(EQUITY_SLEEVE_WARMUP_M, MOM_LOOKBACK_M, RISK_LOOKBACK_M)
        start_dt = pd.to_datetime(start).to_period("M").to_timestamp("M")
        warmup_dt = (start_dt - pd.DateOffset(months=warmup_months)).to_period("M").to_timestamp("M")
        warmup_start = warmup_dt.strftime("%Y-%m-%d")

        # Load data (common for all profiles)
        us_val = read_us_valuation_from_excel(ECY_XLSX_PATH, ECY_SHEET)
        tips = fetch_fred_dfii10(warmup_start, end, FRED_KEY)

        # Use ALLOCATIONS_EQUITY_TICKERS for allocations endpoint (separate from backtest)
        eq_m, _eq_W_target, _eq_W_exec, _eq_P = build_equity_sleeve_monthly(
            ALLOCATIONS_EQUITY_TICKERS, start, end, FMP_KEY,
            method=EQUITY_SLEEVE_METHOD,
            lookback_m=EQUITY_SLEEVE_LOOKBACK_M,
            warmup_m=EQUITY_SLEEVE_WARMUP_M,
            max_cap=EQUITY_MAX_WEIGHT_PER_ASSET,
            min_weight_per_asset=ALLOCATIONS_MIN_WEIGHT_PER_ASSET,
            cov_shrinkage=EQUITY_COV_SHRINKAGE,
            ridge_lambda=EQUITY_RIDGE_LAMBDA,
            clusters=EQUITY_CLUSTERS,
            cluster_risk_budgets=EQUITY_CLUSTER_RISK_BUDGETS,
            benchmark_symbol=BENCHMARK_TICKER,
        )

        # Build Fixed Income sleeve (for current allocations display only)
        print("[CALC] Building Dynamic Fixed Income sleeve...")
        fi_m = build_fi_sleeve_monthly(FI_TICKERS, warmup_start, end, FMP_KEY, ma_lookback=FI_MA_LOOKBACK)

        ne_m = monthly_from_daily_price(fetch_fmp_daily(NON_EQUITY_TICKER, warmup_start, end, FMP_KEY))

        # Common index (including FI data) - include warmup period for calculations
        idx = eq_m.index.intersection(ne_m.index).intersection(fi_m.index)

        # Align valuation & TIPS
        rp_used = us_val["RP_USED"].reindex(idx)
        if rp_used.isna().mean() > 0.50:
            rp_used = us_val["RP_USED"].reindex(idx, method="nearest", tolerance=pd.Timedelta(days=3))
        rp_used = pd.to_numeric(rp_used, errors="coerce").ffill().bfill()

        tips_m = tips["tips10"].reindex(idx)
        if tips_m.isna().mean() > 0.50:
            tips_m = tips["tips10"].reindex(idx, method="nearest", tolerance=pd.Timedelta(days=3))
        tips_m = pd.to_numeric(tips_m, errors="coerce").ffill().bfill()

        # Build panel
        panel = pd.DataFrame(index=idx)
        panel["eq_ret"], panel["eq_tri"] = eq_m.loc[idx,"mret"].astype(float), eq_m.loc[idx,"tri"].astype(float)
        panel["ne_ret"], panel["ne_tri"] = ne_m.loc[idx,"mret"].astype(float), ne_m.loc[idx,"tri"].astype(float)
        panel["RP_USED"] = rp_used
        panel["tips10"]  = tips_m
        panel["RP"]      = panel["RP_USED"] - panel["tips10"]

        # Momentum state
        maN = panel["eq_tri"].rolling(int(MOM_LOOKBACK_M)).mean()
        panel["MOM_STATE"] = np.where(panel["eq_tri"] > maN, 1, -1)
        panel.loc[maN.isna(), "MOM_STATE"] = 0

        # RP anchor
        if RP_ANCHOR_MODE == "median":
            rp_anchor = float(panel["RP"].median(skipna=True))
        elif RP_ANCHOR_MODE == "mean":
            rp_anchor = float(panel["RP"].mean(skipna=True))
        else:
            rp_anchor = float(RP_ANCHOR_FIXED)

        # Relative value (common for all profiles)
        rel_value = (panel["RP"] / rp_anchor) - 1.0

        # Risk dial (common for all profiles)
        equity_ret = panel["eq_ret"].copy()
        realized_vol = equity_ret.rolling(int(RISK_LOOKBACK_M)).std(ddof=0) * np.sqrt(12.0)

        if RISK_REF_METHOD == "fixed":
            vol_ref = pd.Series(RISK_REF_FIXED, index=panel.index, dtype=float)
        elif RISK_REF_METHOD == "rolling_median":
            vol_ref = realized_vol.rolling(120, min_periods=12).median()
        else:
            const_ref = realized_vol.median(skipna=True)
            if not np.isfinite(const_ref) or const_ref <= 0:
                const_ref = 0.15
            vol_ref = pd.Series(const_ref, index=panel.index, dtype=float)

        with np.errstate(divide='ignore', invalid='ignore'):
            r_raw = (vol_ref / realized_vol)
        r_raw = r_raw.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=1e-6, upper=1e6)
        r_mult = (r_raw ** RISK_POWER).clip(lower=RISK_MULT_MIN, upper=RISK_MULT_MAX)

        # Get latest month info
        latest_date = panel.index[-1]
        W_exec_eq = _eq_W_exec.reindex(panel.index).ffill()
        latest_sleeve_weights = W_exec_eq.loc[latest_date]

        # Get latest FI sleeve weights
        W_exec_fi = fi_m[FI_TICKERS].reindex(panel.index).ffill()
        latest_fi_weights = W_exec_fi.loc[latest_date]

        # Calculate allocations for each risk profile
        profiles = {}

        for profile_key, profile_config in RISK_PROFILES.items():
            baseline_w = profile_config['baseline_w']

            # Calculate target weight for this profile
            w_target = _calculate_profile_weight(panel, baseline_w, rel_value, rp_anchor, r_mult)

            # Get latest allocation
            latest_target_equity_weight = float(w_target.loc[latest_date])
            latest_target_safe_weight = 1.0 - latest_target_equity_weight

            # Calculate f_max for this profile
            f_max_dynamic = min((baseline_w + 0.15) * 100, 100)

            # Build allocation JSON for this profile
            allocations = []

            # Add equity allocations (using ALLOCATIONS_EQUITY_TICKERS)
            if isinstance(ALLOCATIONS_EQUITY_TICKERS, str):
                equity_alloc = latest_target_equity_weight * 1.0
                allocations.append({
                    "ticker": ALLOCATIONS_EQUITY_TICKERS,
                    "asset_class": "equity",
                    "weight": round(equity_alloc, 4),
                    "weight_pct": f"{equity_alloc:.2%}"
                })
            else:
                for ticker in ALLOCATIONS_EQUITY_TICKERS:
                    sleeve_weight = float(latest_sleeve_weights[ticker])
                    equity_alloc = latest_target_equity_weight * sleeve_weight
                    allocations.append({
                        "ticker": ticker,
                        "asset_class": "equity",
                        "weight": round(equity_alloc, 4),
                        "weight_pct": f"{equity_alloc:.2%}"
                    })

            # Add dynamic FI allocations (replaces single BIL entry)
            for ticker in FI_TICKERS:
                fi_sleeve_weight = float(latest_fi_weights[ticker])
                fi_alloc = latest_target_safe_weight * fi_sleeve_weight
                allocations.append({
                    "ticker": ticker,
                    "asset_class": "fixed_income",
                    "weight": round(fi_alloc, 4),
                    "weight_pct": f"{fi_alloc:.2%}"
                })

            # Store profile data
            profiles[profile_key] = {
                "name": profile_config['name'],
                "description": profile_config['description'],
                "baseline_w": baseline_w,
                "f_max": round(f_max_dynamic / 100, 4),
                "total_equity": round(latest_target_equity_weight, 4),
                "total_equity_pct": f"{latest_target_equity_weight:.2%}",
                "total_safe": round(latest_target_safe_weight, 4),
                "total_safe_pct": f"{latest_target_safe_weight:.2%}",
                "allocations": allocations
            }

        # Build response
        result = {
            "success": True,
            "calculation_date": datetime.now().isoformat(),
            "allocation_date": latest_date.strftime("%Y-%m-%d"),
            "profiles": profiles,
            "strategy_params": {
                "sleeve_method": EQUITY_SLEEVE_METHOD,
                "band_mode": BAND_MODE,
                "risk_dial_mode": RISK_DIAL_MODE,
                "value_dial": VALUE_DIAL_FRAC,
                "momentum_dial": MOM_BUMP_FRAC,
            }
        }

        # Update cache
        _cache['data'] = result
        _cache['timestamp'] = now

        return result

    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "calculation_date": datetime.now().isoformat()
        }


@app.route('/')
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        "name": "Wealth Utility API",
        "version": "2.6.0",
        "endpoints": {
            "/": "This help page",
            "/allocations": "Get current portfolio allocations for all risk profiles with Dynamic FI sleeve (GET). Query param: profile (optional, returns single profile). Uses separate ALLOCATIONS_EQUITY_TICKERS list",
            "/allocations?profile=moderate": "Get allocations for a specific risk profile (GET). Valid profiles: all_equity, moderate_aggressive, moderate, moderate_conservative, conservative. Note: Uses Dynamic Fixed Income allocation instead of BIL",
            "/allocations/refresh": "Force refresh allocations (POST)",
            "/backtest": "Run historical backtest with performance metrics (GET). Query params: start_date, end_date, baseline_w (0.0-1.0, sets f_max=baseline_w+15%), equity_tickers (comma-separated list), force_refresh",
            "/backtest/refresh": "Force refresh backtest (POST). Query params: start_date, end_date, baseline_w (0.0-1.0, sets f_max=baseline_w+15%), equity_tickers (comma-separated list)",
            "/config": "Get strategy configuration (GET)",
            "/health": "Health check endpoint (GET)"
        },
        "risk_profiles": {
            profile_key: {
                "name": config['name'],
                "baseline_w": config['baseline_w'],
                "f_max": round(min((config['baseline_w'] + 0.15) * 100, 100) / 100, 4),
                "description": config['description']
            }
            for profile_key, config in RISK_PROFILES.items()
        },
        "status": "online"
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/allocations', methods=['GET'])
def get_allocations():
    """
    Get current portfolio allocations for all risk profiles or a specific profile.

    Query parameters:
    - profile: Optional profile key (all_equity, moderate_aggressive, moderate,
               moderate_conservative, conservative). If not specified, returns all profiles.

    Returns cached data if available and fresh.
    """
    result = get_current_allocations_json()

    # Check if user requested a specific profile
    profile_key = request.args.get('profile', None)

    if profile_key:
        # Validate profile exists
        if profile_key not in RISK_PROFILES:
            return jsonify({
                "success": False,
                "error": f"Invalid profile '{profile_key}'. Valid profiles: {list(RISK_PROFILES.keys())}",
                "available_profiles": list(RISK_PROFILES.keys())
            }), 400

        # Return only the requested profile
        if result['success'] and 'profiles' in result:
            single_profile_result = {
                "success": True,
                "calculation_date": result['calculation_date'],
                "allocation_date": result['allocation_date'],
                "profile": result['profiles'][profile_key],
                "strategy_params": result['strategy_params']
            }
            return jsonify(single_profile_result)

    return jsonify(result)


@app.route('/allocations/refresh', methods=['POST'])
def refresh_allocations():
    """
    Force refresh allocations (bypass cache).
    """
    # Clear cache
    _cache['data'] = None
    _cache['timestamp'] = None

    result = get_current_allocations_json()
    return jsonify(result)


@app.route('/config', methods=['GET'])
def get_config():
    """Get current strategy configuration"""
    return jsonify({
        "equity_tickers": EQUITY_TICKER if isinstance(EQUITY_TICKER, list) else [EQUITY_TICKER],
        "non_equity_ticker": NON_EQUITY_TICKER,
        "benchmark": BENCHMARK_TICKER,
        "sleeve_method": EQUITY_SLEEVE_METHOD,
        "band_mode": BAND_MODE,
        "band_absolute": BAND_ABS,
        "risk_dial_mode": RISK_DIAL_MODE,
        "value_dial": VALUE_DIAL_FRAC,
        "momentum_dial": MOM_BUMP_FRAC,
        "baseline_equity": BASELINE_W
    })


@app.route('/backtest', methods=['GET'])
def get_backtest():
    """
    Run full historical backtest and return performance metrics.

    Query parameters:
    - start_date: Optional start date (YYYY-MM-DD format)
    - end_date: Optional end date (YYYY-MM-DD format)
    - baseline_w: Optional baseline equity weight (0.0 to 1.0, defaults to BASELINE_W constant)
                  Note: f_max is automatically set to baseline_w + 15% (capped at 100%)
    - equity_tickers: Optional comma-separated list of equity tickers (e.g., "RPV,QQQ,VTI")
                      Defaults to EQUITY_TICKER constant from production
    - force_refresh: Set to 'true' to bypass cache

    Returns cached data if available and fresh, unless force_refresh=true.
    """
    now = datetime.now()

    # Get optional date parameters
    start_date = request.args.get('start_date', None)
    end_date = request.args.get('end_date', None)

    # Get optional baseline_w parameter and validate
    baseline_w_str = request.args.get('baseline_w', None)
    baseline_w = None
    if baseline_w_str is not None:
        try:
            baseline_w = float(baseline_w_str)
            if not (0.0 <= baseline_w <= 1.0):
                return jsonify({
                    "success": False,
                    "error": f"baseline_w must be between 0.0 and 1.0, got {baseline_w}",
                    "calculation_timestamp": now.isoformat()
                }), 400
        except ValueError:
            return jsonify({
                "success": False,
                "error": f"baseline_w must be a number, got '{baseline_w_str}'",
                "calculation_timestamp": now.isoformat()
            }), 400

    # Get optional equity_tickers parameter (comma-separated)
    equity_tickers_str = request.args.get('equity_tickers', None)
    equity_tickers = None
    if equity_tickers_str is not None:
        try:
            # Split by comma and strip whitespace
            equity_tickers = [ticker.strip() for ticker in equity_tickers_str.split(',') if ticker.strip()]
            if len(equity_tickers) == 0:
                return jsonify({
                    "success": False,
                    "error": "equity_tickers must contain at least one ticker symbol",
                    "calculation_timestamp": now.isoformat()
                }), 400
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Invalid equity_tickers format: {str(e)}",
                "calculation_timestamp": now.isoformat()
            }), 400

    # Generate cache key from parameters
    cache_key = _get_backtest_cache_key(start_date, end_date, baseline_w, equity_tickers)

    # Check for force refresh parameter
    force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'

    # Check if we have cached data that's still valid
    if not force_refresh:
        cached_result = _get_cached_backtest(cache_key)
        if cached_result is not None:
            return jsonify(cached_result)

    print("[BACKTEST] Calculating fresh backtest...")

    try:
        # Run backtest
        result = run_backtest(start_date=start_date, end_date=end_date, baseline_w=baseline_w, equity_tickers=equity_tickers)

        # Add metadata
        result['success'] = True
        result['calculation_timestamp'] = now.isoformat()

        # Update cache
        _set_cached_backtest(cache_key, result)

        return jsonify(result)

    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "calculation_timestamp": now.isoformat()
        }), 500


@app.route('/backtest/refresh', methods=['POST'])
def refresh_backtest():
    """
    Force refresh backtest (bypass cache).

    Query parameters:
    - start_date: Optional start date (YYYY-MM-DD format)
    - end_date: Optional end date (YYYY-MM-DD format)
    - baseline_w: Optional baseline equity weight (0.0 to 1.0, defaults to BASELINE_W constant)
                  Note: f_max is automatically set to baseline_w + 15% (capped at 100%)
    - equity_tickers: Optional comma-separated list of equity tickers (e.g., "RPV,QQQ,VTI")
                      Defaults to EQUITY_TICKER constant from production
    """
    # Run fresh backtest
    now = datetime.now()
    print("[BACKTEST] Force refresh requested...")

    try:
        # Get optional date parameters from query string
        start_date = request.args.get('start_date', None)
        end_date = request.args.get('end_date', None)

        # Get optional baseline_w parameter and validate
        baseline_w_str = request.args.get('baseline_w', None)
        baseline_w = None
        if baseline_w_str is not None:
            try:
                baseline_w = float(baseline_w_str)
                if not (0.0 <= baseline_w <= 1.0):
                    return jsonify({
                        "success": False,
                        "error": f"baseline_w must be between 0.0 and 1.0, got {baseline_w}",
                        "calculation_timestamp": now.isoformat()
                    }), 400
            except ValueError:
                return jsonify({
                    "success": False,
                    "error": f"baseline_w must be a number, got '{baseline_w_str}'",
                    "calculation_timestamp": now.isoformat()
                }), 400

        # Get optional equity_tickers parameter (comma-separated)
        equity_tickers_str = request.args.get('equity_tickers', None)
        equity_tickers = None
        if equity_tickers_str is not None:
            try:
                # Split by comma and strip whitespace
                equity_tickers = [ticker.strip() for ticker in equity_tickers_str.split(',') if ticker.strip()]
                if len(equity_tickers) == 0:
                    return jsonify({
                        "success": False,
                        "error": "equity_tickers must contain at least one ticker symbol",
                        "calculation_timestamp": now.isoformat()
                    }), 400
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": f"Invalid equity_tickers format: {str(e)}",
                    "calculation_timestamp": now.isoformat()
                }), 400

        # Generate cache key and clear only this specific entry
        cache_key = _get_backtest_cache_key(start_date, end_date, baseline_w, equity_tickers)
        if cache_key in _backtest_cache:
            del _backtest_cache[cache_key]
            print(f"[BACKTEST CACHE] Cleared cache for key: {cache_key[:50]}...")

        result = run_backtest(start_date=start_date, end_date=end_date, baseline_w=baseline_w, equity_tickers=equity_tickers)

        # Add metadata
        result['success'] = True
        result['calculation_timestamp'] = now.isoformat()

        # Update cache
        _set_cached_backtest(cache_key, result)

        return jsonify(result)

    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "calculation_timestamp": now.isoformat()
        }), 500


if __name__ == '__main__':
    # Run the Flask development server
    # For production, use a WSGI server like Gunicorn
    print("=" * 80)
    print("WEALTH UTILITY API SERVER v2.6.0")
    print("=" * 80)
    print("Starting Flask development server...")
    print("API will be available at: http://localhost:5000")
    print()
    print("Endpoints:")
    print("  GET  http://localhost:5000/")
    print("  GET  http://localhost:5000/allocations")
    print("       Returns all 5 risk profiles (all_equity, moderate_aggressive, moderate,")
    print("       moderate_conservative, conservative) with their allocations")
    print("       Uses ALLOCATIONS_EQUITY_TICKERS (separate from backtest)")
    print("  GET  http://localhost:5000/allocations?profile=moderate")
    print("       Returns allocation for a specific risk profile")
    print("  POST http://localhost:5000/allocations/refresh")
    print("  GET  http://localhost:5000/backtest?baseline_w=0.6&equity_tickers=RPV,QQQ,VTI")
    print("       (Note: f_max automatically set to baseline_w + 15%)")
    print("       (Note: equity_tickers is comma-separated, defaults to EQUITY_TICKER constant)")
    print("  POST http://localhost:5000/backtest/refresh")
    print("  GET  http://localhost:5000/config")
    print("  GET  http://localhost:5000/health")
    print("=" * 80)
    print()

    app.run(debug=True, host='0.0.0.0', port=5000)
