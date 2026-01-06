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
    'cache_duration_minutes': 60  # Cache for 1 hour
}

# Separate cache for backtest results (more expensive computation)
_backtest_cache = {
    'data': None,
    'timestamp': None,
    'cache_duration_minutes': 120  # Cache for 2 hours
}


def get_current_allocations_json():
    """
    Calculate current allocations and return as JSON-friendly dict.
    Uses caching to avoid recalculating on every request.
    """
    now = datetime.now()

    # Check if we have cached data that's still valid
    if _cache['data'] is not None and _cache['timestamp'] is not None:
        age_minutes = (now - _cache['timestamp']).total_seconds() / 60
        if age_minutes < _cache['cache_duration_minutes']:
            print(f"[CACHE] Returning cached data (age: {age_minutes:.1f} minutes)")
            return _cache['data']

    print("[CALC] Calculating fresh allocations...")

    try:
        # Run the calculation
        start = START_DATE
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

        # Load data
        us_val = read_us_valuation_from_excel(ECY_XLSX_PATH, ECY_SHEET)
        tips = fetch_fred_dfii10(start, end, FRED_KEY)

        eq_m, _eq_W_target, _eq_W_exec, _eq_P = build_equity_sleeve_monthly(
            EQUITY_TICKER, start, end, FMP_KEY,
            method=EQUITY_SLEEVE_METHOD,
            lookback_m=EQUITY_SLEEVE_LOOKBACK_M,
            warmup_m=EQUITY_SLEEVE_WARMUP_M,
            max_cap=EQUITY_MAX_WEIGHT_PER_ASSET,
            cov_shrinkage=EQUITY_COV_SHRINKAGE,
            ridge_lambda=EQUITY_RIDGE_LAMBDA,
            clusters=EQUITY_CLUSTERS,
            cluster_risk_budgets=EQUITY_CLUSTER_RISK_BUDGETS,
            benchmark_symbol=BENCHMARK_TICKER,
        )

        ne_m = monthly_from_daily_price(fetch_fmp_daily(NON_EQUITY_TICKER, start, end, FMP_KEY))

        # Common index
        idx = eq_m.index.intersection(ne_m.index)
        idx = idx[idx >= pd.to_datetime(START_DATE).to_period("M").to_timestamp("M")]

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

        # Normalize dials
        VALUE_DIAL = _to_frac(VALUE_DIAL_FRAC) or 0.0
        MOM_BUMP = _to_frac(MOM_BUMP_FRAC) or 0.0

        # RP anchor
        if RP_ANCHOR_MODE == "median":
            rp_anchor = float(panel["RP"].median(skipna=True))
        elif RP_ANCHOR_MODE == "mean":
            rp_anchor = float(panel["RP"].mean(skipna=True))
        else:
            rp_anchor = float(RP_ANCHOR_FIXED)

        # Value center
        rel_value = (panel["RP"] / rp_anchor) - 1.0
        value_bump = BASELINE_W * VALUE_DIAL * rel_value
        w_value = (BASELINE_W + value_bump).clip(0, 1)

        # Momentum bump
        mom_bump = BASELINE_W * MOM_BUMP * panel["MOM_STATE"]
        w_uncapped = w_value + mom_bump

        # Risk dial
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

        # Final target weight
        f_min_frac = _to_frac(f_min) or 0.0
        f_max_frac = _to_frac(f_max) or 1.0
        panel["w_target"] = w_capped.clip(f_min_frac, f_max_frac).clip(0, 1)

        # Get latest month's allocation
        latest_date = panel.index[-1]
        latest_target_equity_weight = float(panel.loc[latest_date, "w_target"])
        latest_target_safe_weight = 1.0 - latest_target_equity_weight

        # Get within-sleeve weights
        W_exec_eq = _eq_W_exec.reindex(panel.index).ffill()
        latest_sleeve_weights = W_exec_eq.loc[latest_date]

        # Build allocation JSON
        allocations = []

        if isinstance(EQUITY_TICKER, str):
            equity_alloc = latest_target_equity_weight * 1.0
            allocations.append({
                "ticker": EQUITY_TICKER,
                "asset_class": "equity",
                "weight": round(equity_alloc, 4),
                "weight_pct": f"{equity_alloc:.2%}"
            })
        else:
            for ticker in EQUITY_TICKER:
                sleeve_weight = float(latest_sleeve_weights[ticker])
                equity_alloc = latest_target_equity_weight * sleeve_weight
                allocations.append({
                    "ticker": ticker,
                    "asset_class": "equity",
                    "weight": round(equity_alloc, 4),
                    "weight_pct": f"{equity_alloc:.2%}"
                })

        allocations.append({
            "ticker": NON_EQUITY_TICKER,
            "asset_class": "fixed_income",
            "weight": round(latest_target_safe_weight, 4),
            "weight_pct": f"{latest_target_safe_weight:.2%}"
        })

        # Build response
        result = {
            "success": True,
            "calculation_date": datetime.now().isoformat(),
            "allocation_date": latest_date.strftime("%Y-%m-%d"),
            "allocations": allocations,
            "summary": {
                "total_equity": round(latest_target_equity_weight, 4),
                "total_equity_pct": f"{latest_target_equity_weight:.2%}",
                "total_fixed_income": round(latest_target_safe_weight, 4),
                "total_fixed_income_pct": f"{latest_target_safe_weight:.2%}",
            },
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
        "version": "2.0.0",
        "endpoints": {
            "/": "This help page",
            "/allocations": "Get current portfolio allocations (GET)",
            "/allocations/refresh": "Force refresh allocations (POST)",
            "/backtest": "Run historical backtest with performance metrics (GET)",
            "/backtest/refresh": "Force refresh backtest (POST)",
            "/config": "Get strategy configuration (GET)",
            "/health": "Health check endpoint (GET)"
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
    Get current portfolio allocations.
    Returns cached data if available and fresh.
    """
    result = get_current_allocations_json()
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
    - force_refresh: Set to 'true' to bypass cache

    Returns cached data if available and fresh, unless force_refresh=true.
    """
    now = datetime.now()

    # Check for force refresh parameter
    force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'

    # Check if we have cached data that's still valid
    if not force_refresh and _backtest_cache['data'] is not None and _backtest_cache['timestamp'] is not None:
        age_minutes = (now - _backtest_cache['timestamp']).total_seconds() / 60
        if age_minutes < _backtest_cache['cache_duration_minutes']:
            print(f"[BACKTEST CACHE] Returning cached data (age: {age_minutes:.1f} minutes)")
            return jsonify(_backtest_cache['data'])

    print("[BACKTEST] Calculating fresh backtest...")

    try:
        # Get optional date parameters
        start_date = request.args.get('start_date', None)
        end_date = request.args.get('end_date', None)

        # Run backtest
        result = run_backtest(start_date=start_date, end_date=end_date)

        # Add metadata
        result['success'] = True
        result['calculation_timestamp'] = now.isoformat()

        # Update cache
        _backtest_cache['data'] = result
        _backtest_cache['timestamp'] = now

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
    """
    # Clear cache
    _backtest_cache['data'] = None
    _backtest_cache['timestamp'] = None

    # Run fresh backtest
    now = datetime.now()
    print("[BACKTEST] Force refresh requested...")

    try:
        # Get optional date parameters from query string
        start_date = request.args.get('start_date', None)
        end_date = request.args.get('end_date', None)

        result = run_backtest(start_date=start_date, end_date=end_date)

        # Add metadata
        result['success'] = True
        result['calculation_timestamp'] = now.isoformat()

        # Update cache
        _backtest_cache['data'] = result
        _backtest_cache['timestamp'] = now

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
    print("WEALTH UTILITY API SERVER v2.0.0")
    print("=" * 80)
    print("Starting Flask development server...")
    print("API will be available at: http://localhost:5000")
    print()
    print("Endpoints:")
    print("  GET  http://localhost:5000/")
    print("  GET  http://localhost:5000/allocations")
    print("  POST http://localhost:5000/allocations/refresh")
    print("  GET  http://localhost:5000/backtest")
    print("  POST http://localhost:5000/backtest/refresh")
    print("  GET  http://localhost:5000/config")
    print("  GET  http://localhost:5000/health")
    print("=" * 80)
    print()

    app.run(debug=True, host='0.0.0.0', port=5000)
