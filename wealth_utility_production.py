"""
Wealth Utility - Production Scheduler
Runs on the last trading day of every month at 5 PM CT
Outputs current month's allocation percentages
"""

import os
import sys
import warnings
from typing import Optional, List, Dict, Iterable, Mapping, Tuple
from datetime import datetime, time
import pytz

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    # python-dotenv not installed, will use system environment variables only
    pass

warnings.filterwarnings("ignore")

# ===================== USER PARAMETERS =====================
# API Keys - Load from environment variables (from .env file or system)
FMP_KEY  = os.getenv("FMP_KEY")
FRED_KEY = os.getenv("FRED_API_KEY")

# Validate API keys are set
if not FMP_KEY:
    raise ValueError(
        "FMP_KEY environment variable is not set.\n"
        "Please create a .env file (copy from .env.example) and add your API key.\n"
        "Get your key at: https://financialmodelingprep.com/"
    )
if not FRED_KEY:
    raise ValueError(
        "FRED_API_KEY environment variable is not set.\n"
        "Please create a .env file (copy from .env.example) and add your API key.\n"
        "Get your key at: https://fred.stlouisfed.org/"
    )

START_DATE = "2007-01-01"

# Excel file path - use relative path for deployment
ECY_XLSX_PATH = os.getenv("ECY_XLSX_PATH", "ecy4.xlsx")
ECY_SHEET     = "cape"

# Equity tickers
EQUITY_TICKER = ["IVV","RPG","IWR","EFA","QQQ","EEM","VTI","DBC"]
NON_EQUITY_TICKER = "BIL"
BENCHMARK_TICKER  = "SPY"

# Sleeve method
EQUITY_SLEEVE_METHOD = "max_sharpe"
EQUI1 = 12
EQUITY_SLEEVE_LOOKBACK_M     = EQUI1
EQUITY_SLEEVE_WARMUP_M       = EQUITY_SLEEVE_LOOKBACK_M + 1
EQUITY_MAX_WEIGHT_PER_ASSET  = 0.20
EQUITY_COV_SHRINKAGE         = "ledoit_wolf"
EQUITY_RIDGE_LAMBDA          = 0.001

# Max Sharpe optimization controls
EQUITY_MAX_SHARPE_RF_RATE = 0.0
EQUITY_MAX_SHARPE_RETURN_EST = "ema"
EQUITY_MAX_SHARPE_EMA_SPAN = 12
EQUITY_MAX_SHARPE_MIN_WEIGHT = 0.00

EQUITY_CLUSTERS: Optional[Dict[str, List[str]]] = None
EQUITY_CLUSTER_RISK_BUDGETS: Optional[Dict[str, float]] = None

# Dials & bounds
BASELINE_W      = 1.0
MOM_LOOKBACK_M  = 12
VALUE_DIAL_FRAC = 25
MOM_BUMP_FRAC   = 75
CAP_DEV_FRAC    = 50

RP_ANCHOR_MODE  = "fixed"
RP_ANCHOR_FIXED = 0.4

BAND_MODE = "absolute"
BAND_ABS  = 1.0

RISK_DIAL_MODE   = "band"
RISK_LOOKBACK_M  = 12
RISK_REF_METHOD  = "long_run"
RISK_REF_FIXED   = 0.15
RISK_POWER       = 1.00
RISK_MULT_MIN    = 0.30
RISK_MULT_MAX    = 1.0

TURNOVER_BPS    = 0.0
f_min = 0
f_max = 100

# ===================== HELPERS =====================
def _end_date():
    return pd.Timestamp.today().strftime("%Y-%m-%d")

def _to_me(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.to_datetime(idx).to_period("M").to_timestamp("M")

def ensure_unique_index(obj):
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.loc[~obj.index.duplicated(keep="last")]
    return obj

def parse_numeric(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.replace("%","", regex=False).str.replace(",","", regex=False)
    x = pd.to_numeric(x, errors="coerce")
    if x.dropna().median() > 1.5:
        x = x / 100.0
    return x

def _to_frac(x):
    if x is None:
        return None
    try:
        y = float(x)
    except Exception:
        return None
    return y/100.0 if y > 1.0 else y

def _robust_session(total=5, backoff=0.5) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=total,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET","POST"])
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

_HTTP = _robust_session()

def fetch_fmp_daily(symbol: str, start: str, end: str, apikey: str) -> pd.DataFrame:
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
    params = {"from": start, "to": end, "apikey": apikey, "serietype": "line"}
    r = _HTTP.get(url, params=params, timeout=30); r.raise_for_status()
    js = r.json(); hist = js.get("historical", [])
    if not hist: raise ValueError(f"FMP returned no data for {symbol}")
    df = pd.DataFrame(hist)
    df["date"] = pd.to_datetime(df["date"])
    price_col = "adjClose" if "adjClose" in df.columns else ("adj close" if "adj close" in df.columns else "close")
    out = df[["date", price_col]].rename(columns={price_col: "price"}).sort_values("date").set_index("date")
    out = out.asfreq("B").ffill()
    out = ensure_unique_index(out)
    return out

def monthly_from_daily_price(d: pd.DataFrame) -> pd.DataFrame:
    m = d.resample("M").last()
    r = m["price"].pct_change().fillna(0.0)
    tri = (1+r).cumprod()
    m.index = _to_me(m.index)
    out = pd.DataFrame({"mret": r, "tri": tri}, index=m.index)
    return ensure_unique_index(out)

def fetch_fred_dfii10(start: str, end: str, apikey: Optional[str]) -> pd.DataFrame:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id":"DFII10","observation_start":start,"observation_end":end,"file_type":"json"}
    if apikey and "YOUR_FRED_KEY_HERE" not in apikey: params["api_key"] = apikey
    r = _HTTP.get(url, params=params, timeout=30); r.raise_for_status()
    obs = r.json().get("observations", [])
    if not obs: raise ValueError("No DFII10 observations returned")
    df = pd.DataFrame(obs)[["date","value"]]
    df["date"] = pd.to_datetime(df["date"])
    def _to_float(x):
        try: return float(x)/100.0
        except: return np.nan
    df["tips10"] = df["value"].apply(_to_float)
    df = df.drop(columns=["value"]).set_index("date").sort_index()
    df.index = _to_me(df.index)
    df = ensure_unique_index(df)
    return df

def read_us_valuation_from_excel(xlsx: str, sheet: str) -> pd.DataFrame:
    d = pd.read_excel(xlsx, sheet_name=sheet)
    d.columns = [str(c).strip() for c in d.columns]
    if "date" not in d.columns: raise ValueError("Missing 'date' column in Excel.")
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").drop_duplicates("date").set_index("date")
    d.index = _to_me(d.index)
    d = ensure_unique_index(d)

    look = {c.lower(): c for c in d.columns}
    col_e = look.get("sp_real_earnings") or look.get("real_earnings")
    col_d = look.get("sp_div_real") or look.get("sp_real_dividends") or look.get("real_dividends")
    col_caey = look.get("ecy")
    col_cape = look.get("cape") or look.get("pe_10yr")

    if not col_e or not col_d: raise ValueError("Need 'sp_real_earnings' and 'sp_div_real' in Excel.")
    if not col_caey and not col_cape: raise ValueError("Need 'ecy' (CAEY) or 'cape' in Excel.")

    df = pd.DataFrame(index=d.index)
    df["E"] = pd.to_numeric(d[col_e], errors="coerce")
    df["D"] = pd.to_numeric(d[col_d], errors="coerce")

    if col_caey:
        df["CAEY"] = parse_numeric(d[col_caey])
    else:
        CAPE = pd.to_numeric(d[col_cape], errors="coerce")
        df["CAEY"] = 1.0 / CAPE

    if df["CAEY"].notna().sum() == 0 and col_cape:
        CAPE = parse_numeric(d[col_cape])
        df["CAEY"] = 1.0 / CAPE

    df["E10"] = df["E"].rolling(120, min_periods=120).mean()
    payout = (df["D"]/df["E"]).replace([np.inf,-np.inf], np.nan).clip(0,1)

    H = 120
    P_CAE = pd.Series(index=df.index, dtype=float)
    for i, t in enumerate(df.index):
        if i < H-1:
            P_CAE.loc[t] = np.nan
            continue
        w = df.iloc[i-H+1:i+1][["E","CAEY"]].copy().iloc[::-1].reset_index(drop=True)
        p = payout.iloc[i-H+1:i+1][::-1].reset_index(drop=True)
        acc = 0.0
        for k in range(H):
            Ek = float(w.loc[k,"E"])
            pk = float(p.loc[k]) if np.isfinite(p.loc[k]) else 0.0
            CAEYk = float(w.loc[k,"CAEY"])
            years = k/12.0
            acc += pk*Ek + (1-pk)*Ek*(1+CAEYk)**years
        P_CAE.loc[t] = acc/(H/12.0)

    df["P_CAE"]  = P_CAE
    df["P_CAEY"] = df["CAEY"] * (df["P_CAE"]/(df["E10"] + 1e-12))

    out = df[["CAEY","P_CAEY"]].copy()
    out["RP_USED"]   = out["P_CAEY"].combine_first(out["CAEY"])
    out["RP_SOURCE"] = np.where(out["P_CAEY"].notna(), "P_CAEY", "CAEY")
    for c in ["CAEY","P_CAEY","RP_USED"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = ensure_unique_index(out)
    return out

# ===================== Equity Sleeve Builders =====================
def _to_monthly_panel(tickers: Iterable[str], start: str, end: str, apikey: str) -> pd.DataFrame:
    mlist = []
    for s in tickers:
        d = fetch_fmp_daily(s, start, end, apikey)
        m = monthly_from_daily_price(d)[["mret"]].rename(columns={"mret": s})
        mlist.append(m)
    M = pd.concat(mlist, axis=1).dropna(how="any").sort_index()
    return M

def _sample_cov(returns: pd.DataFrame) -> np.ndarray:
    return np.cov(returns.values, rowvar=False, ddof=1)

def _const_corr_target(S: np.ndarray) -> np.ndarray:
    s = S.copy()
    n = s.shape[0]
    std = np.sqrt(np.diag(s))
    with np.errstate(invalid="ignore"):
        R = s / np.outer(std, std)
    np.fill_diagonal(R, 1.0)
    rho = (np.sum(R) - n) / (n * (n - 1))
    T = rho * np.outer(std, std)
    np.fill_diagonal(T, np.diag(s))
    return T

def _shrink_covariance(returns: pd.DataFrame, method: str = "ledoit_wolf",
                       ridge_lambda: float = 0.10, jitter: float = 1e-10) -> np.ndarray:
    X = returns.values
    S = _sample_cov(returns)
    if method == "ledoit_wolf":
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf(store_precision=False, assume_centered=False); lw.fit(X)
            Sigma = lw.covariance_
        except Exception as e:
            warnings.warn(f"LedoitWolf unavailable ({e}); falling back to const-corr shrinkage.")
            T = _const_corr_target(S)
            Sigma = (1.0 - ridge_lambda) * S + ridge_lambda * T
    elif method == "const_corr":
        T = _const_corr_target(S)
        Sigma = (1.0 - ridge_lambda) * S + ridge_lambda * T
    elif method == "ridge":
        D = np.diag(np.diag(S))
        Sigma = (1.0 - ridge_lambda) * S + ridge_lambda * D
    else:
        raise ValueError("Unknown cov_shrinkage method.")
    Sigma = Sigma + np.eye(Sigma.shape[0]) * (jitter * np.trace(Sigma))
    return Sigma

def _risk_parity_weights(Sigma: np.ndarray, budgets: Optional[np.ndarray] = None,
                         w0: Optional[np.ndarray] = None, tol: float = 1e-8, max_iter: int = 200) -> np.ndarray:
    n = Sigma.shape[0]
    b = np.ones(n)/n if budgets is None else np.asarray(budgets, dtype=float)
    b = b / np.sum(b)

    def objective(w):
        portfolio_var = w @ Sigma @ w
        if portfolio_var <= 1e-16:
            return 1e10
        portfolio_vol = np.sqrt(portfolio_var)
        marginal_contrib = Sigma @ w
        rc = w * marginal_contrib / portfolio_vol
        target_rc = b * portfolio_vol
        return np.sum((rc - target_rc) ** 2)

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = [(1e-6, 1.0) for _ in range(n)]
    w_init = np.ones(n) / n if w0 is None else w0

    result = minimize(
        objective,
        w_init,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': tol, 'maxiter': max_iter}
    )

    if not result.success:
        warnings.warn(f"Risk parity optimization did not fully converge: {result.message}")

    return result.x / np.sum(result.x)

def _apply_caps_and_renormalize(w: np.ndarray, max_cap: float, eps: float = 1e-12) -> np.ndarray:
    n = len(w); w = w.copy()
    if max_cap * n < 1.0 - 1e-12:
        warnings.warn("Per-asset max_cap is infeasible (N * cap < 1). Best-effort spread + renorm.")
    for _ in range(1000):
        over = w - max_cap
        viol = over > eps
        if not np.any(viol): break
        overflow = float(np.sum(over[viol]))
        w[viol] = np.minimum(w[viol], max_cap)
        head = np.where(~viol, max_cap - w, 0.0)
        tot = float(np.sum(head))
        if tot <= eps:
            return w / np.sum(w)
        w += overflow * (head / tot)
    return w / np.sum(w)

def _compose_herc_weights(Sigma: np.ndarray, tickers: List[str],
                          cluster_map: Optional[Mapping[str, List[str]]] = None,
                          cluster_budgets: Optional[Mapping[str, float]] = None,
                          max_cap: float = 0.35) -> np.ndarray:
    n = len(tickers)
    if cluster_map is None:
        w = _risk_parity_weights(Sigma)
        return _apply_caps_and_renormalize(w, max_cap=max_cap)

    clusters = {name: [tickers.index(s) for s in members if s in tickers]
                for name, members in cluster_map.items()}
    clusters = {k: v for k, v in clusters.items() if len(v) > 0}
    if not clusters:
        w = _risk_parity_weights(Sigma)
        return _apply_caps_and_renormalize(w, max_cap=max_cap)

    B = np.zeros((n, len(clusters)))
    cluster_names = list(clusters.keys())
    for j, cname in enumerate(cluster_names):
        idx = clusters[cname]
        S_sub = Sigma[np.ix_(idx, idx)]
        w_sub = _risk_parity_weights(S_sub)
        for k, a_idx in enumerate(idx):
            B[a_idx, j] = w_sub[k]

    S_cluster = B.T @ Sigma @ B
    if cluster_budgets is None:
        b_cl = np.ones(S_cluster.shape[0]) / S_cluster.shape[0]
    else:
        b_cl = np.array([cluster_budgets.get(name, 0.0) for name in cluster_names], dtype=float)
        if not np.all(b_cl > 0): raise ValueError("All clusters must have positive budgets.")
        b_cl = b_cl / np.sum(b_cl)

    w_cl = _risk_parity_weights(S_cluster, budgets=b_cl)
    w = B @ w_cl
    return _apply_caps_and_renormalize(w, max_cap=max_cap)

def _beta_max_weights(X: pd.DataFrame, y: pd.Series, max_cap: float) -> np.ndarray:
    L, n = X.shape
    if L < 3:
        return np.ones(n)/n
    Xc = X.values - X.values.mean(axis=0, keepdims=True)
    yc = y.values - y.values.mean()
    c = (Xc.T @ yc) / max(1, (L - 1))
    order = np.argsort(-c)
    w = np.zeros(n, dtype=float)
    remaining = 1.0
    for idx in order:
        if remaining <= 1e-12:
            break
        put = min(max_cap, remaining)
        w[idx] = put
        remaining -= put
    if remaining > 1e-12:
        headroom = np.minimum(max_cap - w, max_cap)
        tot = headroom.sum()
        if tot > 1e-12:
            w += remaining * (headroom / tot)
            remaining = 0.0
    s = w.sum()
    if s <= 0:
        return np.ones(n)/n
    w = w / s
    w = np.clip(w, 0.0, None)
    w = w / w.sum()
    return w

def _max_sharpe_weights(X: pd.DataFrame, Sigma: np.ndarray, rf_rate: float,
                        return_est_method: str, ema_span: int, min_weight: float,
                        max_cap: float) -> np.ndarray:
    n = X.shape[1]

    if return_est_method.lower() == "ema":
        weights_ema = np.exp(-np.arange(len(X))[::-1] / ema_span)
        weights_ema = weights_ema / weights_ema.sum()
        mu_monthly = (X * weights_ema[:, np.newaxis]).sum(axis=0).values
    else:
        mu_monthly = X.mean(axis=0).values

    mu_annual = mu_monthly * 12
    excess_returns = mu_annual - rf_rate

    def neg_sharpe(w):
        portfolio_return = w @ excess_returns
        portfolio_vol = np.sqrt(w @ Sigma @ w)
        if portfolio_vol < 1e-10:
            return 1e10
        return -portfolio_return / portfolio_vol

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = [(min_weight, max_cap) for _ in range(n)]
    w0 = np.ones(n) / n

    result = minimize(
        neg_sharpe,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 500}
    )

    if not result.success:
        warnings.warn(f"Max Sharpe optimization did not converge: {result.message}")

    w_opt = result.x
    return w_opt / np.sum(w_opt)

def build_equity_sleeve_monthly(
    equity_ticker_or_list,
    start: str, end: str, apikey: str,
    method: str = "equal_weight",
    lookback_m: int = 60,
    warmup_m: Optional[int] = None,
    max_cap: float = 0.35,
    min_weight_per_asset: float = 0.0,
    cov_shrinkage: str = "ledoit_wolf",
    ridge_lambda: float = 0.10,
    clusters: Optional[Dict[str, List[str]]] = None,
    cluster_risk_budgets: Optional[Dict[str, float]] = None,
    benchmark_symbol: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    warm = int(lookback_m if warmup_m is None else warmup_m)
    start_dt = pd.to_datetime(start).to_period("M").to_timestamp("M")
    start_warm_dt = (start_dt - pd.DateOffset(months=warm)).to_period("M").to_timestamp("M")
    start_warm = start_warm_dt.strftime("%Y-%m-%d")

    if isinstance(equity_ticker_or_list, str):
        d = fetch_fmp_daily(equity_ticker_or_list, start_warm, end, apikey)
        m = monthly_from_daily_price(d)
        # Return full dataset including warmup period - trimming will be done by caller
        P = d["price"].resample("M").last().to_frame(name=equity_ticker_or_list)
        P.index = _to_me(P.index)
        P_full = P.reindex(m.index).ffill()
        W_target = pd.DataFrame(1.0, index=m.index, columns=[equity_ticker_or_list])
        W_exec   = W_target.copy()
        return m, W_target, W_exec, P_full

    tickers = list(equity_ticker_or_list)
    if len(tickers) == 0:
        raise ValueError("Equity tickers list is empty.")

    R_full = _to_monthly_panel(tickers, start_warm, end, apikey)
    P_list = []
    for s in tickers:
        d = fetch_fmp_daily(s, start_warm, end, apikey)
        p = d["price"].resample("M").last().to_frame(name=s)
        p.index = _to_me(p.index)
        P_list.append(p)
    P_full = pd.concat(P_list, axis=1).reindex(R_full.index).ffill()

    if method.lower() == "beta_max":
        bsym = benchmark_symbol or BENCHMARK_TICKER
        bm_daily = fetch_fmp_daily(bsym, start_warm, end, apikey)
        bm_m = monthly_from_daily_price(bm_daily)[["mret"]].rename(columns={"mret":"bm"})
        bm_m = bm_m.reindex(R_full.index).dropna().astype(float)
        common_idx = R_full.index.intersection(bm_m.index)
        R_full = R_full.loc[common_idx]
        bm_m   = bm_m.loc[common_idx]

    if method.lower() == "equal_weight":
        w = np.ones(len(tickers))/len(tickers)
        W_full = pd.DataFrame(np.tile(w, (len(R_full), 1)), index=R_full.index, columns=tickers)

    elif method.lower() == "optimized":
        if lookback_m < 12: raise ValueError("lookback_m should be at least 12.")
        rows = []
        for t in range(len(R_full)):
            if t < lookback_m:
                rows.append(np.ones(len(tickers))/len(tickers))
                continue
            window = R_full.iloc[t-lookback_m:t]
            Sigma = _shrink_covariance(window, method=cov_shrinkage, ridge_lambda=ridge_lambda, jitter=1e-10)
            w_t = _compose_herc_weights(Sigma, tickers, clusters, cluster_risk_budgets, max_cap=max_cap)
            rows.append(w_t)
        W_full = pd.DataFrame(rows, index=R_full.index, columns=tickers)

    elif method.lower() == "beta_max":
        if lookback_m < 12: raise ValueError("lookback_m should be at least 12 for beta_max.")
        rows = []
        for t in range(len(R_full)):
            if t < lookback_m:
                rows.append(np.ones(len(tickers))/len(tickers))
                continue
            window_X = R_full.iloc[t-lookback_m:t]
            window_y = bm_m.iloc[t-lookback_m:t, 0]
            w_t = _beta_max_weights(window_X, window_y, max_cap=max_cap)
            rows.append(w_t)
        W_full = pd.DataFrame(rows, index=R_full.index, columns=tickers)

    elif method.lower() == "max_sharpe":
        if lookback_m < 12: raise ValueError("lookback_m should be at least 12 for max_sharpe.")
        rf_rate = globals().get('EQUITY_MAX_SHARPE_RF_RATE', 0.02)
        return_est_method = globals().get('EQUITY_MAX_SHARPE_RETURN_EST', 'historical_mean')
        ema_span = globals().get('EQUITY_MAX_SHARPE_EMA_SPAN', 36)
        min_weight = globals().get('EQUITY_MAX_SHARPE_MIN_WEIGHT', 0.01)

        rows = []
        for t in range(len(R_full)):
            if t < lookback_m:
                rows.append(np.ones(len(tickers))/len(tickers))
                continue
            window = R_full.iloc[t-lookback_m:t]
            Sigma = _shrink_covariance(window, method=cov_shrinkage, ridge_lambda=ridge_lambda, jitter=1e-10)
            Sigma_annual = Sigma * 12
            w_t = _max_sharpe_weights(
                window, Sigma_annual, rf_rate, return_est_method,
                ema_span, min_weight, max_cap
            )
            rows.append(w_t)
        W_full = pd.DataFrame(rows, index=R_full.index, columns=tickers)
    else:
        raise ValueError("EQUITY_SLEEVE_METHOD must be 'equal_weight', 'optimized', 'beta_max', or 'max_sharpe'.")

    # Apply minimum weight constraint if specified
    if min_weight_per_asset > 0:
        for idx in W_full.index:
            weights = W_full.loc[idx].values
            # Set weights below minimum to minimum
            weights = np.where(weights < min_weight_per_asset, min_weight_per_asset, weights)
            # Renormalize to sum to 1
            weights = weights / weights.sum()
            W_full.loc[idx] = weights

    W_exec_full = W_full.shift(1).bfill()

    # Return full dataset including warmup period - trimming will be done by caller
    sleeve_ret = (W_exec_full * R_full).sum(axis=1)
    tri = (1.0 + sleeve_ret).cumprod()
    eq_m = pd.DataFrame({"mret": sleeve_ret.astype(float), "tri": tri.astype(float)}, index=R_full.index)

    return eq_m, W_full, W_exec_full, P_full


# ===================== TRADING DAY DETECTION =====================
def is_last_trading_day_of_month() -> bool:
    """
    Determine if today is the last trading day of the current month.

    Uses FMP market hours API to check trading status.
    Returns True if today is a trading day and is the last one this month.
    """
    try:
        # Get current date in Central Time
        central = pytz.timezone('America/Chicago')
        now_ct = datetime.now(central)
        today = now_ct.date()

        # Check if today is a trading day using FMP API
        url = f"https://financialmodelingprep.com/api/v3/is-the-market-open"
        params = {"apikey": FMP_KEY}
        r = _HTTP.get(url, params=params, timeout=10)
        r.raise_for_status()
        market_status = r.json()

        # If market is not open today, not a trading day
        if not market_status.get("isTheStockMarketOpen", False):
            return False

        # Find next trading day after today
        current_year = today.year
        current_month = today.month

        # Check each remaining day in the month
        next_day = today + pd.Timedelta(days=1)
        while next_day.month == current_month:
            # Use pandas to check if it's a business day (simple heuristic)
            if next_day.weekday() < 5:  # Monday-Friday
                # This is likely a trading day, so today is NOT the last
                return False
            next_day = next_day + pd.Timedelta(days=1)

        # If we've checked all remaining days and none are trading days, today is last
        return True

    except Exception as e:
        print(f"Warning: Could not determine if today is last trading day: {e}")
        # Fallback: check if it's the last business day of the month
        today = pd.Timestamp.today().date()
        last_day = pd.Timestamp(today.year, today.month, 1) + pd.offsets.MonthEnd(0)
        last_bday = last_day
        while last_bday.weekday() >= 5:  # Skip weekends
            last_bday = last_bday - pd.Timedelta(days=1)
        return today == last_bday.date()


def should_run_now() -> bool:
    """
    Check if the script should run right now.
    Must be 5 PM CT or later, and must be the last trading day.
    """
    central = pytz.timezone('America/Chicago')
    now_ct = datetime.now(central)

    # Check if it's 5 PM or later
    target_time = time(17, 0)  # 5:00 PM
    if now_ct.time() < target_time:
        print(f"Current time {now_ct.strftime('%I:%M %p CT')} is before 5:00 PM CT. Skipping.")
        return False

    # Check if it's the last trading day
    if not is_last_trading_day_of_month():
        print(f"Today is not the last trading day of the month. Skipping.")
        return False

    return True


# ===================== MAIN EXECUTION =====================
def calculate_current_allocations():
    """
    Calculate and return current month's allocation percentages.
    """
    print("=" * 80)
    print("WEALTH UTILITY - MONTHLY ALLOCATION CALCULATOR")
    print("=" * 80)
    print(f"Run Time: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}")
    print()

    start = START_DATE
    end = _end_date()

    # Calculate warmup period needed for all rolling calculations
    warmup_months = max(EQUITY_SLEEVE_WARMUP_M, MOM_LOOKBACK_M, RISK_LOOKBACK_M)
    start_dt = pd.to_datetime(start).to_period("M").to_timestamp("M")
    warmup_dt = (start_dt - pd.DateOffset(months=warmup_months)).to_period("M").to_timestamp("M")
    warmup_start = warmup_dt.strftime("%Y-%m-%d")

    # Load data
    print("Loading valuation data from Excel...")
    us_val = read_us_valuation_from_excel(ECY_XLSX_PATH, ECY_SHEET)

    print("Fetching TIPS data from FRED...")
    tips = fetch_fred_dfii10(warmup_start, end, FRED_KEY)

    print("Building equity sleeve...")
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

    print("Fetching non-equity data...")
    ne_m = monthly_from_daily_price(fetch_fmp_daily(NON_EQUITY_TICKER, warmup_start, end, FMP_KEY))

    # Common index - include warmup period for calculations
    idx = eq_m.index.intersection(ne_m.index)

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
    latest_target_equity_weight = panel.loc[latest_date, "w_target"]
    latest_target_safe_weight = 1.0 - latest_target_equity_weight

    # Get within-sleeve weights (these are for NEXT month execution)
    W_exec_eq = _eq_W_exec.reindex(panel.index).ffill()
    latest_sleeve_weights = W_exec_eq.loc[latest_date]

    # Build allocation table
    print()
    print("=" * 80)
    print(f"ALLOCATION FOR MONTH ENDING: {latest_date.strftime('%Y-%m-%d')}")
    print("=" * 80)
    print()
    print("EQUITY SLEEVE:")
    print("-" * 80)

    allocations = []

    if isinstance(EQUITY_TICKER, str):
        equity_alloc = latest_target_equity_weight * 1.0
        print(f"  {EQUITY_TICKER:20s} {equity_alloc:>8.2%}")
        allocations.append({"Asset": EQUITY_TICKER, "Weight": equity_alloc})
    else:
        for ticker in EQUITY_TICKER:
            sleeve_weight = latest_sleeve_weights[ticker]
            equity_alloc = latest_target_equity_weight * sleeve_weight
            print(f"  {ticker:20s} {equity_alloc:>8.2%}")
            allocations.append({"Asset": ticker, "Weight": equity_alloc})

    print()
    print("NON-EQUITY:")
    print("-" * 80)
    print(f"  {NON_EQUITY_TICKER:20s} {latest_target_safe_weight:>8.2%}")
    allocations.append({"Asset": NON_EQUITY_TICKER, "Weight": latest_target_safe_weight})

    print()
    print("=" * 80)
    print("SUMMARY:")
    print("-" * 80)
    print(f"  Total Equity:     {latest_target_equity_weight:>8.2%}")
    print(f"  Total Non-Equity: {latest_target_safe_weight:>8.2%}")
    print(f"  Total:            {(latest_target_equity_weight + latest_target_safe_weight):>8.2%}")
    print("=" * 80)
    print()

    # Save to file
    output_file = "current_allocation.txt"
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"WEALTH UTILITY - ALLOCATION FOR {latest_date.strftime('%Y-%m-%d')}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}\n")
        f.write("=" * 80 + "\n\n")
        f.write("EQUITY SLEEVE:\n")
        f.write("-" * 80 + "\n")

        if isinstance(EQUITY_TICKER, str):
            f.write(f"  {EQUITY_TICKER:20s} {latest_target_equity_weight:>8.2%}\n")
        else:
            for ticker in EQUITY_TICKER:
                sleeve_weight = latest_sleeve_weights[ticker]
                equity_alloc = latest_target_equity_weight * sleeve_weight
                f.write(f"  {ticker:20s} {equity_alloc:>8.2%}\n")

        f.write("\nNON-EQUITY:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  {NON_EQUITY_TICKER:20s} {latest_target_safe_weight:>8.2%}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Total Equity:     {latest_target_equity_weight:>8.2%}\n")
        f.write(f"  Total Non-Equity: {latest_target_safe_weight:>8.2%}\n")
        f.write(f"  Total:            {(latest_target_equity_weight + latest_target_safe_weight):>8.2%}\n")
        f.write("=" * 80 + "\n")

    print(f"Allocation saved to: {output_file}")

    return allocations


def perf_stats(nav: pd.Series, rf_monthly: Optional[pd.Series] = None) -> dict:
    """
    Calculate performance statistics for a NAV series.

    Args:
        nav: Series of NAV values (already cumulative product starting at 1.0)
        rf_monthly: Optional series of monthly risk-free rates

    Returns:
        Dictionary of performance metrics
    """
    nav = nav.dropna()
    ret = nav.pct_change().dropna()
    if len(ret) == 0:
        return {}
    ann = 12
    sqrt_ann = np.sqrt(ann)
    # CORRECTED: CAGR formula that properly accounts for starting NAV
    start_val = nav.iloc[0]
    end_val = nav.iloc[-1]
    months = len(ret)
    if start_val <= 0 or end_val <= 0:
        cagr = 0.0
    else:
        cagr = (end_val / start_val) ** (ann / months) - 1
    vol  = ret.std(ddof=0) * sqrt_ann
    mdd  = (nav / nav.cummax() - 1).min()
    sh_total = (ret.mean() / (ret.std(ddof=0) + 1e-12)) * sqrt_ann
    out = {
        "CAGR": cagr, "AnnVol": vol, "MaxDD": mdd, "Sharpe": sh_total,
        "Start": nav.index[0].strftime('%Y-%m-%d'),
        "End": nav.index[-1].strftime('%Y-%m-%d'),
        "Months": len(ret),
    }
    if rf_monthly is not None and rf_monthly.reindex(ret.index).notna().any():
        ex = ret - rf_monthly.reindex(ret.index).fillna(0.0)
        sh_ex = (ex.mean() / (ex.std(ddof=0) + 1e-12)) * sqrt_ann
        out["Sharpe_excess"] = sh_ex
    return out


def run_backtest(start_date: Optional[str] = None, end_date: Optional[str] = None, baseline_w: Optional[float] = None, equity_tickers: Optional[list] = None) -> dict:
    """
    Run full historical backtest and return performance metrics, equity curves, and allocations.

    Args:
        start_date: Optional start date (defaults to START_DATE constant)
        end_date: Optional end date (defaults to today)
        baseline_w: Optional baseline equity weight (0.0 to 1.0, defaults to BASELINE_W constant)
        equity_tickers: Optional list of equity ticker symbols (defaults to EQUITY_TICKER constant)

    Returns:
        Dictionary with backtest results including performance metrics, equity curve,
        monthly returns, drawdowns, and allocation history
    """
    start = start_date or START_DATE
    end = end_date or _end_date()

    # Use provided baseline_w or default to constant
    baseline_weight = baseline_w if baseline_w is not None else BASELINE_W

    # Use provided equity_tickers or default to constant
    equity_ticker_list = equity_tickers if equity_tickers is not None else EQUITY_TICKER

    # Validate baseline_w
    if not (0.0 <= baseline_weight <= 1.0):
        raise ValueError(f"baseline_w must be between 0.0 and 1.0, got {baseline_weight}")

    # Calculate warmup period needed for all rolling calculations
    warmup_months = max(EQUITY_SLEEVE_WARMUP_M, MOM_LOOKBACK_M, RISK_LOOKBACK_M)
    start_dt = pd.to_datetime(start).to_period("M").to_timestamp("M")
    warmup_dt = (start_dt - pd.DateOffset(months=warmup_months)).to_period("M").to_timestamp("M")
    warmup_start = warmup_dt.strftime("%Y-%m-%d")

    # Load data (same as calculate_current_allocations)
    us_val = read_us_valuation_from_excel(ECY_XLSX_PATH, ECY_SHEET)
    tips = fetch_fred_dfii10(warmup_start, end, FRED_KEY)

    # Fetch equity sleeve data with warmup (build_equity_sleeve_monthly handles its own warmup)
    eq_m, _eq_W_target, _eq_W_exec, _eq_P = build_equity_sleeve_monthly(
        equity_ticker_list, start, end, FMP_KEY,
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

    # Fetch non-equity and benchmark data with warmup period
    ne_m = monthly_from_daily_price(fetch_fmp_daily(NON_EQUITY_TICKER, warmup_start, end, FMP_KEY))
    bm_m = monthly_from_daily_price(fetch_fmp_daily(BENCHMARK_TICKER, warmup_start, end, FMP_KEY))

    # Common index - include warmup period for calculations
    idx = eq_m.index.intersection(ne_m.index).intersection(bm_m.index)

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
    panel["bm_ret"], panel["bm_tri"] = bm_m.loc[idx,"mret"].astype(float), bm_m.loc[idx,"tri"].astype(float)
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
    value_bump = baseline_weight * VALUE_DIAL * rel_value
    w_value = (baseline_weight + value_bump).clip(0, 1)

    # Momentum bump
    mom_bump = baseline_weight * MOM_BUMP * panel["MOM_STATE"]
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

    # Final target weight with dynamic f_max based on baseline_w
    f_min_frac = _to_frac(f_min) or 0.0
    # f_max = baseline_w + 15%, capped at 100%
    f_max_dynamic = min((baseline_weight + 0.15) * 100, 100)
    f_max_frac = _to_frac(f_max_dynamic) or 1.0
    panel["w_target"] = w_capped.clip(f_min_frac, f_max_frac).clip(0, 1)

    # Execution lag
    panel["w_exec"] = panel["w_target"].shift(1)
    first_idx = panel["w_exec"].first_valid_index()
    if first_idx is not None and pd.isna(panel.loc[first_idx,"w_exec"]):
        panel.loc[first_idx,"w_exec"] = baseline_weight
    panel["w_exec"] = panel["w_exec"].ffill()

    # Backtest
    panel["dw"]   = panel["w_exec"].diff().abs().fillna(0.0)
    panel["cost"] = (TURNOVER_BPS/10000.0) * panel["dw"]

    gross = panel["w_exec"]*panel["eq_ret"] + (1-panel["w_exec"])*panel["ne_ret"]
    panel["port_ret"] = (gross - panel["cost"]).astype(float)

    panel["port_nav"] = (1 + panel["port_ret"]).cumprod()
    panel["bm_nav"]   = (1 + panel["bm_ret"]).cumprod()
    panel["alloc_bench_ret"] = baseline_weight*panel["bm_ret"] + (1-baseline_weight)*panel["ne_ret"]
    panel["alloc_bench_nav"] = (1 + panel["alloc_bench_ret"]).cumprod()

    # Trim results to user-specified start date (after all calculations with warmup are complete)
    panel = panel.loc[panel.index >= start_dt]
    _eq_W_exec = _eq_W_exec.loc[_eq_W_exec.index >= start_dt]
    _eq_W_target = _eq_W_target.loc[_eq_W_target.index >= start_dt]

    # Drop first row if w_exec is invalid (NaN or 0) due to execution lag shift
    # This ensures strategy and benchmark start from the same valid date
    if len(panel) > 0 and (pd.isna(panel.iloc[0]["w_exec"]) or panel.iloc[0]["w_exec"] == 0):
        first_valid_idx = panel.index[1] if len(panel) > 1 else panel.index[0]
        panel = panel.loc[panel.index >= first_valid_idx]
        _eq_W_exec = _eq_W_exec.loc[_eq_W_exec.index >= first_valid_idx]
        _eq_W_target = _eq_W_target.loc[_eq_W_target.index >= first_valid_idx]

    # Recalculate NAVs from trimmed returns to ensure all start at same point with same number of periods
    panel["port_nav"] = (1 + panel["port_ret"]).cumprod()
    panel["bm_nav"] = (1 + panel["bm_ret"]).cumprod()
    panel["alloc_bench_nav"] = (1 + panel["alloc_bench_ret"]).cumprod()

    # Calculate performance stats
    rf_m = panel["tips10"]/12.0
    portfolio_stats = perf_stats(panel["port_nav"], rf_m)
    benchmark_stats = perf_stats(panel["bm_nav"], rf_m)
    alloc_bench_stats = perf_stats(panel["alloc_bench_nav"], rf_m)

    # Calculate rolling metrics
    strategy_rets = panel["port_nav"].pct_change()
    benchmark_rets = panel["bm_nav"].pct_change()

    # Rolling 12-month beta
    rolling_cov = strategy_rets.rolling(12).cov(benchmark_rets)
    rolling_var = benchmark_rets.rolling(12).var()
    rolling_beta = (rolling_cov / rolling_var).fillna(0)

    # Rolling 12-month volatility
    rolling_vol_strat = strategy_rets.rolling(12).std(ddof=0) * np.sqrt(12)
    rolling_vol_bench = benchmark_rets.rolling(12).std(ddof=0) * np.sqrt(12)

    # Rolling 12-month Sharpe
    def rolling_sharpe(returns, rf_monthly_series, window=12):
        rf_aligned = rf_monthly_series.reindex(returns.index).fillna(0.0)
        excess = returns - rf_aligned
        rolling_mean = excess.rolling(window).mean()
        rolling_std = returns.rolling(window).std(ddof=0)
        return (rolling_mean / (rolling_std + 1e-12)) * np.sqrt(12)

    rolling_sharpe_strat = rolling_sharpe(strategy_rets, rf_m, window=12)
    rolling_sharpe_bench = rolling_sharpe(benchmark_rets, rf_m, window=12)

    # Drawdown
    dd_strat = panel["port_nav"] / panel["port_nav"].cummax() - 1.0
    dd_bench = panel["bm_nav"] / panel["bm_nav"].cummax() - 1.0

    # Monthly allocation history
    W_exec_eq = _eq_W_exec.reindex(panel.index).ffill()
    allocation_history = []

    for dt in panel.index:
        month_alloc = {
            "date": dt.strftime('%Y-%m-%d'),
            "equity_weight": float(panel.loc[dt, "w_exec"]),
            "safe_weight": float(1.0 - panel.loc[dt, "w_exec"]),
            "assets": []
        }

        if isinstance(equity_ticker_list, str):
            month_alloc["assets"].append({
                "ticker": equity_ticker_list,
                "weight": float(panel.loc[dt, "w_exec"])
            })
        else:
            for tk in equity_ticker_list:
                sleeve_weight = float(W_exec_eq.loc[dt, tk])
                portfolio_weight = float(panel.loc[dt, "w_exec"]) * sleeve_weight
                month_alloc["assets"].append({
                    "ticker": tk,
                    "weight": portfolio_weight
                })

        month_alloc["assets"].append({
            "ticker": NON_EQUITY_TICKER,
            "weight": float(1.0 - panel.loc[dt, "w_exec"])
        })

        allocation_history.append(month_alloc)

    # Helper function to clean NaN values for JSON serialization
    def clean_value(val):
        """Convert NaN/inf to None for valid JSON"""
        if isinstance(val, (float, np.floating)):
            if np.isnan(val) or np.isinf(val):
                return None
        return val

    def clean_list(lst):
        """Clean a list of values"""
        return [clean_value(v) for v in lst]

    # Prepare result
    result = {
        "performance": {
            "portfolio": portfolio_stats,
            "benchmark": benchmark_stats,
            "allocation_benchmark": alloc_bench_stats
        },
        "equity_curve": {
            "dates": [d.strftime('%Y-%m-%d') for d in panel.index],
            "portfolio": clean_list(panel["port_nav"].tolist()),
            "benchmark": clean_list(panel["bm_nav"].tolist()),
            "allocation_benchmark": clean_list(panel["alloc_bench_nav"].tolist())
        },
        "monthly_returns": {
            "dates": [d.strftime('%Y-%m-%d') for d in panel.index],
            "portfolio": clean_list(panel["port_ret"].tolist()),
            "benchmark": clean_list(panel["bm_ret"].tolist())
        },
        "drawdown": {
            "dates": [d.strftime('%Y-%m-%d') for d in panel.index],
            "portfolio": clean_list(dd_strat.tolist()),
            "benchmark": clean_list(dd_bench.tolist())
        },
        "rolling_metrics": {
            "dates": [d.strftime('%Y-%m-%d') for d in panel.index],
            "beta_12m": clean_list(rolling_beta.tolist()),
            "volatility_12m": {
                "portfolio": clean_list(rolling_vol_strat.tolist()),
                "benchmark": clean_list(rolling_vol_bench.tolist())
            },
            "sharpe_12m": {
                "portfolio": clean_list(rolling_sharpe_strat.tolist()),
                "benchmark": clean_list(rolling_sharpe_bench.tolist())
            }
        },
        "allocation_history": [
            {
                "date": month["date"],
                "equity_weight": clean_value(month["equity_weight"]),
                "safe_weight": clean_value(month["safe_weight"]),
                "assets": [
                    {
                        "ticker": asset["ticker"],
                        "weight": clean_value(asset["weight"])
                    }
                    for asset in month["assets"]
                ]
            }
            for month in allocation_history
        ],
        "config": {
            "equity_tickers": EQUITY_TICKER if isinstance(EQUITY_TICKER, list) else [EQUITY_TICKER],
            "non_equity_ticker": NON_EQUITY_TICKER,
            "benchmark_ticker": BENCHMARK_TICKER,
            "sleeve_method": EQUITY_SLEEVE_METHOD,
            "baseline_w": baseline_weight,
            "f_min": f_min,
            "f_max": f_max_dynamic,
            "start_date": start,
            "end_date": end
        }
    }

    return result


def main():
    """
    Main entry point for the production script.
    Checks if it should run, then calculates allocations.
    """
    # Override for testing: set to True to always run regardless of time/day
    # Can be enabled via environment variable or command-line argument
    FORCE_RUN = os.getenv("FORCE_RUN", "false").lower() == "true"

    # Check for --test or --force command-line argument
    if len(sys.argv) > 1 and sys.argv[1] in ["--test", "--force", "-f", "-t"]:
        FORCE_RUN = True
        print("FORCE_RUN mode enabled via command-line argument")
        print()

    if not FORCE_RUN and not should_run_now():
        return

    print("[OK] Conditions met. Running allocation calculation...")
    print()

    try:
        allocations = calculate_current_allocations()
        print()
        print("[OK] Allocation calculation completed successfully!")

    except Exception as e:
        print()
        print(f"[ERROR] Error during calculation: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()