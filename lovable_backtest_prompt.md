# Lovable.dev Prompt: Wealth Utility Backtest Tab

## Task
Create a new tab called "Wealth Utility Backtest" in the AdvisorTS webapp that displays historical backtest performance metrics, charts, and analysis for the Wealth Utility dynamic allocation strategy.

## API Endpoint
**Base URL:** `https://wealthutilitylive-production.up.railway.app`

**Endpoint:** `GET /backtest`

**Response Structure:**
```json
{
  "success": true,
  "calculation_timestamp": "2026-01-06T15:30:00",
  "performance": {
    "portfolio": {
      "CAGR": 0.0618,
      "AnnVol": 0.1234,
      "MaxDD": -0.3456,
      "Sharpe": 0.49,
      "Start": "2008-03-31",
      "End": "2025-12-31",
      "Months": 215
    },
    "benchmark": {
      "CAGR": 0.0965,
      "AnnVol": 0.1789,
      "MaxDD": -0.5123,
      "Sharpe": 0.54,
      "Start": "2008-03-31",
      "End": "2025-12-31",
      "Months": 215
    },
    "allocation_benchmark": {
      "CAGR": 0.0787,
      "AnnVol": 0.1456,
      "MaxDD": -0.4234,
      "Sharpe": 0.52,
      "Start": "2008-03-31",
      "End": "2025-12-31",
      "Months": 215
    }
  },
  "equity_curve": {
    "dates": ["2008-03-31", "2008-04-30", ...],
    "portfolio": [1.0, 1.02, 1.05, ...],
    "benchmark": [1.0, 1.01, 1.03, ...],
    "allocation_benchmark": [1.0, 1.015, 1.04, ...]
  },
  "monthly_returns": {
    "dates": ["2008-03-31", ...],
    "portfolio": [0.02, -0.01, 0.03, ...],
    "benchmark": [0.01, -0.02, 0.04, ...]
  },
  "drawdown": {
    "dates": ["2008-03-31", ...],
    "portfolio": [0, -0.05, -0.10, ...],
    "benchmark": [0, -0.08, -0.15, ...]
  },
  "rolling_metrics": {
    "dates": ["2008-03-31", ...],
    "beta_12m": [null, null, ..., 0.65, 0.68, ...],
    "volatility_12m": {
      "portfolio": [null, ..., 0.12, 0.13, ...],
      "benchmark": [null, ..., 0.18, 0.17, ...]
    },
    "sharpe_12m": {
      "portfolio": [null, ..., 0.45, 0.52, ...],
      "benchmark": [null, ..., 0.50, 0.54, ...]
    }
  },
  "allocation_history": [
    {
      "date": "2008-03-31",
      "equity_weight": 0.65,
      "safe_weight": 0.35,
      "assets": [
        {"ticker": "ACWI", "weight": 0.13},
        {"ticker": "RPG", "weight": 0.10},
        ...
        {"ticker": "BIL", "weight": 0.35}
      ]
    },
    ...
  ],
  "config": {
    "equity_tickers": ["ACWI", "RPG", "IWR", "EFA", "QQQ", "EEM", "MGK", "DBC", "REZ"],
    "non_equity_ticker": "BIL",
    "benchmark_ticker": "SPY",
    "sleeve_method": "max_sharpe",
    "start_date": "2000-01-01",
    "end_date": "2026-01-06"
  }
}
```

## Implementation Requirements

### 1. Create New Tab
- Add a new navigation tab labeled "Wealth Utility Backtest" next to existing tabs
- Icon: Use a chart/analytics icon (e.g., TrendingUp, BarChart, or Activity)
- Route: `/wealth-utility-backtest`

### 2. Layout Structure
Create a responsive dashboard layout with the following sections (top to bottom):

#### Section A: Performance Summary Cards (Top Row)
Display three metric cards side-by-side:
- **Wealth Utility Portfolio**
  - CAGR: Show as percentage (e.g., "6.18%")
  - Sharpe Ratio: Show to 2 decimals (e.g., "0.49")
  - Max Drawdown: Show as percentage (e.g., "-34.56%")
  - Annualized Volatility: Show as percentage (e.g., "12.34%")
- **S&P 500 Benchmark**
  - Same metrics as portfolio
- **100% Allocation Benchmark**
  - Same metrics as portfolio

Design: Use Card components with colored headers (blue for portfolio, gray for benchmarks). Make cards responsive (stack on mobile, 3-column on desktop).

#### Section B: Cumulative Performance Chart
**Chart Type:** Line chart (use Recharts library)

**Data:**
- X-axis: Dates from `equity_curve.dates`
- Y-axis: NAV indexed to 100
- Three lines:
  - Wealth Utility (primary blue, thicker line)
  - S&P 500 (gray)
  - Alloc Benchmark (light gray, dashed)

**Features:**
- Logarithmic Y-axis scale (better for long-term performance visualization)
- Tooltip showing date and NAV values for all three series
- Legend
- Responsive height (400px on desktop, 300px on mobile)

#### Section C: Drawdown Chart
**Chart Type:** Area chart

**Data:**
- X-axis: Dates from `drawdown.dates`
- Y-axis: Drawdown as percentage
- Two filled areas:
  - Wealth Utility (blue, semi-transparent)
  - S&P 500 (gray, semi-transparent)

**Features:**
- Y-axis formatted as percentages
- Tooltip showing date and drawdown values
- Legend

#### Section D: Rolling Metrics (3 Charts Side-by-Side)
Create three smaller charts in a row (stack on mobile):

1. **Rolling 12-Month Beta**
   - Line chart
   - Data: `rolling_metrics.beta_12m`
   - Add horizontal line at beta = 1.0
   - Filter out null values at start

2. **Rolling 12-Month Volatility**
   - Line chart
   - Two lines: portfolio and benchmark
   - Data: `rolling_metrics.volatility_12m`
   - Y-axis formatted as percentages

3. **Rolling 12-Month Sharpe Ratio**
   - Line chart
   - Two lines: portfolio and benchmark
   - Data: `rolling_metrics.sharpe_12m`
   - Add horizontal line at Sharpe = 0

#### Section E: Monthly Returns Heatmap
**Chart Type:** Calendar heatmap

**Data:**
- Transform `monthly_returns` into a year × month grid
- Rows: Years (2008-2025)
- Columns: Months (Jan-Dec)
- Cell color: Green for positive returns, red for negative
- Cell value: Return percentage

**Library:** Use Recharts or a custom component

#### Section F: Allocation History Table
**Component:** Scrollable data table

**Columns:**
- Date
- Equity Weight (%)
- Safe Weight (%)
- Individual asset weights (expandable/collapsible detail)

**Features:**
- Pagination or virtual scrolling
- Sort by date (newest first by default)
- Export to CSV button

### 3. Loading & Error States
- Show loading spinner while fetching data (API call takes ~10 seconds)
- Display friendly error message if API fails
- Add "Refresh Data" button to re-fetch

### 4. Data Fetching Strategy
```typescript
const API_URL = 'https://wealthutilitylive-production.up.railway.app/backtest';

async function fetchBacktestData() {
  const response = await fetch(API_URL);
  if (!response.ok) {
    throw new Error(`Failed to fetch backtest data: ${response.statusText}`);
  }
  const data = await response.json();
  if (!data.success) {
    throw new Error(data.error || 'Backtest calculation failed');
  }
  return data;
}

// Use React Query or similar for caching
const { data, isLoading, error, refetch } = useQuery(
  'backtestData',
  fetchBacktestData,
  {
    staleTime: 1000 * 60 * 60 * 2, // Cache for 2 hours
    refetchOnWindowFocus: false
  }
);
```

### 5. Styling Guidelines
- Use your existing color scheme and design system
- Ensure responsive design (mobile-first)
- Use consistent spacing and typography
- Add smooth transitions for chart interactions
- Use your existing Card, Button, and Table components

### 6. Performance Considerations
- The `/backtest` endpoint is cached for 2 hours on the server
- Memoize expensive chart components
- Use virtualization for the allocation history table (only render visible rows)
- Consider lazy-loading charts below the fold

### 7. Additional Features (Optional Enhancements)
- **Date Range Selector:** Add UI to specify `start_date` and `end_date` query parameters
- **Compare Mode:** Toggle to show/hide benchmark lines
- **Chart Export:** Allow users to download charts as PNG
- **Print View:** Optimized layout for printing reports
- **Statistics Tooltip:** Add info icons explaining metrics (CAGR, Sharpe, etc.)

## Sample Code Structure

```typescript
import { useQuery } from '@tanstack/react-query';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const WealthUtilityBacktest = () => {
  const { data, isLoading, error, refetch } = useQuery(
    ['backtest'],
    async () => {
      const response = await fetch('https://wealthutilitylive-production.up.railway.app/backtest');
      const json = await response.json();
      return json;
    },
    { staleTime: 1000 * 60 * 120 } // 2 hour cache
  );

  if (isLoading) return <LoadingSpinner />;
  if (error) return <ErrorMessage error={error} onRetry={refetch} />;

  const performanceMetrics = data.performance;
  const equityCurve = prepareEquityCurveData(data.equity_curve);

  return (
    <div className="space-y-6 p-6">
      <h1 className="text-3xl font-bold">Wealth Utility Backtest</h1>

      {/* Performance Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <PerformanceCard title="Wealth Utility" metrics={performanceMetrics.portfolio} color="blue" />
        <PerformanceCard title="S&P 500" metrics={performanceMetrics.benchmark} color="gray" />
        <PerformanceCard title="Alloc Benchmark" metrics={performanceMetrics.allocation_benchmark} color="gray" />
      </div>

      {/* Cumulative Performance */}
      <Card>
        <CardHeader>
          <CardTitle>Cumulative Performance (Log Scale)</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={equityCurve}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis scale="log" domain={['auto', 'auto']} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="portfolio" stroke="#3b82f6" strokeWidth={3} name="Wealth Utility" />
              <Line type="monotone" dataKey="benchmark" stroke="#9ca3af" strokeWidth={2} name="S&P 500" />
              <Line type="monotone" dataKey="allocation_benchmark" stroke="#d1d5db" strokeWidth={2} strokeDasharray="5 5" name="Alloc Bench" />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Add remaining sections: Drawdown, Rolling Metrics, Heatmap, Table */}
    </div>
  );
};

function prepareEquityCurveData(equityCurve) {
  return equityCurve.dates.map((date, i) => ({
    date: new Date(date).toLocaleDateString(),
    portfolio: equityCurve.portfolio[i],
    benchmark: equityCurve.benchmark[i],
    allocation_benchmark: equityCurve.allocation_benchmark[i]
  }));
}
```

## Testing Checklist
- [ ] Tab navigation works and displays correctly
- [ ] All charts render without errors
- [ ] Performance metrics match API response
- [ ] Charts handle null values gracefully (early months)
- [ ] Loading state displays during API call
- [ ] Error state displays when API fails
- [ ] Refresh button re-fetches data
- [ ] Layout is responsive on mobile, tablet, and desktop
- [ ] Charts are interactive (tooltips work)
- [ ] Export/print functionality works (if implemented)

## Notes
- The backtest endpoint takes ~10 seconds to run on first call, then results are cached for 2 hours
- Early months may have null values for rolling metrics (first 12 months)
- The strategy uses Max Sharpe optimization with 9 equity ETFs and BIL for cash
- Benchmark is SPY (S&P 500)
- Strategy includes value, momentum, and risk dials that dynamically adjust equity allocation
