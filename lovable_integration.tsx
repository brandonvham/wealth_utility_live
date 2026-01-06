// Wealth Utility - Lovable.dev Integration Example
// This React/TypeScript component fetches and displays allocation data

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { RefreshCw, TrendingUp, TrendingDown } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

// API Configuration - update with your deployed API URL
const API_BASE_URL = 'http://localhost:5000'; // Change to your deployed URL

interface Allocation {
  ticker: string;
  asset_class: string;
  weight: number;
  weight_pct: string;
}

interface AllocationResponse {
  success: boolean;
  calculation_date: string;
  allocation_date: string;
  allocations: Allocation[];
  summary: {
    total_equity: number;
    total_equity_pct: string;
    total_fixed_income: number;
    total_fixed_income_pct: string;
  };
  strategy_params: {
    sleeve_method: string;
    band_mode: string;
    risk_dial_mode: string;
    value_dial: number;
    momentum_dial: number;
  };
  error?: string;
}

export default function WealthUtilityDashboard() {
  const [data, setData] = useState<AllocationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const { toast } = useToast();

  // Fetch allocations from API
  const fetchAllocations = async (forceRefresh = false) => {
    try {
      setLoading(!forceRefresh);
      setRefreshing(forceRefresh);

      const endpoint = forceRefresh
        ? `${API_BASE_URL}/allocations/refresh`
        : `${API_BASE_URL}/allocations`;

      const method = forceRefresh ? 'POST' : 'GET';

      const response = await fetch(endpoint, { method });
      const result = await response.json();

      if (result.success) {
        setData(result);
        if (forceRefresh) {
          toast({
            title: 'Allocations Refreshed',
            description: 'Portfolio allocations have been recalculated.',
          });
        }
      } else {
        throw new Error(result.error || 'Failed to fetch allocations');
      }
    } catch (error) {
      console.error('Error fetching allocations:', error);
      toast({
        title: 'Error',
        description: 'Failed to load portfolio allocations. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  // Load data on component mount
  useEffect(() => {
    fetchAllocations();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p className="text-muted-foreground">Loading portfolio allocations...</p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="text-center p-8">
        <p className="text-muted-foreground">No allocation data available</p>
      </div>
    );
  }

  // Separate equity and fixed income allocations
  const equityAllocations = data.allocations.filter((a) => a.asset_class === 'equity');
  const fixedIncomeAllocations = data.allocations.filter((a) => a.asset_class === 'fixed_income');

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Wealth Utility Portfolio</h2>
          <p className="text-muted-foreground">
            Allocation as of {new Date(data.allocation_date).toLocaleDateString()}
          </p>
        </div>
        <Button
          onClick={() => fetchAllocations(true)}
          disabled={refreshing}
          variant="outline"
        >
          <RefreshCw className={`mr-2 h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
          {refreshing ? 'Refreshing...' : 'Refresh'}
        </Button>
      </div>

      {/* Summary Cards */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Equity</CardTitle>
            <TrendingUp className="h-4 w-4 text-green-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{data.summary.total_equity_pct}</div>
            <p className="text-xs text-muted-foreground">
              Stocks, ETFs, and Growth Assets
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Fixed Income</CardTitle>
            <TrendingDown className="h-4 w-4 text-blue-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{data.summary.total_fixed_income_pct}</div>
            <p className="text-xs text-muted-foreground">
              Bonds and Safe Haven Assets
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Allocations */}
      <Card>
        <CardHeader>
          <CardTitle>Portfolio Allocation Breakdown</CardTitle>
          <CardDescription>
            Strategy: {data.strategy_params.sleeve_method} | Band Mode: {data.strategy_params.band_mode}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {/* Equity Section */}
            {equityAllocations.length > 0 && (
              <div>
                <h3 className="text-lg font-semibold mb-3">Equity Sleeve</h3>
                <div className="space-y-2">
                  {equityAllocations.map((allocation) => (
                    <div
                      key={allocation.ticker}
                      className="flex items-center justify-between p-3 rounded-lg border"
                    >
                      <div>
                        <p className="font-medium">{allocation.ticker}</p>
                        <p className="text-sm text-muted-foreground capitalize">
                          {allocation.asset_class.replace('_', ' ')}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-lg font-bold">{allocation.weight_pct}</p>
                        <div className="w-32 bg-secondary rounded-full h-2 mt-1">
                          <div
                            className="bg-primary h-2 rounded-full transition-all"
                            style={{ width: allocation.weight_pct }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Fixed Income Section */}
            {fixedIncomeAllocations.length > 0 && (
              <div>
                <h3 className="text-lg font-semibold mb-3">Fixed Income</h3>
                <div className="space-y-2">
                  {fixedIncomeAllocations.map((allocation) => (
                    <div
                      key={allocation.ticker}
                      className="flex items-center justify-between p-3 rounded-lg border"
                    >
                      <div>
                        <p className="font-medium">{allocation.ticker}</p>
                        <p className="text-sm text-muted-foreground capitalize">
                          {allocation.asset_class.replace('_', ' ')}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-lg font-bold">{allocation.weight_pct}</p>
                        <div className="w-32 bg-secondary rounded-full h-2 mt-1">
                          <div
                            className="bg-blue-600 h-2 rounded-full transition-all"
                            style={{ width: allocation.weight_pct }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Strategy Parameters */}
      <Card>
        <CardHeader>
          <CardTitle>Strategy Parameters</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-sm text-muted-foreground">Value Dial</p>
              <p className="text-lg font-semibold">{data.strategy_params.value_dial}%</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Momentum Dial</p>
              <p className="text-lg font-semibold">{data.strategy_params.momentum_dial}%</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Risk Dial</p>
              <p className="text-lg font-semibold capitalize">{data.strategy_params.risk_dial_mode}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Last Updated</p>
              <p className="text-lg font-semibold">
                {new Date(data.calculation_date).toLocaleTimeString()}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
