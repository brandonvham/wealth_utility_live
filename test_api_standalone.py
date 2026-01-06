"""
Standalone API Test - Tests the calculation logic directly
No need to run the Flask server
"""

import json
from datetime import datetime

print("="*80)
print("WEALTH UTILITY API - STANDALONE TEST")
print("="*80)
print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Import the API function
print("[*] Importing API functions...")
try:
    from wealth_utility_api import get_current_allocations_json
    print("   [OK] Successfully imported API module")
except Exception as e:
    print(f"   [ERROR] Failed to import: {e}")
    exit(1)

print()
print("="*80)
print("TEST 1: Calculate Current Allocations")
print("="*80)
print()
print("[*] Running allocation calculation...")
print("   This may take 10-30 seconds on first run...")
print()

try:
    # Call the calculation function
    start_time = datetime.now()
    result = get_current_allocations_json()
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    print(f"   [OK] Calculation completed in {elapsed:.2f} seconds")
    print()

    # Check if successful
    if result.get('success'):
        print("="*80)
        print("RESULTS - API Response")
        print("="*80)
        print()

        # Display formatted JSON
        print(json.dumps(result, indent=2))

        print()
        print("="*80)
        print("ALLOCATION BREAKDOWN")
        print("="*80)
        print()

        # Display summary
        print(">> SUMMARY:")
        print(f"   Allocation Date: {result['allocation_date']}")
        print(f"   Total Equity:    {result['summary']['total_equity_pct']}")
        print(f"   Total Fixed Inc: {result['summary']['total_fixed_income_pct']}")
        print()

        # Display individual allocations
        print(">> EQUITY SLEEVE:")
        equity_allocs = [a for a in result['allocations'] if a['asset_class'] == 'equity']
        for alloc in equity_allocs:
            print(f"   {alloc['ticker']:15s} {alloc['weight_pct']:>8s}")

        print()
        print(">> FIXED INCOME:")
        fixed_allocs = [a for a in result['allocations'] if a['asset_class'] == 'fixed_income']
        for alloc in fixed_allocs:
            print(f"   {alloc['ticker']:15s} {alloc['weight_pct']:>8s}")

        print()
        print(">> STRATEGY PARAMETERS:")
        params = result['strategy_params']
        print(f"   Sleeve Method:   {params['sleeve_method']}")
        print(f"   Band Mode:       {params['band_mode']}")
        print(f"   Risk Dial Mode:  {params['risk_dial_mode']}")
        print(f"   Value Dial:      {params['value_dial']}%")
        print(f"   Momentum Dial:   {params['momentum_dial']}%")

        print()
        print("="*80)
        print("[OK] API TEST PASSED!")
        print("="*80)
        print()
        print("Your API is working correctly and ready to use!")
        print()
        print("Next Steps:")
        print("  1. Start the Flask server: python wealth_utility_api.py")
        print("  2. Open test_api.html in your browser")
        print("  3. Or deploy to Railway/Render for production use")
        print()

    else:
        print("="*80)
        print("[ERROR] API TEST FAILED")
        print("="*80)
        print()
        print("Error:", result.get('error', 'Unknown error'))
        if 'traceback' in result:
            print()
            print("Traceback:")
            print(result['traceback'])

except Exception as e:
    print(f"   [ERROR] Error during calculation: {e}")
    print()
    import traceback
    traceback.print_exc()
    print()
    print("Troubleshooting:")
    print("  - Check that ecy4.xlsx is in the correct location")
    print("  - Verify API keys (FMP_KEY, FRED_API_KEY) are valid")
    print("  - Ensure all dependencies are installed: pip install -r requirements_api.txt")

print()
print("="*80)
