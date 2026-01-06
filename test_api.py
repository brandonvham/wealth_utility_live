"""
Test script for Wealth Utility API
"""

import requests
import json
import time
from datetime import datetime

API_BASE_URL = "http://localhost:5000"

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def test_endpoint(method, endpoint, description):
    """Test an API endpoint and display results"""
    url = f"{API_BASE_URL}{endpoint}"

    print(f"\n🔍 Testing: {description}")
    print(f"   {method} {url}")
    print("-" * 80)

    try:
        start = time.time()

        if method == "GET":
            response = requests.get(url, timeout=120)
        elif method == "POST":
            response = requests.post(url, timeout=120)
        else:
            print(f"❌ Unknown method: {method}")
            return False

        elapsed = time.time() - start

        print(f"   Status: {response.status_code}")
        print(f"   Time: {elapsed:.2f}s")

        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success!")
            print(f"\n   Response Preview:")
            print("   " + "-" * 76)

            # Pretty print the JSON with indentation
            formatted = json.dumps(data, indent=2)
            # Only show first 20 lines
            lines = formatted.split('\n')
            for line in lines[:20]:
                print(f"   {line}")
            if len(lines) > 20:
                print(f"   ... ({len(lines) - 20} more lines)")

            return True
        else:
            print(f"   ❌ Failed with status {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"   ❌ Connection failed - Is the API server running?")
        return False
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False

def main():
    """Run all API tests"""
    print_section("WEALTH UTILITY API - COMPREHENSIVE TEST")
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API URL: {API_BASE_URL}")

    results = {}

    # Test 1: Home/Documentation
    print_section("TEST 1: API Documentation")
    results['home'] = test_endpoint("GET", "/", "Get API documentation")

    # Test 2: Health Check
    print_section("TEST 2: Health Check")
    results['health'] = test_endpoint("GET", "/health", "Check API health status")

    # Test 3: Configuration
    print_section("TEST 3: Strategy Configuration")
    results['config'] = test_endpoint("GET", "/config", "Get strategy parameters")

    # Test 4: Get Allocations (with caching)
    print_section("TEST 4: Get Current Allocations (Cached)")
    results['allocations'] = test_endpoint("GET", "/allocations", "Get portfolio allocations (may use cache)")

    # Test 5: Refresh Allocations
    print_section("TEST 5: Force Refresh Allocations")
    print("\n⚠️  This will recalculate everything - may take 10-30 seconds...")
    results['refresh'] = test_endpoint("POST", "/allocations/refresh", "Force recalculate allocations")

    # Test 6: Get Allocations Again (should be fast - cached)
    print_section("TEST 6: Get Allocations Again (Should Be Fast)")
    results['allocations_cached'] = test_endpoint("GET", "/allocations", "Get cached allocations")

    # Summary
    print_section("TEST SUMMARY")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed

    print(f"\n   Total Tests: {total}")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   Success Rate: {(passed/total)*100:.1f}%")

    print("\n" + "="*80)

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Your API is working perfectly!")
        print("\nNext Steps:")
        print("  1. Keep the API server running")
        print("  2. Deploy to Railway/Render for production")
        print("  3. Update the API URL in your Lovable component")
        print("  4. Test from your Lovable app")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        print("\nTroubleshooting:")
        print("  - Make sure the API server is running: python wealth_utility_api.py")
        print("  - Check that all dependencies are installed: pip install -r requirements_api.txt")
        print("  - Verify ecy4.xlsx file is in the correct location")
        print("  - Check API keys (FMP_KEY, FRED_API_KEY)")

    print("="*80 + "\n")

if __name__ == "__main__":
    # Check if API server is running
    print("\n" + "="*80)
    print("WEALTH UTILITY API TEST SUITE")
    print("="*80)
    print("\n⚠️  Make sure the API server is running before testing!")
    print("\nIn another terminal, run:")
    print("  cd \"C:\\Users\\BrandonVanLandingham\\OneDrive - Perissos Private Wealth Management\\1 Perissos Private Wealth Management\\5 Python\\Wealth Utility\"")
    print("  python wealth_utility_api.py")
    print("\nPress Enter to start testing...")
    input()

    main()
