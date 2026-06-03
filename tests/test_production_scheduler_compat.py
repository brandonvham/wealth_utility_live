import os
import unittest
from unittest.mock import patch


# `wealth_utility_production.py` validates these at import time. These dummy
# values keep the compatibility test isolated from local or CI secrets.
os.environ["FMP_KEY"] = "test-fmp-key"
os.environ["FRED_API_KEY"] = "test-fred-key"

import wealth_utility_production as production


class ProductionSchedulerCompatTests(unittest.TestCase):
    def test_last_trading_day_delegates_to_scheduling_module(self):
        with patch.object(production.scheduling, "is_last_nyse_trading_day", return_value=True) as guard:
            self.assertTrue(production.is_last_trading_day_of_month())

        guard.assert_called_once_with()

    def test_should_run_now_delegates_and_returns_true(self):
        with patch.object(production.scheduling, "should_run_now", return_value=True) as guard:
            self.assertTrue(production.should_run_now())

        guard.assert_called_once_with()

    def test_should_run_now_delegates_and_returns_false(self):
        with patch.object(production.scheduling, "should_run_now", return_value=False) as guard:
            self.assertFalse(production.should_run_now())

        guard.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
