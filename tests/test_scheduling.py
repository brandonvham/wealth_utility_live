from datetime import datetime
import unittest

import pytz

from scheduling import is_after_cutoff, is_last_nyse_trading_day, should_run_now


CENTRAL = pytz.timezone("America/Chicago")


def ct(year, month, day, hour, minute=0):
    return CENTRAL.localize(datetime(year, month, day, hour, minute))


class SchedulingTests(unittest.TestCase):
    def test_is_after_cutoff_false_before_5pm_central(self):
        self.assertFalse(is_after_cutoff(ct(2026, 6, 30, 16, 59)))

    def test_is_after_cutoff_true_at_5pm_central(self):
        self.assertTrue(is_after_cutoff(ct(2026, 6, 30, 17, 0)))

    def test_last_nyse_trading_day_true_for_normal_month_end(self):
        self.assertTrue(is_last_nyse_trading_day(ct(2026, 6, 30, 17, 0)))

    def test_last_nyse_trading_day_false_for_prior_business_day(self):
        self.assertFalse(is_last_nyse_trading_day(ct(2026, 6, 29, 17, 0)))

    def test_last_nyse_trading_day_handles_good_friday_month_end(self):
        # NYSE was closed Friday 2024-03-29 for Good Friday, so Thursday
        # 2024-03-28 was the final NYSE trading day of March 2024.
        self.assertTrue(is_last_nyse_trading_day(ct(2024, 3, 28, 17, 0)))

    def test_should_run_now_requires_last_trading_day_and_cutoff(self):
        self.assertTrue(should_run_now(ct(2026, 6, 30, 17, 1)))
        self.assertFalse(should_run_now(ct(2026, 6, 30, 16, 59)))
        self.assertFalse(should_run_now(ct(2026, 6, 29, 17, 1)))


if __name__ == "__main__":
    unittest.main()
