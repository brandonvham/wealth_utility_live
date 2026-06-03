import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import scheduled_refresh


class ScheduledRefreshTests(unittest.TestCase):
    def test_default_api_base_url_points_to_live_railway_app(self):
        self.assertEqual(
            scheduled_refresh.DEFAULT_API_BASE_URL,
            "https://wealthutilitylive-production.up.railway.app",
        )

    def test_main_skips_when_scheduler_says_not_to_run(self):
        with patch("scheduled_refresh.should_run_now", return_value=False), \
             patch("scheduled_refresh.refresh_allocations") as refresh:
            exit_code = scheduled_refresh.main([])
        self.assertEqual(exit_code, 0)
        refresh.assert_not_called()

    def test_main_force_refresh_bypasses_schedule_guard(self):
        with patch("scheduled_refresh.should_run_now", return_value=False), \
             patch("scheduled_refresh.refresh_allocations", return_value={"success": True}) as refresh:
            exit_code = scheduled_refresh.main(["--force"])
        self.assertEqual(exit_code, 0)
        refresh.assert_called_once()

    def test_refresh_allocations_posts_to_refresh_endpoint_and_writes_artifact(self):
        response = Mock()
        response.json.return_value = {"success": True, "allocation_date": "2026-06-30"}
        response.raise_for_status.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir, \
             patch("scheduled_refresh.requests.post", return_value=response):
            artifact_path = os.path.join(tmpdir, "current_allocation.json")
            payload = scheduled_refresh.refresh_allocations(
                base_url="https://wealthutilitylive-production.up.railway.app",
                artifact_path=artifact_path,
            )

            self.assertEqual(payload["allocation_date"], "2026-06-30")
            with open(artifact_path, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
            self.assertEqual(saved["allocation_date"], "2026-06-30")

    def test_refresh_allocations_fails_when_api_reports_failure(self):
        response = Mock()
        response.json.return_value = {"success": False, "error": "bad data"}
        response.raise_for_status.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir, \
             patch("scheduled_refresh.requests.post", return_value=response), \
             self.assertRaises(RuntimeError):
            scheduled_refresh.refresh_allocations(
                base_url="https://wealthutilitylive-production.up.railway.app",
                artifact_path=os.path.join(tmpdir, "current_allocation.json"),
            )


if __name__ == "__main__":
    unittest.main()
