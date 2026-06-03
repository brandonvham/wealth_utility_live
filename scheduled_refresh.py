import argparse
import json
import os
import sys
from typing import List, Optional

import requests

from scheduling import should_run_now


DEFAULT_API_BASE_URL = "https://wealthutilitylive-production.up.railway.app"
DEFAULT_ARTIFACT_PATH = "current_allocation.json"


def refresh_allocations(
    base_url: str,
    artifact_path: str = DEFAULT_ARTIFACT_PATH,
    timeout_seconds: int = 300,
) -> dict:
    refresh_url = base_url.rstrip("/") + "/allocations/refresh"
    response = requests.post(refresh_url, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()

    if not payload.get("success", False):
        raise RuntimeError(f"Allocation refresh failed: {payload}")

    with open(artifact_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"Refreshed allocations at {refresh_url}")
    print(f"Saved refresh artifact to {artifact_path}")
    return payload


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh Wealth Utility API allocations on schedule.")
    parser.add_argument("--force", action="store_true", help="Refresh even when the schedule guard would skip.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("WEALTH_UTILITY_API_BASE_URL", DEFAULT_API_BASE_URL),
        help="Base URL for the deployed Wealth Utility API.",
    )
    parser.add_argument(
        "--artifact-path",
        default=os.getenv("WEALTH_UTILITY_ARTIFACT_PATH", DEFAULT_ARTIFACT_PATH),
        help="Path where the JSON response should be written.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if not args.force and not should_run_now():
        print("Skipping refresh: not after 5 PM CT on the last NYSE trading day.")
        return 0

    try:
        refresh_allocations(base_url=args.base_url, artifact_path=args.artifact_path)
    except Exception as exc:
        print(f"Refresh failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
