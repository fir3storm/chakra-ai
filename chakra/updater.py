"""
Auto-updater for Chakra AI. Checks PyPI for newer versions on startup.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib.request import urlopen, Request

UPDATE_CACHE = Path.home() / ".chakra_update_check"
CHECK_INTERVAL = 3600 * 6  # 6 hours between checks
CURRENT_VERSION = "0.2.0"


def _fetch_pypi_version() -> Optional[str]:
    """Fetch latest version from PyPI JSON API."""
    try:
        req = Request(
            "https://pypi.org/pypi/chakra-ai/json",
            headers={"User-Agent": "chakra-ai"},
        )
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return data["info"]["version"]
    except Exception:
        return None


def _fetch_github_version() -> Optional[str]:
    """Fetch latest release tag from GitHub API."""
    try:
        req = Request(
            "https://api.github.com/repos/fir3storm/chakra-ai/releases/latest",
            headers={"User-Agent": "chakra-ai", "Accept": "application/vnd.github.v3+json"},
        )
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            tag = data.get("tag_name", "")
            return tag.lstrip("v")
    except Exception:
        return None


def _should_check() -> bool:
    """Check if enough time has passed since last update check."""
    if not UPDATE_CACHE.exists():
        return True
    try:
        last = float(UPDATE_CACHE.read_text().strip())
        return (time.time() - last) > CHECK_INTERVAL
    except Exception:
        return True


def _mark_checked() -> None:
    """Save timestamp of last check."""
    UPDATE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    UPDATE_CACHE.write_text(str(time.time()))


def check_for_updates(silent: bool = False) -> Tuple[bool, str]:
    """Check for newer version. Returns (update_available, latest_version).

    Args:
        silent: If True, only print when update is available.
    """
    if not _should_check():
        return False, CURRENT_VERSION

    _mark_checked()

    # Try PyPI first, fall back to GitHub
    latest = _fetch_pypi_version()
    if not latest:
        latest = _fetch_github_version()
    if not latest:
        return False, CURRENT_VERSION

    is_newer = _compare_versions(latest, CURRENT_VERSION) > 0
    return is_newer, latest


def _compare_versions(a: str, b: str) -> int:
    """Compare semantic versions. Returns 1 if a > b, -1 if a < b, 0 if equal."""
    try:
        pa = tuple(int(x) for x in a.split("."))
        pb = tuple(int(x) for x in b.split("."))
        return (pa > pb) - (pa < pb)
    except Exception:
        return 0


def print_update_banner(latest_version: str) -> None:
    """Print a friendly update banner."""
    print(f"""
  ╔════════════════════════════════════════════════════════╗
  ║  \033[93m✨ Update Available!\033[0m  v{CURRENT_VERSION} → \033[92mv{latest_version}\033[0m                    ║
  ║                                                        ║
  ║  To upgrade:                                            ║
  ║    pip install --upgrade chakra-ai                      ║
  ║                                                        ║
  ║  Full installer (re-run for clean setup):               ║
  ║    powershell -c "irm .../install.ps1 | iex"            ║
  ╚════════════════════════════════════════════════════════╝
""")


def check_and_notify() -> None:
    """Check for updates and print banner if available. Call on startup."""
    try:
        available, version = check_for_updates()
        if available:
            print_update_banner(version)
    except Exception:
        pass  # Never block startup on update check failure


if __name__ == "__main__":
    available, version = check_for_updates(silent=False)
    if available:
        print_update_banner(version)
    else:
        print(f"Chakra AI v{CURRENT_VERSION} is up to date.")
        if version != CURRENT_VERSION:
            print(f"(PyPI has v{version}, same or older)")
