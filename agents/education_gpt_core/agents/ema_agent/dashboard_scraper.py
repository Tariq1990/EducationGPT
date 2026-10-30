"""Playwright-based headless browser automation for the EMA/IMU Dashboard.

Handles login, navigation, and report download with retry logic.
"""

import asyncio
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment & Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

DASHBOARD_URL: str = os.getenv(
    "EMA_DASHBOARD_URL",
    os.getenv("DASHBOARD_URL", "http://175.107.63.44:81/imudashboard/index.php"),
)
DASHBOARD_USER: str = os.getenv(
    "EMA_DASHBOARD_USER", os.getenv("DASHBOARD_USER", "dmo.mohmand")
)
DASHBOARD_PASS: str = os.getenv(
    "EMA_DASHBOARD_PASS", os.getenv("DASHBOARD_PASS", "imu@mohmand")
)

_env_download_dir = os.getenv("EMA_DOWNLOAD_DIR", "./data/raw_reports")
_download_candidate = Path(_env_download_dir)
if not _download_candidate.is_absolute():
    DOWNLOAD_DIR: str = str((PROJECT_ROOT / _env_download_dir).resolve())
else:
    DOWNLOAD_DIR: str = str(_download_candidate.resolve())

MAX_RETRIES = 3
RETRY_DELAY_S = 5


class EMADashboardScraper:
    """Async Playwright scraper for the EMA/IMU monitoring dashboard."""

    def __init__(
        self,
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        download_dir: Optional[str] = None,
        headless: bool = True,
    ):
        self.url = url or DASHBOARD_URL
        self.username = username or DASHBOARD_USER or "dmo.mohmand"
        self.password = password or DASHBOARD_PASS or "imu@mohmand"
        self._download_dir = Path(download_dir or DOWNLOAD_DIR)
        self._download_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless

        # Playwright objects — initialised in .start()
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    async def start(self):
        """Launch browser and create a context with download support."""
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is not installed. Please install playwright via 'pip install playwright' and run 'playwright install chromium'."
            ) from exc

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._context = await self._browser.new_context(
            accept_downloads=True,
            viewport={"width": 1280, "height": 800},
        )
        self._page = await self._context.new_page()

    async def stop(self):
        """Gracefully close everything."""
        if self._context:
            try:
                await self._context.close()
            except Exception:
                pass
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Login
    # ------------------------------------------------------------------
    async def login(self) -> bool:
        """Navigate to the dashboard and authenticate.

        Returns True on successful login.
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"[EMA] Login attempt {attempt}/{MAX_RETRIES} to {self.url} ...")
                await self._page.goto(self.url, timeout=30_000)
                await self._page.wait_for_load_state("networkidle", timeout=15_000)

                # --- Fill credentials ---
                user_input = await self._find_input(
                    ["input[name='username']", "input[name='user']", "#username", "#user", "input[type='text']"]
                )
                pass_input = await self._find_input(
                    ["input[name='password']", "input[name='pass']", "#password", "#pass", "input[type='password']"]
                )

                if not user_input or not pass_input:
                    print("[EMA] Could not locate username/password fields on page.")
                    return False

                await user_input.fill(self.username)
                await pass_input.fill(self.password)

                # --- Submit ---
                submit_btn = await self._page.query_selector(
                    "button[type='submit'], input[type='submit'], .btn-login, #login-btn"
                )
                if submit_btn:
                    await submit_btn.click()
                else:
                    await pass_input.press("Enter")

                # Wait for navigation / dashboard indicator
                await self._page.wait_for_load_state("networkidle", timeout=20_000)

                # Verify we are no longer on the login page
                current_url = self._page.url
                if "login" not in current_url.lower():
                    print(f"[EMA] Login successful. Current URL: {current_url}")
                    return True

                print(f"[EMA] Still on login URL ({current_url}), retrying ...")

            except Exception as exc:
                print(f"[EMA] Attempt {attempt} failed: {exc}")

            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY_S)

        return False

    # ------------------------------------------------------------------
    # Navigation & Download
    # ------------------------------------------------------------------
    async def navigate_to_reports(self) -> bool:
        """Find and navigate to the reports / monitoring section."""
        report_selectors = [
            "a[href*='report']",
            "a[href*='monitoring']",
            "a[href*='download']",
            "text=Reports",
            "text=Monthly Report",
            "text=Monitoring Data",
        ]
        for sel in report_selectors:
            try:
                el = await self._page.query_selector(sel)
                if el and await el.is_visible():
                    print(f"[EMA] Navigating via selector: {sel}")
                    await el.click()
                    await self._page.wait_for_load_state("networkidle", timeout=15_000)
                    return True
            except Exception:
                continue

        print("[EMA] No dedicated report link found; will attempt page scrape from current view.")
        return False

    async def download_reports(self) -> List[Path]:
        """Trigger any available download buttons/links and return saved paths."""
        downloaded: List[Path] = []
        await self.navigate_to_reports()

        # Look for export / download buttons (CSV, Excel)
        export_selectors = [
            "a[href$='.csv']",
            "a[href$='.xlsx']",
            "a[href$='.xls']",
            "button:has-text('Export')",
            "button:has-text('Download')",
            "button:has-text('CSV')",
            "button:has-text('Excel')",
            ".dt-button.buttons-csv",
            ".dt-button.buttons-excel",
        ]

        for sel in export_selectors:
            try:
                buttons = await self._page.query_selector_all(sel)
                for btn in buttons:
                    if await btn.is_visible():
                        async with self._page.expect_download(timeout=10_000) as download_info:
                            await btn.click()
                        download = await download_info.value
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        suggested_name = download.suggested_filename
                        dest = self._download_dir / f"ema_report_{ts}_{suggested_name}"
                        await download.save_as(dest)
                        print(f"[EMA] Downloaded: {dest}")
                        downloaded.append(dest)
            except Exception:
                continue

        # If no explicit file download was triggered, scrape tabular data from page
        if not downloaded:
            print("[EMA] No file download triggered; saving current dashboard view.")
            scraped = await self._save_page_table()
            downloaded.extend(scraped)

        return downloaded

    async def _save_page_table(self) -> List[Path]:
        """Fallback: dump visible tables to CSV / HTML."""
        saved: List[Path] = []
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        html_content = await self._page.content()
        html_path = self._download_dir / f"dashboard_page_{ts}.html"
        html_path.write_text(html_content, encoding="utf-8")
        print(f"[EMA] Saved page HTML to: {html_path}")
        saved.append(html_path)

        return saved

    # ------------------------------------------------------------------
    # High-level: login + download in one call
    # ------------------------------------------------------------------
    async def run(self) -> List[Path]:
        """Full pipeline: start browser → login → download → stop."""
        await self.start()
        try:
            ok = await self.login()
            if not ok:
                print("[EMA] Login ultimately failed after retries.")
                return []
            return await self.download_reports()
        finally:
            await self.stop()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    async def _find_input(self, selectors: list):
        """Return the first visible element matching one of *selectors*."""
        for sel in selectors:
            try:
                el = await self._page.query_selector(sel)
                if el and await el.is_visible():
                    return el
            except Exception:
                continue
        return None


# ---------------------------------------------------------------------------
# Convenience sync wrapper
# ---------------------------------------------------------------------------
def run_scraper_sync(**kwargs) -> List[Path]:
    """Run the async scraper safely from synchronous code."""
    import concurrent.futures

    scraper = EMADashboardScraper(**kwargs)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, scraper.run()).result()
    return asyncio.run(scraper.run())


if __name__ == "__main__":
    files = run_scraper_sync()
    print(f"\n[EMA] Downloaded {len(files)} file(s): {files}")
