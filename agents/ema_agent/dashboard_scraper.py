"""Dual-mode scraper for the KPEMA/IMU Monitoring Dashboard.

Provides:
  1. Fast HTTP Session Scraper (instant requests-based POST auth & HTML/JSON extraction)
  2. Full Headless Playwright Scraper (browser automation for download buttons and JS tables)
  3. Offline Fallback Parser (extracts metrics from saved HTML files in data/raw_reports)
"""

import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from agents.ema_agent.config import (
    DOWNLOAD_DIR,
    EMA_DASHBOARD_PASS,
    EMA_DASHBOARD_URL,
    EMA_DASHBOARD_USER,
    MOHMAND_TEHSILS,
    PROJECT_ROOT,
)


def extract_js_variables_from_html(html_text: str) -> Dict[str, Any]:
    """Extracts all embedded JavaScript data models from the KPEMA dashboard HTML."""
    extracted: Dict[str, Any] = {}

    patterns = {
        "surveydata": r"var\s+surveydata\s*=\s*(\{.*?\});",
        "schoolStatusData": r"var\s+schoolStatusData\s*=\s*(\{.*?\});",
        "TeacherPresenceData": r"var\s+TeacherPresenceData\s*=\s*(\{.*?\});",
        "non_teacherPresence": r"var\s+non_teacherPresence\s*=\s*(\{.*?\});",
        "facilities": r"var\s+facilities\s*=\s*(\{.*?\});",
        "school_occupancy": r"var\s+school_occupancy\s*=\s*(\{.*?\});",
        "students_attendance": r"var\s+students_attendance\s*=\s*(\{.*?\});",
        "villegalBuildingStatus": r"var\s+villegalBuildingStatus\s*=\s*(\{.*?\});",
        "school_without": r"var\s+school_without\s*=\s*(\{.*?\});",
        "TopDistricts": r"var\s+TopDistricts\s*=\s*(\{.*?\});",
        "vacantTeachers": r"var\s+vacantTeachers\s*=\s*(\{.*?\});",
        "singleTeacherSchools": r"var\s+singleTeacherSchools\s*=\s*(\{.*?\});",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, html_text, re.DOTALL)
        if match:
            raw_json = match.group(1).strip()
            try:
                extracted[key] = json.loads(raw_json)
            except Exception:
                extracted[key] = None

    # Also capture simple variable assignments
    dist_match = re.search(r"var\s+selected_district\s*=\s*['\"]([^'\"]+)['\"];", html_text)
    if dist_match:
        extracted["selected_district"] = dist_match.group(1)

    date_match = re.search(r"var\s+selectedDate\s*=\s*['\"]([^'\"]+)['\"];", html_text)
    if date_match:
        extracted["selectedDate"] = date_match.group(1)

    # Scrape total and surveyed counts from DOM if present
    soup = BeautifulSoup(html_text, "html.parser")
    surveyed_span = soup.find("span", class_="surveyed-schools")
    total_span = soup.find("span", class_="total-schools")
    if surveyed_span and total_span:
        extracted["dom_surveyed"] = surveyed_span.get_text(strip=True)
        extracted["dom_total"] = total_span.get_text(strip=True)

    return extracted


class HTTPDashboardClient:
    """Fast, lightweight HTTP client for authenticated EMA dashboard interaction."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        download_dir: Optional[Path] = None,
    ):
        self.base_url = base_url or EMA_DASHBOARD_URL
        self.username = username or EMA_DASHBOARD_USER
        self.password = password or EMA_DASHBOARD_PASS
        self.download_dir = Path(download_dir or DOWNLOAD_DIR)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) EducationGPT/1.0"
        })
        self._is_logged_in = False

    def login(self) -> bool:
        """Authenticates with the dashboard via POST."""
        try:
            login_data = {
                "username": self.username,
                "password": self.password,
            }
            resp = self.session.post(self.base_url, data=login_data, timeout=12)
            if resp.status_code == 200 and ("logout.php" in resp.text or "dashboard" in resp.text.lower()):
                self._is_logged_in = True
                return True
            # Even if redirect status isn't 200, check cookies
            if "PHPSESSID" in self.session.cookies:
                self._is_logged_in = True
                return True
        except Exception as e:
            print(f"[HTTPDashboardClient] Live login failed: {e}")
        return False

    def fetch_dashboard_page(
        self,
        district: str = "mohmand",
        date_str: str = "Aug-2026",
        save_snapshot: bool = True,
    ) -> Optional[str]:
        """Loads the dashboard view for the specified district and date."""
        if not self._is_logged_in:
            ok = self.login()
            if not ok:
                return None

        try:
            # Query the dashboard form endpoint
            action_url = self.base_url.replace("index.php", "dashboard.php")
            payload = {
                "dashboard_date": date_str,
                "district": district,
                "thesil": "",
                "ucs": "",
                "gender": "",
                "school_level": "",
                "SearchDistrict": "Search",
            }
            resp = self.session.post(action_url, data=payload, timeout=15)
            if resp.status_code == 200:
                html = resp.text
                if save_snapshot:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = self.download_dir / f"dashboard_page_{ts}.html"
                    out_path.write_text(html, encoding="utf-8")
                return html
        except Exception as e:
            print(f"[HTTPDashboardClient] Error fetching dashboard page: {e}")
        return None

    def fetch_report(
        self,
        report_slug: str,
        district: str = "mohmand",
        date_range: str = "08-01-2026 - 08-30-2026",
    ) -> Optional[str]:
        """Fetches a specific drill-down report URL based on ema_website_structure.txt."""
        if not self._is_logged_in:
            if not self.login():
                return None

        report_url = self.base_url.replace("index.php", "reports.php")
        params = {
            "r": report_slug,
            "districtCommaSep": district,
            "reports_start_date": date_range,
        }
        try:
            resp = self.session.get(report_url, params=params, timeout=20)
            if resp.status_code == 200:
                return resp.text
        except Exception as e:
            print(f"[HTTPDashboardClient] Failed to fetch report '{report_slug}': {e}")
        return None


class EMADashboardScraper:
    """Async Playwright scraper for deep interactive headless browser automation."""

    def __init__(
        self,
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        download_dir: Optional[Path] = None,
        headless: bool = True,
    ):
        self.url = url or EMA_DASHBOARD_URL
        self.username = username or EMA_DASHBOARD_USER
        self.password = password or EMA_DASHBOARD_PASS
        self._download_dir = Path(download_dir or DOWNLOAD_DIR)
        self._download_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless

        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    async def start(self):
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise RuntimeError("Playwright is not installed.") from exc

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._context = await self._browser.new_context(
            accept_downloads=True,
            viewport={"width": 1280, "height": 800},
        )
        self._page = await self._context.new_page()

    async def stop(self):
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

    async def login(self) -> bool:
        for attempt in range(1, 4):
            try:
                await self._page.goto(self.url, timeout=30_000)
                await self._page.wait_for_load_state("networkidle", timeout=15_000)

                user_input = await self._page.query_selector(
                    "input[name='username'], input[name='user'], #username"
                )
                pass_input = await self._page.query_selector(
                    "input[name='password'], input[name='pass'], #password"
                )

                if not user_input or not pass_input:
                    return False

                await user_input.fill(self.username)
                await pass_input.fill(self.password)

                submit_btn = await self._page.query_selector("button[type='submit'], input[type='submit']")
                if submit_btn:
                    await submit_btn.click()
                else:
                    await pass_input.press("Enter")

                await self._page.wait_for_load_state("networkidle", timeout=20_000)
                if "login" not in self._page.url.lower():
                    return True
            except Exception as e:
                print(f"[PlaywrightScraper] Attempt {attempt} failed: {e}")
                await asyncio.sleep(2)
        return False

    async def download_reports(self) -> List[Path]:
        downloaded: List[Path] = []
        export_selectors = [
            "a[href$='.csv']",
            "a[href$='.xlsx']",
            "button:has-text('Export')",
            "button:has-text('CSV')",
            "button:has-text('Excel')",
            ".dt-button.buttons-csv",
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
                        dest = self._download_dir / f"ema_report_{ts}_{download.suggested_filename}"
                        await download.save_as(dest)
                        downloaded.append(dest)
            except Exception:
                continue

        # Save HTML dump
        html_content = await self._page.content()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_html = self._download_dir / f"dashboard_page_{ts}.html"
        dest_html.write_text(html_content, encoding="utf-8")
        downloaded.append(dest_html)
        return downloaded

    async def run(self) -> List[Path]:
        await self.start()
        try:
            ok = await self.login()
            if not ok:
                return []
            return await self.download_reports()
        finally:
            await self.stop()


def get_latest_cached_dashboard_html() -> Optional[str]:
    """Retrieves the newest cached dashboard HTML snapshot from data/raw_reports."""
    if not DOWNLOAD_DIR.exists():
        return None
    html_files = sorted(DOWNLOAD_DIR.glob("*.html"), key=os.path.getmtime, reverse=True)
    if html_files:
        return html_files[0].read_text(encoding="utf-8", errors="replace")
    return None
