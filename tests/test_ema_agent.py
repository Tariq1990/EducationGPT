"""End-to-end tests for the EMA Dashboard Automation Agent.

Tests cover:
  1. Scraper initialisation and configuration
  2. Report processor: CSV/XLSX reading, cleaning, metric computation
  3. Tool wrapper integration (metrics summary, query)
"""

import asyncio
import os
import sys
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.ema_agent.report_processor import EMAReportProcessor
from agents.ema_agent.dashboard_scraper import EMADashboardScraper
from agents.ema_agent.ema_tool import (
    download_latest_ema_report,
    get_ema_metrics_summary,
    query_dashboard_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
SAMPLE_DIR = PROJECT_ROOT / "data" / "raw_reports"


def _create_sample_csv(path: Path) -> Path:
    """Write a small CSV that mimics a dashboard report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "school_name": ["School A", "School B", "School C", "School D", "School E"],
        "district": ["Peshawar", "Mardan", "Peshawar", "Swat", "Mardan"],
        "enrollment": [450, 320, 280, 510, 190],
        "attendance": [0.89, 0.76, 0.92, 0.81, 0.65],
        "teachers": [18, 12, 15, 22, 8],
        "performance_score": [72.5, 65.0, 80.1, 78.3, 55.2],
        "facility_index": [0.7, 0.4, 0.8, 0.6, 0.3],
    })
    df.to_csv(path, index=False)
    return path


def _create_sample_xlsx(path: Path) -> Path:
    """Write a small XLSX that mimics a dashboard report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "School Name": ["School X", "School Y"],
        "District": ["Abbottabad", "Haripur"],
        "Total Enrollment": [300, 450],
        "Attendance": [0.85, 0.90],
        "Teachers": [10, 15],
    })
    df.to_excel(path, index=False, engine="openpyxl")
    return path


# ===========================================================================
# Test Cases
# ===========================================================================
class TestReportProcessor(unittest.TestCase):
    """Test the EMAReportProcessor class."""

    def setUp(self):
        self.csv_path = SAMPLE_DIR / "_test_sample.csv"
        self.xlsx_path = SAMPLE_DIR / "_test_sample.xlsx"
        _create_sample_csv(self.csv_path)
        _create_sample_xlsx(self.xlsx_path)
        self.processor = EMAReportProcessor(download_dir=str(SAMPLE_DIR))

    def tearDown(self):
        for p in (self.csv_path, self.xlsx_path):
            if p.exists():
                p.unlink()

    def test_discover_files(self):
        files = self.processor.discover_files()
        self.assertGreaterEqual(len(files), 2, "Should discover test CSV and XLSX")

    def test_read_csv(self):
        df = EMAReportProcessor.read_file(self.csv_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5)

    def test_read_xlsx(self):
        df = EMAReportProcessor.read_file(self.xlsx_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)

    def test_clean(self):
        raw = EMAReportProcessor.read_file(self.csv_path)
        cleaned = EMAReportProcessor.clean(raw)
        self.assertFalse(cleaned.empty)
        # Column names should be lowercase+underscored
        for col in cleaned.columns:
            self.assertEqual(col, col.lower())
            self.assertNotIn(" ", col)

    def test_compute_metrics(self):
        raw = EMAReportProcessor.read_file(self.csv_path)
        cleaned = EMAReportProcessor.clean(raw)
        cleaned = EMAReportProcessor.normalise_columns(cleaned)
        metrics = EMAReportProcessor.compute_metrics(cleaned)
        self.assertEqual(metrics["total_records"], 5)
        self.assertIn("numeric_summary", metrics)
        print("\n=== Computed Metrics ===")
        for k, v in metrics.items():
            if k != "numeric_summary":
                print(f"  {k}: {v}")

    def test_process_file_csv(self):
        result = self.processor.process_file(self.csv_path)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["shape"][0], 5)
        print("\n=== CSV Head ===")
        print(result["dataframe"].head())

    def test_process_file_xlsx(self):
        result = self.processor.process_file(self.xlsx_path)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["shape"][0], 2)

    def test_metrics_to_context(self):
        result = self.processor.process_file(self.csv_path)
        ctx = EMAReportProcessor.metrics_to_context(result["metrics"])
        self.assertIsInstance(ctx, str)
        self.assertIn("EMA Report Metrics", ctx)
        print("\n=== LLM Context Text ===")
        print(ctx)


class TestEMATools(unittest.TestCase):
    """Test the tool wrapper functions."""

    def setUp(self):
        self.csv_path = SAMPLE_DIR / "_tool_test_sample.csv"
        _create_sample_csv(self.csv_path)

    def tearDown(self):
        if self.csv_path.exists():
            self.csv_path.unlink()

    def test_get_metrics_summary(self):
        summary = get_ema_metrics_summary()
        self.assertIn("status", summary)
        if summary["status"] == "ok":
            self.assertIn("metrics", summary)
            self.assertIn("context", summary)
            print("\n=== Metrics Summary ===")
            print(summary["context"])

    def test_query_dashboard_data(self):
        result = query_dashboard_data(district="Peshawar")
        self.assertIn("status", result)
        if result["status"] == "ok":
            print(f"\n=== Query Result: {result['total_matched']} matches ===")
            for rec in result["records"][:3]:
                print(f"  {rec}")

    def test_query_freetext(self):
        result = query_dashboard_data(query="School A")
        self.assertIn("status", result)


class TestScraperInit(unittest.TestCase):
    """Test scraper initialisation (no actual browser needed)."""

    def test_default_config(self):
        scraper = EMADashboardScraper()
        self.assertIn("175.107.63.44", scraper.url)
        self.assertEqual(scraper.username, "dmo.mohmand")
        self.assertTrue(scraper.headless)

    def test_custom_config(self):
        scraper = EMADashboardScraper(
            url="http://example.com",
            username="test_user",
            password="test_pass",
            download_dir=str(SAMPLE_DIR),
            headless=False,
        )
        self.assertEqual(scraper.url, "http://example.com")
        self.assertEqual(scraper.username, "test_user")
        self.assertFalse(scraper.headless)


if __name__ == "__main__":
    unittest.main(verbosity=2)
