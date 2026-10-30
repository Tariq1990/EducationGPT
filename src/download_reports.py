import os
import sys
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load .env
project_root = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=project_root / ".env")

url = os.getenv("DASHBOARD_URL")
username = os.getenv("DASHBOARD_USER")
password = os.getenv("DASHBOARD_PASS")
if not all([url, username, password]):
    sys.stderr.write("Missing dashboard env vars\n")
    sys.exit(1)

session = requests.Session()
login_resp = session.post(url, data={"username": username, "password": password})
if login_resp.status_code != 200:
    sys.stderr.write(f"Login failed: {login_resp.status_code}\n")
    sys.exit(1)

# Get the dashboard page after login
page_resp = session.get(url)
page_resp.raise_for_status()

soup = BeautifulSoup(page_resp.text, "html.parser")
report_links = []
for a in soup.find_all("a", href=True):
    href = a["href"]
    if any(href.lower().endswith(ext) for ext in [".csv", ".xls", ".xlsx", ".pdf", ".json"]):
        # Build absolute URL if needed
        if href.startswith("http"):
            full_url = href
        else:
            full_url = requests.compat.urljoin(url, href)
        report_links.append(full_url)

if not report_links:
    sys.stderr.write("No report files found on the dashboard page.\n")
    sys.exit(0)

# Ensure destination folder exists
reports_dir = project_root / "data" / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

for link in report_links:
    filename = link.split('/')[-1]
    dest_path = reports_dir / filename
    r = session.get(link)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(r.content)
    print(f"Downloaded {filename} to {dest_path}")
