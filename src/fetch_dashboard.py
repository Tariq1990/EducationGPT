import os
import sys
import json
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load environment variables from .env in project root
project_root = Path(__file__).resolve().parents[1]
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# Retrieve credentials and URL
url = os.getenv("DASHBOARD_URL")
username = os.getenv("DASHBOARD_USER")
password = os.getenv("DASHBOARD_PASS")

if not all([url, username, password]):
    sys.stderr.write("Missing dashboard configuration in .env\n")
    sys.exit(1)

# Perform login POST (the login form uses POST to the same URL)
login_data = {
    "username": username,
    "password": password,
}

session = requests.Session()
resp = session.post(url, data=login_data)
if resp.status_code != 200:
    sys.stderr.write(f"Login failed: HTTP {resp.status_code}\n")
    sys.exit(1)

# After login, attempt to fetch a "reports" page. If a specific endpoint is known, replace below.
# For demonstration, we request the same URL again which typically returns the dashboard after auth.
report_resp = session.get(url)
report_resp.raise_for_status()

# Print the HTML (or JSON) of the dashboard/report page to stdout
print(report_resp.text)
