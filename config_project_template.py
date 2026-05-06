from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
city = os.environ.get("QC_PROJECT_ID", PROJECT_ROOT.name)

first_date = os.environ.get("QC_FIRST_DATE", "01-01-2021 00:00")
last_date = os.environ.get("QC_LAST_DATE", "31-12-2021 23:00")
lat = float(os.environ.get("QC_PROJECT_LAT", "0"))
long = float(os.environ.get("QC_PROJECT_LON", "0"))
extent = None
plot = False

cwd_data_raw = str(PROJECT_ROOT / "data" / "0_raw")
cwd_data_raw_netatmo = str(PROJECT_ROOT / "data" / "0_raw" / "netatmo")
cwd_data_raw_wunder = str(PROJECT_ROOT / "data" / "0_raw" / "wunderground")
cwd_data_raw_ows = str(PROJECT_ROOT / "data" / "0_raw" / "ows")
cwd_data_meta = str(PROJECT_ROOT / "data" / "0_metadata")
cwd_data_str = str(PROJECT_ROOT / "data" / "1_structured")
cwd_data_qc = str(PROJECT_ROOT / "data" / "2_filtered")
cwd_results = str(PROJECT_ROOT / "results")
cwd_results_str = str(PROJECT_ROOT / "results" / "11_data_structure")
cwd_results_qc = str(PROJECT_ROOT / "results" / "20_quality_control")

color_net = "#1f77b4"
color_wund = "#ff7f0e"
color_ows = "#2ca02c"
color_cws = "#9467bd"
color_outliers = "#d62728"
