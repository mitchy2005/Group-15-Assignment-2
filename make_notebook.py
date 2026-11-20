from pathlib import Path
import subprocess
import sys

# Ensure nbformat, nbconvert, and ipykernel are available
for pkg in ["nbformat", "nbconvert", "ipykernel"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

DATA_DIR = Path(__file__).resolve().parent
py_path = DATA_DIR / "hasts206_demographic_pipeline.py"
ipynb_path = DATA_DIR / "HASTS206_Applied_Demographic_Analysis.ipynb"
html_path = DATA_DIR / "HASTS206_Applied_Demographic_Analysis.html"

code_text = py_path.read_text(encoding="utf-8")

# Split the script cleanly into logical notebook cells at major section banners
markers = [
    ("# STEP 0:", "## Step 0: Data Cleaning, Quality Audit & Reconciliation\n*Verifying schemas, primary keys, referential integrity, missing piped-water values (`D04`, `D17`), and the 2025 demographic balancing equation.*"),
    ("# STEP 1:", "## Step 1: Population Structure, Vital Rates & Abridged Life Table (20 Marks)\n*Comparing contrasting districts (`D05` vs `D01`), plotting shared-axis population pyramids, and computing the abridged life table ($l_0 = 100,000$).*"),
    ("# STEP 2:", "## Step 2: Population Projections to 1 January 2030 & Sensitivity Analysis (20 Marks)\n*Comparing linear and compound aggregate growth against Low, Base, High, and High-Fertility sensitivity cohort-component projections for `D05`.*"),
    ("# STEP 3:", "## Step 3: Three-State Continuous-Time Markov Model (20 Marks)\n*Estimating generator matrix $Q$ across pooled exposures, holding times, next-jump probabilities, matrix exponentials $P(1)$ and $P(2)$, and expected death claims.*"),
    ("# STEP 4:", "## Step 4: District K-Means Clustering & Stability Diagnostics (20 Marks)\n*Standardizing 4 demographic/infrastructure features across all 24 districts, evaluating WCSS elbow and silhouette scores ($k=1\\dots 6$), and verifying ARI stability.*"),
]

cells = [
    new_markdown_cell(
        "# HASTS206 Applied Demographic Analysis — Complete Reproducible Pipeline\n"
        "**Deliverables:** Executed Jupyter Notebook (`HTML`), Clean Power BI Tables (`CSV`), and Diagnostic Figures (`PNG`)."
    )
]

remaining = code_text
for idx, (marker, md_header) in enumerate(markers):
    if marker in remaining:
        before, after = remaining.split(marker, 1)
        if idx == 0 and before.strip():
            cells.append(new_markdown_cell("### Setup & Portable Directory Configuration"))
            cells.append(new_code_cell(before.strip()))
        remaining = marker + after
        # Find where the next marker starts
        next_pos = len(remaining)
        for next_marker, _ in markers[idx + 1:]:
            pos = remaining.find(next_marker)
            if pos != -1 and pos < next_pos:
                next_pos = pos
        chunk = remaining[:next_pos].strip()
        remaining = remaining[next_pos:]
        cells.append(new_markdown_cell(md_header))
        cells.append(new_code_cell(chunk))

nb = new_notebook(cells=cells)
ipynb_path.write_text(nbformat.writes(nb), encoding="utf-8")
print(f"Created editable notebook: {ipynb_path.name}")

# Execute notebook and export directly to HTML
subprocess.check_call([
    sys.executable, "-m", "jupyter", "nbconvert",
    "--to", "html",
    "--execute",
    "--ExecutePreprocessor.timeout=300",
    str(ipynb_path),
    "--output", str(html_path)
])

print(f"SUCCESS! Executed HTML notebook saved to: {html_path}")
