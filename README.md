# HASTS206 Applied Demographic Analysis — Group 15

**Course:** HASTS206 — Applied Demographic Analysis  
**Submission Date:** 22 November 2026  
**Group:** Group 15 (9 Active Members)  

---

## 1. Project Overview
This repository contains the complete, reproducible demographic data pipeline, executed Jupyter Notebook, Power BI dashboard, and technical report evaluating 24 administrative districts across four core modules:
1. **Data Quality Audit & Vital Rates (Step 0 & Step 1):** Schema validation, 2025 demographic balancing equation reconciliation, missing piped-water median imputation (`D04`, `D17`), contrasting district pyramids (`D05` vs. `D01`), and an abridged life table (\(e_0 = 73.11\text{ years}\)).
2. **Population Projections to 1 January 2030 (Step 2):** Arithmetic and geometric aggregate models compared against Low, Base, High, and High-Fertility sensitivity cohort-component projections for District `D05`.
3. **Three-State Continuous-Time Markov Model (Step 3):** Generator matrix \(Q\) across pooled exposures (`36,000` insured lives), holding times, matrix exponentials \(P(1)\) and \(P(2)\), and Year 1 (`USD 616,500.63`) and Year 2 (`USD 634,456.54`) death benefit claims.
4. **District K-Means Clustering (Step 4):** Standardized segmentation (\(k = 3\), \(\text{ARI} = 1.0000\)) across `% under 15`, `% 65+`, `2025 NMR`, and `piped_water_pct`.

---

## 2. Repository Structure

```text
├── data/
│   ├── raw/                     # Original 5 input CSVs (districts, age_sex, life_table, scenarios, transitions)
│   └── cleaned/                 # 7 reconciled, deduplicated & imputed CSV datasets + quality audit log
├── src/
│   ├── data_cleaning.py         # Standalone data validation and cleaning script
│   ├── hasts206_demographic_pipeline.py  # Master end-to-end Python pipeline (Steps 0–4)
│   ├── make_notebook.py         # Automated converter generating .ipynb and executed .html
│   └── export_part_3_pdf.py     # Standalone Part 3 Markov summary CSV and PDF exporter
├── notebooks/
│   ├── HASTS206_Applied_Demographic_Analysis.ipynb  # Editable Jupyter Notebook
│   └── HASTS206_Applied_Demographic_Analysis.html   # Executed HTML Notebook deliverable
├── results/
│   ├── step1_*.csv / *.png      # Life table, district indicators, and shared-axis pyramids
│   ├── step2_*.csv / *.png      # 2030 cohort projections, summary comparison, and chart
│   ├── step3_*.csv / *.png / .pdf # Generator Q, P(1), P(2), occupancy/claims, state diagram, Part 3 PDF
│   └── step4_*.csv / *.png      # Clustering diagnostics, unstandardized centroids, elbow & silhouette plots
├── dashboard/
│   ├── powerbi_districts_master.csv              # Master 24-district dataset with k=3 cluster labels
│   └── POWER BI FILE (assignment 2 Group 15).pbix # 4-page interactive Power BI workbook
├── report/
│   └── Group15_Final_Report.pdf # 6-page technical report + GitHub PR contribution appendix
└── README.md                    # Project documentation and reproduction guide
```

---

## 3. How to Reproduce the Analysis

### Prerequisites
* **Python 3.10+** with `pandas`, `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `nbformat`, `nbconvert`, and `ipykernel`
* **Power BI Desktop** (to open and refresh the `.pbix` workbook)

### Step-by-Step Execution
1. **Clone the repository and navigate to the project folder:**
   ```bash
   git clone <INSERT_YOUR_GITHUB_REPO_URL>
   cd <INSERT_REPO_FOLDER_NAME>
   ```
2. **Install required Python libraries:**
   ```bash
   pip install pandas numpy scipy matplotlib scikit-learn nbformat nbconvert ipykernel
   ```
3. **Run the complete analytical pipeline (Steps 0–4):**
   ```bash
   python src/hasts206_demographic_pipeline.py
   ```
   *All cleaned tables, summary CSVs, and diagnostic PNG figures will generate inside `results/`.*
4. **Regenerate the executed Jupyter Notebook (`.ipynb` and `.html`):**
   ```bash
   python src/make_notebook.py
   ```
5. **View or refresh the Power BI Dashboard:**
   * Open `dashboard/POWER BI FILE (assignment 2 Group 15).pbix` in Power BI Desktop and click **Refresh** on the Home ribbon to pull updated CSV outputs.

---

## 4. Group 15 Team Contributions

| Member | Full Name | GitHub Branch | Owned Deliverables & Files | PR Link / Status |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Amelia Samuel** | `m1-raw-datasets` | Raw input datasets (`data/raw/*.csv` — 5 files) | `[Insert PR #1 Link]` |
| **2** | **Enerst Dzimati** | `m2-data-cleaning` | `src/data_cleaning.py` & `data/cleaned/*.csv` (8 files) | `[Insert PR #2 Link]` |
| **3** | **Martin Simbarasi** | `m3-step1-demographics` | Step 1 life table, vital indicators & pyramid (`results/step1_*`) | `[Insert PR #3 Link]` |
| **4** | **Wisdom Masimba** | `m4-step2-projections` | Step 2 2030 projections & sensitivity outputs (`results/step2_*`) | `[Insert PR #4 Link]` |
| **5** | **Keith Muganiri** | `m5-step3-markov` | `src/export_part_3_pdf.py` & Step 3 Markov/claims outputs (`results/step3_*`) | `[Insert PR #5 Link]` |
| **6** | **Michell T. Kufa** | `m6-step4-clustering` | Step 4 k-means diagnostics, centroids & silhouette plots (`results/step4_*`) | `[Insert PR #6 Link]` |
| **7** | **Roland Manyonda** | `m7-pipeline-notebook` | `hasts206_demographic_pipeline.py`, `make_notebook.py`, `.ipynb` & `.html` | `[Insert PR #7 Link]` |
| **8** | **Darlington K. Makora** | `m8-powerbi-dashboard` | `powerbi_districts_master.csv` & `POWER BI FILE (assignment 2 Group 15).pbix` | `[Insert PR #8 Link]` |
| **9** | **Tanatswa Makuvire** | `m9-readme-final-report` | `README.md` & `report/Group15_Final_Report.pdf` | `[Insert PR #9 Link]` |
