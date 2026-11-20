from pathlib import Path
import numpy as np
import pandas as pd

# ==============================================================================
# 1. AUTOMATICALLY LOCATE THE 'DATA' FOLDER
# ==============================================================================
try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path.cwd()

# Check if we are already inside DATA, inside a subfolder of DATA, or in Documents
if (script_dir / "districts.csv").exists():
    DATA_DIR = script_dir
elif (script_dir / "DATA" / "districts.csv").exists():
    DATA_DIR = script_dir / "DATA"
elif (Path.home() / "Documents" / "DATA" / "districts.csv").exists():
    DATA_DIR = Path.home() / "Documents" / "DATA"
else:
    raise FileNotFoundError("Could not find 'districts.csv' inside Documents\\DATA.")

print(f"Saving cleaned outputs directly into: {DATA_DIR}")

# ==============================================================================
# 2. LOAD RAW DATASETS FROM DATA FOLDER
# ==============================================================================
districts = pd.read_csv(DATA_DIR / "districts.csv")
age_sex = pd.read_csv(DATA_DIR / "age_sex.csv")
life_table = pd.read_csv(DATA_DIR / "life_table_inputs.csv")
scenarios = pd.read_csv(DATA_DIR / "projection_scenarios.csv")
transitions = pd.read_csv(DATA_DIR / "insured_transitions.csv")

# ==============================================================================
# 3. RUN DATA CLEANING, ASSERTIONS & RECONCILIATION
# ==============================================================================
# Primary key & join assertions
assert districts["district_id"].is_unique
assert not age_sex.duplicated(["district_id", "age_start", "sex"]).any()
assert not transitions.duplicated(["district_id", "origin", "destination"]).any()
assert set(age_sex["district_id"]) == set(districts["district_id"])
assert set(transitions["district_id"]) == set(districts["district_id"])

# Aggregate 2025 stocks & flows from age_sex.csv
age_sex_agg = age_sex.groupby("district_id", as_index=False).agg(
    pop_2025_stock=("population_2025", "sum"),
    baseline_deaths_2025=("baseline_deaths_2025", "sum"),
    arrivals_age_sex=("arrivals_2025", "sum"),
    departures_age_sex=("departures_2025", "sum"),
    births_age_sex=("births_to_women_2025", "sum"),
)

reconcile = districts.merge(age_sex_agg, on="district_id", how="inner")
reconcile["net_migration_2025"] = reconcile["arrivals_2025"] - reconcile["departures_2025"]
reconcile["pop_2026_expected"] = (
    reconcile["pop_2025_stock"]
    + reconcile["births_2025"]
    - reconcile["deaths_2025"]
    + reconcile["net_migration_2025"]
)
assert (reconcile["population_2026"] == reconcile["pop_2026_expected"]).all(), "Balancing equation failed!"

# Isolate infant deaths among 2025 births (explains difference between deaths_2025 and baseline_deaths_2025)
reconcile["infant_deaths_2025_births"] = reconcile["deaths_2025"] - reconcile["baseline_deaths_2025"]

# Clean districts table + median imputation for missing piped_water_pct (D04 & D17)
districts_clean = reconcile.drop(
    columns=["arrivals_age_sex", "departures_age_sex", "births_age_sex", "pop_2026_expected"]
).copy()
districts_clean["piped_water_imputed_median"] = districts_clean["piped_water_pct"].fillna(
    districts_clean["piped_water_pct"].median()
)
districts_clean["piped_water_was_missing"] = districts_clean["piped_water_pct"].isna()

# Deduplicate insured exposures per (district_id, origin) for Step 3
insured_origin_dedup = (
    transitions.drop_duplicates(subset=["district_id", "origin"])
    [["district_id", "origin", "exposure_py", "initial_lives"]]
    .sort_values(["district_id", "origin"])
    .reset_index(drop=True)
)

# Data-quality log summary
dq_log = pd.DataFrame([
    {"table": name, "rows": df.shape, "cols": df.shape, "nulls": int(df.isna().sum().sum())}
    for name, df in [
        ("districts.csv", districts),
        ("age_sex.csv", age_sex),
        ("life_table_inputs.csv", life_table),
        ("projection_scenarios.csv", scenarios),
        ("insured_transitions.csv", transitions),
    ]
])

# ==============================================================================
# 4. AUTOMATICALLY SAVE CLEANED OUTPUT FILES DIRECTLY INTO 'DATA'
# ==============================================================================
dq_log.to_csv(DATA_DIR / "cleaned_data_quality_log.csv", index=False)
districts_clean.to_csv(DATA_DIR / "cleaned_districts.csv", index=False)
age_sex.to_csv(DATA_DIR / "cleaned_age_sex.csv", index=False)
life_table.to_csv(DATA_DIR / "cleaned_life_table_inputs.csv", index=False)
scenarios.to_csv(DATA_DIR / "cleaned_projection_scenarios.csv", index=False)
transitions.to_csv(DATA_DIR / "cleaned_insured_transitions.csv", index=False)
insured_origin_dedup.to_csv(DATA_DIR / "cleaned_insured_origin_dedup.csv", index=False)

print("Done! The following cleaned files have been added directly to your DATA folder:")
for f in sorted(DATA_DIR.glob("cleaned_*.csv")):
    print(f" - {f.name}")
