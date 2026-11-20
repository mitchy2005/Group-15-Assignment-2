from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# 0. EXACT FOLDER SETUP (C:\Users\USER\Documents\DATA)
# ==============================================================================
try:
    DATA_DIR = Path(__file__).resolve().parent
except NameError:
    DATA_DIR = Path.cwd()

if not (DATA_DIR / "districts.csv").exists():
    if (DATA_DIR.parent / "districts.csv").exists():
        DATA_DIR = DATA_DIR.parent
    elif (Path.home() / "Documents" / "DATA" / "districts.csv").exists():
        DATA_DIR = Path.home() / "Documents" / "DATA"

print(f"Using DATA folder: {DATA_DIR}")

# ==============================================================================
# STEP 0: LOAD RAW CSVs, AUDIT DATA QUALITY & SAVE CLEANED TABLES
# ==============================================================================
districts = pd.read_csv(DATA_DIR / "districts.csv")
age_sex = pd.read_csv(DATA_DIR / "age_sex.csv")
life_table = pd.read_csv(DATA_DIR / "life_table_inputs.csv")
scenarios = pd.read_csv(DATA_DIR / "projection_scenarios.csv")
transitions = pd.read_csv(DATA_DIR / "insured_transitions.csv")

assert districts["district_id"].is_unique
assert not age_sex.duplicated(["district_id", "age_start", "sex"]).any()
assert not transitions.duplicated(["district_id", "origin", "destination"]).any()
assert set(age_sex["district_id"]) == set(districts["district_id"])
assert set(transitions["district_id"]) == set(districts["district_id"])

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

reconcile["infant_deaths_2025_births"] = reconcile["deaths_2025"] - reconcile["baseline_deaths_2025"]

districts_clean = reconcile.drop(
    columns=["arrivals_age_sex", "departures_age_sex", "births_age_sex", "pop_2026_expected"]
).copy()
districts_clean["piped_water_imputed_median"] = districts_clean["piped_water_pct"].fillna(
    districts_clean["piped_water_pct"].median()
)
districts_clean["piped_water_was_missing"] = districts_clean["piped_water_pct"].isna()

insured_origin_dedup = (
    transitions.drop_duplicates(subset=["district_id", "origin"])
    [["district_id", "origin", "exposure_py", "initial_lives"]]
    .sort_values(["district_id", "origin"])
    .reset_index(drop=True)
)

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

dq_log.to_csv(DATA_DIR / "cleaned_data_quality_log.csv", index=False)
districts_clean.to_csv(DATA_DIR / "cleaned_districts.csv", index=False)
age_sex.to_csv(DATA_DIR / "cleaned_age_sex.csv", index=False)
life_table.to_csv(DATA_DIR / "cleaned_life_table_inputs.csv", index=False)
scenarios.to_csv(DATA_DIR / "cleaned_projection_scenarios.csv", index=False)
transitions.to_csv(DATA_DIR / "cleaned_insured_transitions.csv", index=False)
insured_origin_dedup.to_csv(DATA_DIR / "cleaned_insured_origin_dedup.csv", index=False)
print("Step 0 complete: Cleaned CSVs saved to DATA folder.")

# ==============================================================================
# STEP 1: POPULATION STRUCTURE, VITAL RATES & ABRIDGED LIFE TABLE
# ==============================================================================
age_sex_clean = age_sex.copy()
life_table_clean = life_table.copy()

age_sex_clean["age_group"] = np.where(
    age_sex_clean["age_start"] == 85,
    "85+",
    age_sex_clean["age_start"].astype(str) + "-" + (age_sex_clean["age_start"] + 4).astype(str),
)

age_sex_clean["broad_age"] = pd.cut(
    age_sex_clean["age_start"],
    bins=[-1, 14, 64, 120],
    labels=["pop_0_14", "pop_15_64", "pop_65_plus"],
)

broad_age_piv = age_sex_clean.pivot_table(
    index="district_id",
    columns="broad_age",
    values="population_2025",
    aggfunc="sum",
    observed=False,
).reset_index()

sex_piv = age_sex_clean.pivot_table(
    index="district_id",
    columns="sex",
    values="population_2025",
    aggfunc="sum",
).reset_index().rename(columns={"F": "pop_female_2025", "M": "pop_male_2025"})

step1_districts = (
    districts_clean
    .merge(broad_age_piv, on="district_id")
    .merge(sex_piv, on="district_id")
)

step1_districts["mid_pop_2025"] = 0.5 * (
    step1_districts["pop_2025_stock"] + step1_districts["population_2026"]
)
step1_districts["cbr_per_1000"] = (step1_districts["births_2025"] / step1_districts["mid_pop_2025"]) * 1000
step1_districts["cdr_per_1000"] = (step1_districts["deaths_2025"] / step1_districts["mid_pop_2025"]) * 1000
step1_districts["nmr_per_1000"] = (step1_districts["net_migration_2025"] / step1_districts["mid_pop_2025"]) * 1000

step1_districts["sex_ratio_m_per_100f"] = (
    step1_districts["pop_male_2025"] / step1_districts["pop_female_2025"]
) * 100
step1_districts["youth_dependency_ratio"] = (
    step1_districts["pop_0_14"] / step1_districts["pop_15_64"]
) * 100
step1_districts["old_age_dependency_ratio"] = (
    step1_districts["pop_65_plus"] / step1_districts["pop_15_64"]
) * 100
step1_districts["pct_under_15"] = (step1_districts["pop_0_14"] / step1_districts["pop_2025_stock"]) * 100
step1_districts["pct_65_plus"] = (step1_districts["pop_65_plus"] / step1_districts["pop_2025_stock"]) * 100

step1_districts.to_csv(DATA_DIR / "step1_district_indicators.csv", index=False)

# Shared-Axis Population Pyramids (D05 vs D01)
selected_ids = ["D05", "D01"]
pyr_sub = age_sex_clean[age_sex_clean["district_id"].isin(selected_ids)].copy()
pyr_sub["pct_of_district"] = (
    pyr_sub["population_2025"]
    / pyr_sub.groupby("district_id")["population_2025"].transform("sum")
) * 100

max_pct = float(np.ceil(pyr_sub["pct_of_district"].max() + 1.0))

plt.close("all")
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 6.8), sharey=True, sharex=True)
colors = {"M": "#2b5c8f", "F": "#d95f02"}

for ax, d_id in zip([ax_left, ax_right], selected_ids):
    d_name = step1_districts.loc[step1_districts["district_id"] == d_id, "district_name"].squeeze()
    df_d = pyr_sub[pyr_sub["district_id"] == d_id].sort_values("age_start")
    males = df_d[df_d["sex"] == "M"]
    females = df_d[df_d["sex"] == "F"]

    ax.barh(males["age_group"], -males["pct_of_district"], color=colors["M"], label="Male (Left)")
    ax.barh(females["age_group"], females["pct_of_district"], color=colors["F"], label="Female (Right)")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-max_pct, max_pct)
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(x):.0f}%" for x in xticks])
    ax.set_title(f"{d_id}: {d_name} (1 Jan 2025)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Share of District Population (%)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

ax_left.set_ylabel("Age Group (Years)")
fig.savefig(DATA_DIR / "step1_population_pyramids_v2.png", dpi=300, bbox_inches="tight")
fig.suptitle("Age–Sex Population Pyramids (Shared Scale, 1 January 2025)", fontsize=13, fontweight="bold")

fig.savefig(DATA_DIR / "step1_population_pyramids.png", dpi=300, bbox_inches="tight")
plt.close(fig)


fig.suptitle("Age–Sex Population Pyramids (Shared Scale, 1 January 2025)", fontsize=13, fontweight="bold", y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.88])
plt.savefig(DATA_DIR / "step1_population_pyramids.png", dpi=300)
plt.close()

# Print Population Balancing Equation & Mid-Year Demographic Rates (D05 & D01)
bal_cols = [
    "district_id", "district_name", "pop_2025_stock", "births_2025",
    "deaths_2025", "net_migration_2025", "population_2026",
    "cbr_per_1000", "cdr_per_1000", "nmr_per_1000",
    "sex_ratio_m_per_100f", "youth_dependency_ratio", "old_age_dependency_ratio"
]
print("\n=== STEP 1A & 1B: POPULATION BALANCE, VITAL RATES & DEPENDENCY RATIOS (D05 vs D01) ===")
print(step1_districts[step1_districts["district_id"].isin(["D05", "D01"])][bal_cols].to_string(index=False))
fig.suptitle("Age–Sex Population Pyramids (Shared Scale, 1 January 2025)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(DATA_DIR / "step1_population_pyramids.png", dpi=300)
plt.close()

# Abridged Life Table (radix l0 = 100,000)
lt = life_table_clean.sort_values("age_start").reset_index(drop=True)
lt["age_group"] = np.where(
    lt["age_start"] == 85,
    "85+",
    lt["age_start"].astype(str) + "-" + (lt["age_start"] + lt["interval_years"].fillna(0).astype(int) - 1).astype(str),
)
lt["m_x"] = lt["deaths"] / lt["person_years"]
lt["q_x"] = np.where(
    lt["interval_years"].notna(),
    (lt["interval_years"] * lt["m_x"]) / (1.0 + (lt["interval_years"] - lt["ax"]) * lt["m_x"]),
    1.0,
)

l_x, d_x, L_x = [100000.0], [], []
for _, row in lt.iterrows():
    curr_l = l_x[-1]
    curr_d = curr_l * row["q_x"]
    d_x.append(curr_d)
    if pd.notna(row["interval_years"]):
        next_l = curr_l - curr_d
        l_x.append(next_l)
        L_x.append(row["interval_years"] * next_l + row["ax"] * curr_d)
    else:
        L_x.append(curr_l / row["m_x"])

lt["l_x"] = l_x
lt["d_x"] = d_x
lt["L_x"] = L_x
lt["T_x"] = lt["L_x"][::-1].cumsum()[::-1]
lt["e_x"] = lt["T_x"] / lt["l_x"]

assert np.isclose(lt["d_x"].sum(), 100000.0)
assert np.isclose(lt.loc[0, "T_x"], lt["L_x"].sum())
lt.to_csv(DATA_DIR / "step1_abridged_life_table.csv", index=False)

print(f"Step 1 complete! e_0 = {lt.loc[0, 'e_x']:.2f} years. All Step 1 files saved to {DATA_DIR}.")
import io

# Helper to save figures safely without Windows/OneDrive file locks
def save_fig_safe(fig, filepath):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    Path(filepath).write_bytes(buf.getvalue())
    plt.close(fig)
def save_csv_safe(df, filepath):
    Path(filepath).write_text(df.to_csv(index=False), encoding="utf-8")

# ==============================================================================
# STEP 2: POPULATION PROJECTIONS TO 1 JANUARY 2030 (20 MARKS)
# ==============================================================================
# Focus district for Step 2: D05 (high-growth in-migration hub)
target_dist = "D05"
d_info = step1_districts.loc[step1_districts["district_id"] == target_dist].squeeze()

# ------------------------------------------------------------------------------
# 2A. AGGREGATE 5-YEAR GROWTH MODELS (2020 -> 2025 -> 2030)
# ------------------------------------------------------------------------------
P2020 = float(d_info["population_2020"])
P2025 = float(d_info["pop_2025_stock"])

# Linear (Arithmetic) growth: P_2030 = P_2025 + 5 * ((P_2025 - P_2020) / 5)
annual_linear_gain = (P2025 - P2020) / 5.0
proj_linear_2030 = P2025 + 5.0 * annual_linear_gain

# Compound (Geometric) growth: r = (P_2025 / P_2020)^(1/5) - 1; P_2030 = P_2025 * (1 + r)^5
geom_r = (P2025 / P2020) ** (1.0 / 5.0) - 1.0
proj_compound_2030 = P2025 * ((1.0 + geom_r) ** 5.0)

print("\n=== STEP 2A: AGGREGATE 2030 PROJECTIONS FOR D05 ===")
print(f"2020 Population: {P2020:,.0f} | 2025 Population: {P2025:,.0f}")
print(f"Linear 2030 Projection:   {proj_linear_2030:,.2f} (annual gain = {annual_linear_gain:,.2f}/yr)")
print(f"Compound 2030 Projection: {proj_compound_2030:,.2f} (annual rate r = {geom_r*100:.3f}%)")

# ------------------------------------------------------------------------------
# 2B. COHORT-COMPONENT PROJECTION MODEL (LOW, BASE, HIGH + SENSITIVITY)
# ------------------------------------------------------------------------------
def project_cohort_component(dist_id, fert_mult, arr_mult, dep_mult, nb_surv_f, nb_surv_m):
    sub = age_sex_clean[age_sex_clean["district_id"] == dist_id].copy()
    total_births_2025 = sub["births_to_women_2025"].sum()
    sub["survivors"] = sub["population_2025"] * sub["survival_5yr"]

    rows = []
    for sex, nb_surv in [("F", nb_surv_f), ("M", nb_surv_m)]:
        s_df = sub[sub["sex"] == sex].sort_values("age_start").set_index("age_start")
        pop_2030 = {}

        # 1. Births over 5 years -> age 0 (0-4 cohort): 49% female, 51% male
        sex_share = 0.49 if sex == "F" else 0.51
        pop_2030.update({0: 5.0 * total_births_2025 * fert_mult * sex_share * nb_surv})

        # 2. Advance 5-year cohorts from age 5 (5-9) up to age 80 (80-84)
        for age in range(5, 85, 5):
            pop_2030[age] = s_df.loc[age - 5, "survivors"]

        # 3. Combine surviving 80-84 and retained 85+ into open 85+ interval
        pop_2030.update({85: s_df.loc[80, "survivors"] + s_df.loc[85, "survivors"]})

        # 4. Add 5-year net migration at end of period across all age-sex groups
        for age, survived_pop in pop_2030.items():
            arr_5yr = 5.0 * s_df.loc[age, "arrivals_2025"] * arr_mult
            dep_5yr = 5.0 * s_df.loc[age, "departures_2025"] * dep_mult
            rows.append({
                "district_id": dist_id,
                "sex": sex,
                "age_start": age,
                "pop_2030": survived_pop + arr_5yr - dep_5yr,
            })
    return pd.DataFrame(rows)

# Run Low, Base, High scenarios
cc_frames = []
for _, sc in scenarios.iterrows():
    df_sc = project_cohort_component(
        target_dist,
        sc["fertility_multiplier"],
        sc["arrivals_multiplier"],
        sc["departures_multiplier"],
        sc["newborn_survival_f"],
        sc["newborn_survival_m"],
    )
    df_sc["scenario"] = sc["scenario"]
    cc_frames.append(df_sc)

# One-factor sensitivity test: Base scenario with High fertility multiplier
# One-factor sensitivity test: Base scenario with High fertility multiplier
base_row = scenarios.loc[scenarios["scenario"] == "Base"].squeeze()
high_row = scenarios.loc[scenarios["scenario"] == "High"].squeeze()
df_sens = project_cohort_component(
    target_dist,
    high_row["fertility_multiplier"],
    base_row["arrivals_multiplier"],
    base_row["departures_multiplier"],
    base_row["newborn_survival_f"],
    base_row["newborn_survival_m"],
)
df_sens["scenario"] = "Sensitivity_Base_HighFert"
cc_frames.append(df_sens)

# Save all new outputs into a clean subfolder DATA\results so Windows never locks them
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

step2_cohorts = pd.concat(cc_frames, ignore_index=True)
save_csv_safe(step2_cohorts, RESULTS_DIR / "step2_cohort_projections_2030.csv")

# Summarize 2030 totals and broad age shares (<15 and 65+) across models
summary_rows = [
    {"model": "2025 Actual Baseline", "total_pop": P2025,
     "pct_under_15": float(d_info["pct_under_15"]), "pct_65_plus": float(d_info["pct_65_plus"])},
    {"model": "Linear Growth (2030)", "total_pop": proj_linear_2030, "pct_under_15": np.nan, "pct_65_plus": np.nan},
    {"model": "Compound Growth (2030)", "total_pop": proj_compound_2030, "pct_under_15": np.nan, "pct_65_plus": np.nan},
]

for scen_name, grp in step2_cohorts.groupby("scenario", sort=False):
    tot = grp["pop_2030"].sum()
    u15 = grp[grp["age_start"] < 15]["pop_2030"].sum() / tot * 100
    o65 = grp[grp["age_start"] >= 65]["pop_2030"].sum() / tot * 100
    summary_rows.append({"model": f"Cohort-Component: {scen_name}", "total_pop": tot, "pct_under_15": u15, "pct_65_plus": o65})

step2_summary = pd.DataFrame(summary_rows)
save_csv_safe(step2_summary, RESULTS_DIR / "step2_projection_summary.csv")

print("\n=== STEP 2B & 2C: 2030 PROJECTION COMPARISON & SENSITIVITY SUMMARY (D05) ===")
print(step2_summary.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

# Accounting identity verification for Base scenario
base_cc = step2_cohorts[step2_cohorts["scenario"] == "Base"]
surv_init = (age_sex_clean[age_sex_clean["district_id"] == target_dist]["population_2025"] *
             age_sex_clean[age_sex_clean["district_id"] == target_dist]["survival_5yr"]).sum()
tot_b25 = age_sex_clean[age_sex_clean["district_id"] == target_dist]["births_to_women_2025"].sum()
surv_births = 5.0 * tot_b25 * base_row["fertility_multiplier"] * (
    0.49 * base_row["newborn_survival_f"] + 0.51 * base_row["newborn_survival_m"]
)
net_mig_5y = 5.0 * (
    age_sex_clean[age_sex_clean["district_id"] == target_dist]["arrivals_2025"].sum() * base_row["arrivals_multiplier"]
    - age_sex_clean[age_sex_clean["district_id"] == target_dist]["departures_2025"].sum() * base_row["departures_multiplier"]
)
assert np.isclose(base_cc["pop_2030"].sum(), surv_init + surv_births + net_mig_5y), "Accounting total mismatch!"
print(f"\nAccounting check PASSED: Surviving cohorts ({surv_init:,.1f}) + Surviving births ({surv_births:,.1f}) + 5-yr net migration ({net_mig_5y:,.1f}) = {base_cc['pop_2030'].sum():,.1f}")

# Quantify exact effect on ages 0-4 (Base vs Sensitivity_Base_HighFert)
age0_4_base = step2_cohorts[(step2_cohorts["scenario"] == "Base") & (step2_cohorts["age_start"] == 0)]["pop_2030"].sum()
age0_4_sens = step2_cohorts[(step2_cohorts["scenario"] == "Sensitivity_Base_HighFert") & (step2_cohorts["age_start"] == 0)]["pop_2030"].sum()
diff_0_4 = age0_4_sens - age0_4_base
pct_0_4 = (diff_0_4 / age0_4_base) * 100
print(f"Ages 0-4 Sensitivity Impact: Base = {age0_4_base:,.1f} vs High-Fertility = {age0_4_sens:,.1f} (+{diff_0_4:,.1f} children, +{pct_0_4:.2f}%)")

# Plot 2030 Projection Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 5.5))
plot_df = step2_summary.iloc[1:].copy()
bars = ax.bar(plot_df["model"], plot_df["total_pop"], color=["#7f7f7f", "#595959", "#6baed6", "#2171b5", "#08306b", "#d95f02"])
ax.axhline(P2025, color="black", linestyle="--", linewidth=1, label=f"2025 Baseline ({P2025:,.0f})")
ax.set_ylabel("Projected Population (1 Jan 2030)")
ax.set_title(f"{target_dist} Population Projections to 1 January 2030 Across Models", fontweight="bold")
ax.set_ylim(P2025 * 0.95, plot_df["total_pop"].max() * 1.04)
plt.xticks(rotation=25, ha="right")
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 350, f"{h:,.0f}", ha="center", va="bottom", fontsize=9)
ax.legend()
save_fig_safe(fig, RESULTS_DIR / "step2_projections_2030.png")
print(f"Step 2 complete! Projection CSVs and chart saved to {RESULTS_DIR}.")
from scipy.linalg import expm
from matplotlib.patches import FancyArrowPatch, Circle
from scipy.linalg import expm
from matplotlib.patches import FancyArrowPatch, Circle

# ==============================================================================
# STEP 3: THREE-STATE CONTINUOUS-TIME MARKOV MODEL (20 MARKS)
# ==============================================================================
# ------------------------------------------------------------------------------
# 3A. POOL COUNTS, CONSTRUCT GENERATOR Q & VERIFY SIGNS / ROW SUMS
# ------------------------------------------------------------------------------
pooled_origin = insured_origin_dedup.groupby("origin", as_index=False).agg(
    total_exposure=("exposure_py", "sum"),
    total_initial_lives=("initial_lives", "sum"),
)
pooled_jumps = transitions.groupby(["origin", "destination"], as_index=False)["transition_count"].sum()
rates_df = pooled_jumps.merge(pooled_origin, on="origin")
rates_df["q_ij"] = rates_df["transition_count"] / rates_df["total_exposure"]

q_lookup = {(r["origin"], r["destination"]): float(r["q_ij"]) for _, r in rates_df.iterrows()}
q_HI = q_lookup[("H", "I")]
q_HD = q_lookup[("H", "D")]
q_IH = q_lookup[("I", "H")]
q_ID = q_lookup[("I", "D")]

# Generator matrix Q in exact state order: H, I, D
q_HH = -(q_HI + q_HD)
q_II = -(q_IH + q_ID)
Q = np.array([
    [q_HH, q_HI, q_HD],
    [q_IH, q_II, q_ID],
    [0.0,  0.0,  0.0],
])

# Explicit verification of Q row sums and off-diagonal signs
off_diag_mask = ~np.eye(3, dtype=bool)
assert np.allclose(Q.sum(axis=1), 0.0), "ERROR: Generator Q row sums must equal 0!"
assert np.all(Q[off_diag_mask] >= 0.0), "ERROR: Off-diagonal transition intensities must be >= 0!"
assert np.all(np.diag(Q) <= 0.0), "ERROR: Diagonal entries of Q must be <= 0!"

Q_df = pd.DataFrame(Q, index=["H", "I", "D"], columns=["H", "I", "D"])
save_csv_safe(Q_df.reset_index().rename(columns={"index": "from_state"}), RESULTS_DIR / "step3_generator_Q.csv")

# Draw Healthy-Ill-Dead State Diagram with recovery and absorbing death
fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.5)
ax.axis("off")

pos = {"H": (2.2, 4.5), "I": (7.8, 4.5), "D": (5.0, 1.3)}
node_colors = {"H": "#d9edf7", "I": "#fcf8e3", "D": "#f2dede"}
edge_colors = {"H": "#2b5c8f", "I": "#d95f02", "D": "#595959"}
labels_full = {"H": "Healthy\n(H)", "I": "Ill\n(I)", "D": "Dead (D)\n[Absorbing]"}

for state, (x, y) in pos.items():
    circ = Circle((x, y), radius=0.85, facecolor=node_colors[state], edgecolor=edge_colors[state], linewidth=2.2, zorder=3)
    ax.add_patch(circ)
    ax.text(x, y, labels_full[state], ha="center", va="center", fontsize=10.5, fontweight="bold", zorder=4)

arrows = [
    ("H", "I", 0.22, f"q_HI = {q_HI:.4f}/yr", (5.0, 5.55)),
    ("I", "H", 0.22, f"q_IH = {q_IH:.4f}/yr (Recovery)", (5.0, 3.45)),
    ("H", "D", 0.15, f"q_HD = {q_HD:.4f}/yr", (2.5, 2.65)),
    ("I", "D", -0.15, f"q_ID = {q_ID:.4f}/yr", (7.5, 2.65)),
]
for u, v, rad, lbl, (lx, ly) in arrows:
    arrow = FancyArrowPatch(
        pos[u], pos[v],
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>,head_length=8,head_width=5",
        shrinkA=48, shrinkB=48,
        linewidth=1.8, color="#333333", zorder=2
    )
    ax.add_patch(arrow)
    ax.text(lx, ly, lbl, ha="center", va="center", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.95))

ax.set_title("Three-State Continuous-Time Markov Model (Healthy – Ill – Dead)", fontsize=12.5, fontweight="bold", pad=10)
save_fig_safe(fig, RESULTS_DIR / "step3_markov_state_diagram.png")

# ------------------------------------------------------------------------------
# 3B. MEAN HOLDING TIMES & NEXT-JUMP PROBABILITIES FROM ILL (I)
# ------------------------------------------------------------------------------
mean_hold_H = 1.0 / (-q_HH)
mean_hold_I = 1.0 / (-q_II)

exit_rate_I = q_IH + q_ID
prob_next_jump_I_to_H = q_IH / exit_rate_I
prob_next_jump_I_to_D = q_ID / exit_rate_I
assert np.isclose(prob_next_jump_I_to_H + prob_next_jump_I_to_D, 1.0)

# ------------------------------------------------------------------------------
# 3C. MATRIX EXPONENTIAL P(1), P(2), VERIFICATION & EXPECTED OCCUPANCY
# ------------------------------------------------------------------------------
P1 = expm(Q)
P2 = expm(2.0 * Q)

P1_df = pd.DataFrame(P1, index=["H", "I", "D"], columns=["H", "I", "D"])
P2_df = pd.DataFrame(P2, index=["H", "I", "D"], columns=["H", "I", "D"])

# Mandatory checks: non-negativity, row sums = 1, D absorbing, P(2) == P(1)^2
assert np.all(P1 >= -1e-12) and np.all(P2 >= -1e-12), "ERROR: Negative probabilities!"
assert np.allclose(P1.sum(axis=1), 1.0) and np.allclose(P2.sum(axis=1), 1.0), "ERROR: Rows do not sum to 1!"
assert np.allclose(P1_df.loc["D"].values, [0.0, 0.0, 1.0]) and np.allclose(P2_df.loc["D"].values, [0.0, 0.0, 1.0]), "ERROR: State D must be absorbing!"
assert np.allclose(P2, P1 @ P1), "ERROR: P(2) does not match P(1)^2 within numerical tolerance!"

save_csv_safe(P1_df.reset_index().rename(columns={"index": "from_state"}), RESULTS_DIR / "step3_transition_P1.csv")
save_csv_safe(P2_df.reset_index().rename(columns={"index": "from_state"}), RESULTS_DIR / "step3_transition_P2.csv")

prob_1yr_I_to_H = float(P1_df.loc["I", "H"])

# Pooled initial count row vector n0 = [N_H(0), N_I(0), N_D(0)]
init_map = dict(zip(pooled_origin["origin"], pooled_origin["total_initial_lives"]))
n0 = np.array([float(init_map["H"]), float(init_map["I"]), 0.0])
n1 = n0 @ P1
n2 = n0 @ P2

# ------------------------------------------------------------------------------
# 3D. SEPARATE YEAR 1 & YEAR 2 BENEFIT PAYMENTS (USD 1,000 PER DEATH)
# ------------------------------------------------------------------------------
n0_H, n0_I, n0_D = map(float, n0)
n1_H, n1_I, n1_D = map(float, n1)
n2_H, n2_I, n2_D = map(float, n2)

new_deaths_y1 = n1_D - n0_D
new_deaths_y2 = n2_D - n1_D
claims_y1_usd = new_deaths_y1 * 1000.0
claims_y2_usd = new_deaths_y2 * 1000.0

step3_occupancy = pd.DataFrame({
    "year": list(range(3)),
    "Healthy_H": [n0_H, n1_H, n2_H],
    "Ill_I": [n0_I, n1_I, n2_I],
    "Dead_D": [n0_D, n1_D, n2_D],
    "new_deaths": [0.0, new_deaths_y1, new_deaths_y2],
    "annual_claims_usd": [0.0, claims_y1_usd, claims_y2_usd],
    "cumulative_claims_usd": [0.0, claims_y1_usd, claims_y1_usd + claims_y2_usd],
})
save_csv_safe(step3_occupancy, RESULTS_DIR / "step3_markov_occupancy_claims.csv")

# Print full Step 3 summary & conceptual interpretations
print("\n=== STEP 3A: GENERATOR MATRIX Q (H, I, D) ===")
print(Q_df.to_string(float_format=lambda x: f"{x:.6f}"))
print("Verification PASSED: Off-diagonal q_ij >= 0, diagonal q_ii <= 0, row sums = 0.")

print("\n=== STEP 3B: HOLDING TIMES & NEXT-JUMP PROBABILITIES FROM ILL (I) ===")
print(f"Mean holding time in Healthy (H): {mean_hold_H:.4f} years")
print(f"Mean holding time in Ill (I):     {mean_hold_I:.4f} years")
print(f"Next-jump destination from Ill -> Healthy (H): {prob_next_jump_I_to_H:.4f} ({prob_next_jump_I_to_H*100:.2f}%)")
print(f"Next-jump destination from Ill -> Dead (D):    {prob_next_jump_I_to_D:.4f} ({prob_next_jump_I_to_D*100:.2f}%)")
print(f"Comparison vs 1-yr fixed-date P_IH(1) = {prob_1yr_I_to_H:.4f} ({prob_1yr_I_to_H*100:.2f}%):")
print("  -> Eventual next-jump recovery has no fixed time horizon and considers only the single first exit")
print("     from I, whereas P_IH(1) requires being in H at exact time t = 1 year, accounting for individuals")
print("     who remain in I at year-end or who recover and relapse/die within the year.")

print("\n=== STEP 3C & 3D: P(1), P(2), EXPECTED OCCUPANCY & BENEFIT CLAIMS ===")
print("1-Year Transition Matrix P(1):")
print(P1_df.to_string(float_format=lambda x: f"{x:.6f}"))
print("\n2-Year Transition Matrix P(2):")
print(P2_df.to_string(float_format=lambda x: f"{x:.6f}"))
print("\nExpected Insured Occupancy & Separate Annual Benefit Payments:")
print(step3_occupancy.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
print(f"\nYear 1 Expected Claims: USD {claims_y1_usd:,.2f} ({new_deaths_y1:,.2f} deaths)")
print(f"Year 2 Expected Claims: USD {claims_y2_usd:,.2f} ({new_deaths_y2:,.2f} deaths)")
print("Limitation of pooling constant intensities: Assumes time-homogeneous rates and ignores")
print("  age, sex, illness duration, and district heterogeneity (e.g., lower piped-water access in rural districts).")
print(f"Step 3 complete! State diagram, matrices, and claims CSV saved to {RESULTS_DIR}.")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score

# ==============================================================================
# STEP 4: DISTRICT CLUSTERING (K-MEANS, 20 MARKS)
# ==============================================================================
# ------------------------------------------------------------------------------
# 4A. FEATURE PREPARATION, IMPUTATION, OUTLIER CHECK & STANDARDIZATION
# ------------------------------------------------------------------------------
cluster_features = ["pct_under_15", "pct_65_plus", "nmr_per_1000", "piped_water_imputed_median"]
X_raw = step1_districts[cluster_features].copy()

missing_ids = step1_districts.loc[step1_districts["piped_water_was_missing"], "district_id"].tolist()
median_water = float(step1_districts["piped_water_pct"].median())

print("\n=== STEP 4A: FEATURE PREPARATION, IMPUTATION & OUTLIER CHECK ===")
print(f"Imputed missing piped_water_pct for {missing_ids} using median = {median_water:.2f}%.")

# Outlier audit via z-scores (flag |z| > 2.5)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
z_df = pd.DataFrame(X_scaled, columns=cluster_features, index=step1_districts["district_id"])
outliers = z_df[(z_df.abs() > 2.5).any(axis=1)]
print(f"Extreme outliers (|z| > 2.5): {len(outliers)} districts found.")

# ------------------------------------------------------------------------------
# 4B. K-MEANS DIAGNOSTICS (k = 1..6) & SILHOUETTE PLOTS
# ------------------------------------------------------------------------------
diag_rows = []
models_by_k = {}

for k in range(1, 7):
    km = KMeans(n_clusters=k, n_init=50, random_state=42).fit(X_scaled)
    models_by_k[k] = km
    sil_val = float(silhouette_score(X_scaled, km.labels_)) if k >= 2 else np.nan
    diag_rows.append({"k": k, "wcss": float(km.inertia_), "silhouette_score": sil_val})

step4_diag = pd.DataFrame(diag_rows)
step4_diag["wcss_pct_reduction"] = -step4_diag["wcss"].pct_change() * 100.0
save_csv_safe(step4_diag, RESULTS_DIR / "step4_clustering_diagnostics.csv")

print("\n=== STEP 4B: K-MEANS DIAGNOSTICS TABLE (k = 1..6) ===")
print(step4_diag.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

# Plot 1: Elbow Curve (WCSS) & Average Silhouette Score (k = 2..6)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
ax1.plot(step4_diag["k"], step4_diag["wcss"], marker="o", linewidth=2, color="#2171b5")
ax1.set_title("Elbow Curve: WCSS vs Number of Clusters (k)", fontweight="bold")
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Within-Cluster Sum of Squares (WCSS)")
ax1.grid(True, linestyle="--", alpha=0.4)

ax2.plot(step4_diag["k"].iloc[1:], step4_diag["silhouette_score"].iloc[1:], marker="s", linewidth=2, color="#d95f02")
ax2.set_title("Average Silhouette Score (k = 2..6)", fontweight="bold")
ax2.set_xlabel("Number of Clusters (k)")
ax2.set_ylabel("Mean Silhouette Coefficient")
ax2.grid(True, linestyle="--", alpha=0.4)
save_fig_safe(fig, RESULTS_DIR / "step4_elbow_and_silhouette.png")

# Plot 2: Individual Silhouette Plots for Two Candidate k Values (k = 3 and k = 4)
fig, (ax_k3, ax_k4) = plt.subplots(1, 2, figsize=(12.5, 5.2))
for ax, k_cand in [(ax_k3, 3), (ax_k4, 4)]:
    labels_c = models_by_k[k_cand].labels_
    sample_sil = silhouette_samples(X_scaled, labels_c)
    avg_sil = float(silhouette_score(X_scaled, labels_c))
    y_lower = 10

    for c_id in range(k_cand):
        c_vals = np.sort(sample_sil[labels_c == c_id])
        y_upper = y_lower + len(c_vals)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, c_vals, alpha=0.85)
        ax.text(-0.06, y_lower + 0.5 * len(c_vals), f"C{c_id}", fontweight="bold")
        y_lower = y_upper + 10

    ax.axvline(avg_sil, color="red", linestyle="--", linewidth=1.3, label=f"Mean = {avg_sil:.3f}")
    ax.set_title(f"Individual Silhouette Plot (k = {k_cand})", fontweight="bold")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Districts Grouped by Cluster")
    ax.legend(loc="lower right")

save_fig_safe(fig, RESULTS_DIR / "step4_individual_silhouettes_k3_k4.png")

# ------------------------------------------------------------------------------
# 4C. FINAL CLUSTER SELECTION, STABILITY CHECKS (ARI) & UNSTANDARDIZED CENTROIDS
# ------------------------------------------------------------------------------
k_opt = 3
final_labels = models_by_k[k_opt].labels_

# Stability Check 1: Alternative random seed (random_state = 99)
seed99_labels = KMeans(n_clusters=k_opt, n_init=50, random_state=99).fit(X_scaled).labels_
ari_seed = float(adjusted_rand_score(final_labels, seed99_labels))

# Stability Check 2: Alternative imputation (mean instead of median for piped_water_pct)
X_mean_imp = step1_districts[["pct_under_15", "pct_65_plus", "nmr_per_1000"]].copy()
X_mean_imp["piped_water_mean"] = step1_districts["piped_water_pct"].fillna(
    step1_districts["piped_water_pct"].mean()
)
mean_imp_labels = KMeans(n_clusters=k_opt, n_init=50, random_state=42).fit(
    StandardScaler().fit_transform(X_mean_imp)
).labels_
ari_imputation = float(adjusted_rand_score(final_labels, mean_imp_labels))

print(f"\n=== STEP 4C: CLUSTER STABILITY & UNSTANDARDIZED CENTROIDS (k = {k_opt}) ===")
print(f"Stability ARI (New Random Seed = 99):      {ari_seed:.4f}")
print(f"Stability ARI (Mean vs Median Imputation): {ari_imputation:.4f}")

# Save master Power BI table with cluster labels
step1_districts["cluster_k3"] = final_labels
save_csv_safe(step1_districts, RESULTS_DIR / "powerbi_districts_master.csv")

# Unstandardized cluster centroids (original units)
centroids_orig = step1_districts.groupby("cluster_k3", as_index=False)[cluster_features].mean()
counts_df = step1_districts.groupby("cluster_k3", as_index=False).size().rename(columns={"size": "district_count"})
centroids_orig = centroids_orig.merge(counts_df, on="cluster_k3")
save_csv_safe(centroids_orig, RESULTS_DIR / "step4_cluster_centroids_final.csv")
print("\nUnstandardized Cluster Centroids (Original Units):")
print(centroids_orig.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
print("\nDistrict Assignments by Cluster:")
for c_id, grp in step1_districts.groupby("cluster_k3"):
    print(f"  Cluster {c_id} ({len(grp)} districts): {', '.join(grp['district_id'].tolist())}")

print(f"\nALL 4 STEPS COMPLETE! All CSVs and PNGs saved to {RESULTS_DIR}.")
print("\n=== STEP 4 CONCEPTUAL CHECKS (FOR REPORT & NOTEBOOK) ===")
print("1. Why district_id is excluded: Arbitrary nominal ID with no demographic meaning; including it distorts Euclidean distances.")
print("2. WCSS & Silhouette meaning: WCSS measures cluster compactness (sum of squared distances to centroids). Near-zero silhouette")
print("   means a district lies on the boundary between clusters; negative silhouette indicates probable cluster misassignment.")
print("3. Use of clustering: Enables targeted regional resource allocation (e.g., water infrastructure vs. urban clinic capacity).")
print("4. Limitation of clustering: Hard boundaries on static 2025 averages hide within-district heterogeneity and borderline overlap.")
