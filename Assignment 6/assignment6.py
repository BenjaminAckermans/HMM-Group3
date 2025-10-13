#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 12:38:16 2025

@author: ariannaperini
"""

# Full script for Parts A, B, C, D (tested)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math, warnings
from itertools import product
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

# -----------------------------
# EDIT THIS PATH before running
DATA_PATH = r"Data for Assignment 6.xlsx"
# -----------------------------

# Load workbook
xls = pd.ExcelFile(DATA_PATH)
patients = pd.read_excel(xls, sheet_name="patients")
nursing_hours = pd.read_excel(xls, sheet_name="nursing hours")
nurse_costs = pd.read_excel(xls, sheet_name="nurse costs")

# Normalize column names
patients.columns = [c.strip() for c in patients.columns]
nursing_hours.columns = [c.strip() for c in nursing_hours.columns]
nurse_costs.columns = [c.strip() for c in nurse_costs.columns]

# Use the correct ward/specialism column (your file uses 'specilism')
patients['ward'] = patients['specilism'].astype(str).str.strip().str.upper()

# Parse date columns safely
for c in ["OR date", "ready for ward", "birthdate"]:
    if c in patients.columns:
        patients[c] = pd.to_datetime(patients[c], errors='coerce', dayfirst=True)

# -----------------------------
# PART A — Daycare
# -----------------------------
print("\n=== PART A: Daycare ===")
# Daycare detection: ward == 'DAYCARE' or LOS == 0
daycare = patients[(patients['ward']=='DAYCARE') | (patients.get("LOS")==0)].copy()
daycare = daycare.dropna(subset=["ready for ward"])
print("Daycare records:", len(daycare))

# Assumption: empty at 08:00 => only count admissions with ready time >= 08:00
START_H, END_H = 8, 21  # 8:00 - 21:00 coverage (slots 8..20)
hours = list(range(START_H, END_H))
slots = [f"{h:02d}:00-{h+1:02d}:00" for h in hours]

records = []
for _, r in daycare.iterrows():
    adm = pd.to_datetime(r["ready for ward"])
    if adm.hour < START_H:
        continue   # empty at 08:00 assumption
    stay_end = adm + pd.Timedelta(hours=4)
    adm_date = adm.date()
    for h in hours:
        slot_start = datetime.combine(adm_date, datetime.min.time()).replace(hour=h)
        slot_end = slot_start + pd.Timedelta(hours=1)
        if (adm < slot_end) and (stay_end > slot_start):
            records.append({"date": adm_date, "hour": h})

if not records:
    avg_hourly_df = pd.DataFrame({"slot": slots, "avg_required_nurse_hours": [0]*len(slots)})
    print("No daycare presence after empty-at-08:00 assumption — avg hourly zeros.")
else:
    presence = pd.DataFrame(records)
    all_dates = pd.date_range(presence["date"].min(), presence["date"].max(), freq="D").date
    all_combos = pd.MultiIndex.from_product([all_dates, hours], names=["date","hour"]).to_frame(index=False)
    hourly_counts = presence.groupby(["date","hour"]).size().reset_index(name="count")
    hourly_full = all_combos.merge(hourly_counts, on=["date","hour"], how="left").fillna(0)
    hourly_full["count"] = hourly_full["count"].astype(int)
    # 15 minutes = 0.25 nurse-hours per patient per hour
    hourly_full["required_nurse_hours"] = hourly_full["count"] * 0.25
    avg_series = hourly_full.groupby("hour")["required_nurse_hours"].mean().reindex(hours).fillna(0)
    avg_hourly_df = pd.DataFrame({"slot": slots, "avg_required_nurse_hours": avg_series.values})
    print("Average hourly required nurse-hours (daycare):")
    print(avg_hourly_df.to_string(index=False))

# Shift candidates (4h and 8h) chosen to be able to cover 08:00-21:00
four_shifts = [(8,12),(12,16),(16,20)]
eight_shifts = [(8,16),(13,21)]
candidates = []
for s,e in four_shifts:
    candidates.append({"type":"4h","name":f"{s}-{e}","cov":np.array([1 if (h>=s and h<e) else 0 for h in hours]),"cost":40})
for s,e in eight_shifts:
    candidates.append({"type":"8h","name":f"{s}-{e}","cov":np.array([1 if (h>=s and h<e) else 0 for h in hours]),"cost":70})

# Minimum staffing: at least 1 nurse between 08:00-21:00 -> enforce by making req >= 1 nurse-hour
req = np.maximum(avg_hourly_df["avg_required_nurse_hours"].values if not avg_hourly_df.empty else np.zeros(len(hours)), 1.0)

cov_mat = np.vstack([c["cov"] for c in candidates])
best_cost = float("inf"); best_combo = None
# brute-force search in small ranges (adjust range upper bounds as needed)
RANGE = range(0,7)
for combo in product(*[RANGE for _ in candidates]):
    combo = np.array(combo)
    coverage = cov_mat.T.dot(combo)
    if np.all(coverage >= req - 1e-8):
        cost = sum(combo[i] * candidates[i]["cost"] for i in range(len(candidates)))
        if cost < best_cost:
            best_cost = cost; best_combo = combo.copy()

if best_combo is None:
    print("No feasible shift mix found for daycare within search bounds.")
    solution_df = pd.DataFrame()
else:
    rows = []
    for i,c in enumerate(candidates):
        if best_combo[i] > 0:
            rows.append({"shift_type":c["type"], "shift_name":c["name"], "nurses_assigned":int(best_combo[i]), "daily_cost":int(best_combo[i])*c["cost"]})
    solution_df = pd.DataFrame(rows)
    coverage = cov_mat.T.dot(best_combo)
    util_df = pd.DataFrame({"slot": slots, "avg_required_nurse_hours": req, "nurses_on_duty": coverage, "utilization_per_nurse": np.where(coverage>0, req/coverage, np.nan)})
    print("\nOptimal daycare shift mix (daily):")
    print(solution_df.to_string(index=False))
    print("\nHourly utilization (daycare):")
    print(util_df.to_string(index=False))
    # safe plot
    try:
        plt.figure(figsize=(9,3)); plt.plot(slots, req, marker='o', label='Required'); plt.plot(slots, coverage, marker='s', label='Staffed'); plt.xticks(rotation=45); plt.legend(); plt.title("Daycare demand vs coverage"); plt.tight_layout(); plt.show()
    except Exception as e:
        print("Daycare plot skipped due to plotting error:", e)

# -----------------------------
# PART B — Specialist wards (AOS, VTS, TRS)
# -----------------------------
print("\n=== PART B: Specialist wards ===")
SPECIAL_WARDS = ["AOS","VTS","TRS"]
spec = patients[patients['ward'].isin(SPECIAL_WARDS)].copy()
print("Specialist records found:", len(spec))

# Add initial patients (10 per ward arriving 2010-01-26) — sample LOS from ward LOS distribution
los_dist = {}
for w in SPECIAL_WARDS:
    vals = spec[spec['ward'] == w]['LOS'].dropna().astype(int).values
    los_dist[w] = vals if len(vals)>0 else np.array([1,2,3,4])

# create synthetic initial patients
init_adm = pd.to_datetime("2010-01-26")
synth = []
for w in SPECIAL_WARDS:
    for i in range(10):
        sampled_los = int(np.random.choice(los_dist[w]))
        synth.append({"patientnr":f"INIT_{w}_{i+1}", "OR date": init_adm, "specilism": w, "ward": w, "ready for ward": init_adm, "LOS": sampled_los})
synth_df = pd.DataFrame(synth)
spec_all = pd.concat([spec, synth_df], ignore_index=True, sort=False)
spec_all["ready for ward"] = pd.to_datetime(spec_all["ready for ward"], errors='coerce')

# build patient-day records
patient_days = []
for _, r in spec_all.iterrows():
    adm = r["ready for ward"]
    if pd.isna(adm): continue
    los = int(r["LOS"]) if not pd.isna(r["LOS"]) else 1
    for d in range(los):
        patient_days.append({"patientnr": r["patientnr"], "ward": r["ward"], "date": (adm + pd.Timedelta(days=d)).date(), "day_on_ward": d+1, "ASA": r.get("ASAClass", np.nan)})
pd_days = pd.DataFrame(patient_days)
print("Patient-day rows (specialist):", len(pd_days))

# parse nursing_hours sheet: robust fallback to daily row mean if exact structure unclear
nh = nursing_hours.copy()
first_col = nh.columns[0]
numeric_cols = nh.select_dtypes(include=[np.number]).columns.tolist()
day_to_daily = {}
if len(numeric_cols) > 0:
    nh["daily_hours_mean"] = nh[numeric_cols].mean(axis=1)
    for idx, row in nh.iterrows():
        key = int(row[first_col]) if not pd.isna(row[first_col]) and str(row[first_col]).isdigit() else (idx+1)
        day_to_daily[key] = float(row["daily_hours_mean"])
overall_mean_daily = np.mean(list(day_to_daily.values())) if day_to_daily else 4.0

# assign daily nurse-hours per patient-day (fallback uses overall mean)
pd_days["daily_nurse_hours"] = pd_days["day_on_ward"].apply(lambda d: day_to_daily.get(int(d), overall_mean_daily))
# split to shifts (assumption)
pd_days["morning_hours"] = pd_days["daily_nurse_hours"] * 0.5
pd_days["evening_hours"] = pd_days["daily_nurse_hours"] * 0.3
pd_days["night_hours"] = pd_days["daily_nurse_hours"] * 0.2

# aggregate per ward-date
agg = pd_days.groupby(["ward","date"]).agg({"morning_hours":"sum","evening_hours":"sum","night_hours":"sum"}).reset_index()
agg["weekday"] = pd.to_datetime(agg["date"]).dt.day_name()

# Demand distribution (mean/std)
rows = []
for ward in agg["ward"].unique():
    sub = agg[agg["ward"]==ward]
    for shift in ["morning_hours","evening_hours","night_hours"]:
        rows.append({"ward": ward, "shift": shift.replace("_hours", ""), "mean": sub[shift].mean(), "std": sub[shift].std()})
demand_stats = pd.DataFrame(rows)
print("\nDemand stats (ward/shift):")
print(demand_stats.to_string(index=False))

# Initial staffing: cap utilization at 75%, effective hours per nurse per shift = 8 * 0.75
effective_hours = 8.0 * 0.75
staff_rows = []
for _, r in demand_stats.iterrows():
    required_nurses = math.ceil(r["mean"] / effective_hours) if effective_hours>0 else 0
    staff_rows.append({"ward": r["ward"], "shift": r["shift"], "mean_required_hours": r["mean"], "required_nurses_initial": required_nurses})
staffing_df = pd.DataFrame(staff_rows)
print("\nInitial staffing per ward-shift:")
print(staffing_df.to_string(index=False))

# Validation against reality
validation_rows = []
req_map = staffing_df.set_index(["ward","shift"])["required_nurses_initial"].to_dict()
for _, r in agg.iterrows():
    for shift, col in [("Morning","morning_hours"), ("Evening","evening_hours"), ("Night","night_hours")]:
        ward = r["ward"]; date = r["date"]
        workload = r[col]; assigned = int(req_map.get((ward, shift), 0))
        available = assigned * 8
        utilization = (workload / available) if available>0 else np.nan
        over = workload > available if available>0 else True
        validation_rows.append({"ward":ward,"date":date,"shift":shift,"workload_hours":workload,"assigned_nurses":assigned,"available_hours":available,"utilization":utilization,"overutilization":over})
validation_df = pd.DataFrame(validation_rows)
val_summary = validation_df.groupby(["ward","shift"]).agg(util_mean=("utilization","mean"), util_std=("utilization","std"), overutil_frac=("overutilization","mean"), overutil_count=("overutilization","sum")).reset_index()
print("\nValidation summary (util & overutil freq):")
print(val_summary.to_string(index=False))

# Iterative refinement: increase nurses until overutil_frac <= 1% or up to +3 increases
refined = staffing_df.copy()
refined["required_nurses_refined"] = refined["required_nurses_initial"]
refined["overutil_frac_after"] = 0.0
for idx, r in refined.iterrows():
    ward = r["ward"]; shift = r["shift"]; base = int(r["required_nurses_initial"])
    for extra in range(0,4):
        assigned = base + extra
        mask = (validation_df["ward"]==ward) & (validation_df["shift"]==shift)
        subset = validation_df[mask].copy()
        subset["available_adj"] = assigned * 8
        subset["over_adj"] = subset["workload_hours"] > subset["available_adj"]
        over_frac = float(subset["over_adj"].mean()) if not subset.empty else 0.0
        if over_frac <= 0.01:
            refined.at[idx,"required_nurses_refined"] = assigned
            refined.at[idx,"overutil_frac_after"] = over_frac
            break
    # if not achieved, keep base + 3 as safety
    if refined.at[idx,"required_nurses_refined"] == base:
        refined.at[idx,"required_nurses_refined"] = base + 3

print("\nRefined staffing (per ward-shift):")
print(refined.to_string(index=False))

# -----------------------------
# PART C — create a 13-week rotating schedule
# -----------------------------
print("\n=== PART C: 13-week schedule ===")

weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# Step 1. Build weekly template (ward × weekday) using mean daily hours → nurses
weekly_template = []
for ward in SPECIAL_WARDS:
    for wd in weekdays:
        sub = agg[(agg["ward"] == ward) & (agg["weekday"] == wd)]
        if sub.empty:
            m = e = n = 0.0
        else:
            m = sub["morning_hours"].mean()
            e = sub["evening_hours"].mean()
            n = sub["night_hours"].mean()
        m_n = math.ceil(m / effective_hours)
        e_n = math.ceil(e / effective_hours)
        n_n = math.ceil(n / effective_hours)
        weekly_template.append({
            "ward": ward,
            "weekday": wd,
            "morning_nurses": m_n,
            "evening_nurses": e_n,
            "night_nurses": n_n
        })

weekly_template_df = pd.DataFrame(weekly_template)
print("Weekly template (sample):")
print(weekly_template_df.head(10).to_string(index=False))

# Step 2. Estimate cost using official pattern rates (approximation)
# ------------------------------------------------------------------
patterns = {
    1: {"morning":5, "evening":0, "night":0, "rest":2, "yearly_cost":52143},
    2: {"morning":2, "evening":2, "night":2, "rest":4, "yearly_cost":55480},
    3: {"morning":2, "evening":2, "night":0, "rest":4, "yearly_cost":40150},
    4: {"morning":0, "evening":0, "night":5, "rest":5, "yearly_cost":54750},
    5: {"morning":0, "evening":4, "night":2, "rest":4, "yearly_cost":56940},
}
for p in patterns.values():
    p["weekly_cost"] = p["yearly_cost"] / 52

# Approximate total nurses per week (mean of all shift needs)
avg_weekly_nurses = (
    weekly_template_df[["morning_nurses","evening_nurses","night_nurses"]].mean().sum()
)
# Approximate pattern mix (simple heuristic)
num_pat1 = int(avg_weekly_nurses * 0.5)  # mainly morning
num_pat3 = int(avg_weekly_nurses * 0.2)  # mixed day/evening
num_pat4 = int(avg_weekly_nurses * 0.2)  # night
num_pat5 = int(avg_weekly_nurses * 0.1)  # mixed evening/night

approx_weekly_cost = (
    num_pat1 * patterns[1]["weekly_cost"]
    + num_pat3 * patterns[3]["weekly_cost"]
    + num_pat4 * patterns[4]["weekly_cost"]
    + num_pat5 * patterns[5]["weekly_cost"]
)
approx_13w_cost = approx_weekly_cost * 13

print(f"\nEstimated weekly nurses (avg): {avg_weekly_nurses:.1f}")
print(f"Approximate weekly cost: €{approx_weekly_cost:,.0f}")
print(f"Estimated total 13-week cost: €{approx_13w_cost:,.0f}")

# Step 3. Build 13-week repeating schedule (keeps Part D happy)
# -------------------------------------------------------------
schedule_rows = []
for week in range(1, 14):
    for _, r in weekly_template_df.iterrows():
        schedule_rows.append({
            "week": week,
            "ward": r["ward"],
            "weekday": r["weekday"],
            "morning_nurses": r["morning_nurses"],
            "evening_nurses": r["evening_nurses"],
            "night_nurses": r["night_nurses"],
        })

schedule_13w = pd.DataFrame(schedule_rows)
print("\n13-week schedule excerpt:")
print(schedule_13w.head(10).to_string(index=False))

weekly_cost_per_nurse = 40150 / 52
MAX_FLEX_PER_SHIFT = 2

# -----------------------------
# PART D — Flexible nurse integration
# -----------------------------
print("\n=== PART D: Flexible integration (simulation) ===")
REDUCE_FRAC = 0.10
MAX_FLEX_PER_SHIFT = 2
FLEX_COST_MULT = 1.5

weekly_template_df["regular_morning"] = (weekly_template_df["morning_nurses"] * (1-REDUCE_FRAC)).astype(int)
weekly_template_df["regular_evening"] = (weekly_template_df["evening_nurses"] * (1-REDUCE_FRAC)).astype(int)
weekly_template_df["regular_night"] = (weekly_template_df["night_nurses"] * (1-REDUCE_FRAC)).astype(int)

dates = sorted(agg["date"].unique())
sim_rows = []
for d in dates:
    wd = pd.to_datetime(d).day_name()
    for ward in SPECIAL_WARDS:
        tpl = weekly_template_df[(weekly_template_df["ward"]==ward)&(weekly_template_df["weekday"]==wd)]
        if tpl.empty:
            reg_m=reg_e=reg_n=0
        else:
            reg_m=int(tpl["regular_morning"].values[0]); reg_e=int(tpl["regular_evening"].values[0]); reg_n=int(tpl["regular_night"].values[0])
        actual = agg[(agg["ward"]==ward)&(agg["date"]==d)]
        if actual.empty:
            act_m=act_e=act_n=0.0
        else:
            act_m=float(actual["morning_hours"].values[0]); act_e=float(actual["evening_hours"].values[0]); act_n=float(actual["night_hours"].values[0])
        req_m = math.ceil(act_m / effective_hours); req_e = math.ceil(act_e / effective_hours); req_n = math.ceil(act_n / effective_hours)
        flex_m = max(0, min(MAX_FLEX_PER_SHIFT, req_m - reg_m)); flex_e = max(0, min(MAX_FLEX_PER_SHIFT, req_e - reg_e)); flex_n = max(0, min(MAX_FLEX_PER_SHIFT, req_n - reg_n))
        assigned_m = reg_m + flex_m; assigned_e = reg_e + flex_e; assigned_n = reg_n + flex_n
        sim_rows.append({"date":d,"ward":ward,"reg_m":reg_m,"req_m":req_m,"flex_m":flex_m,"assigned_m":assigned_m,"reg_e":reg_e,"req_e":req_e,"flex_e":flex_e,"assigned_e":assigned_e,"reg_n":reg_n,"req_n":req_n,"flex_n":flex_n,"assigned_n":assigned_n})
sim_df = pd.DataFrame(sim_rows)
total_flex = sim_df[["flex_m","flex_e","flex_n"]].sum().sum()
avg_flex_week = total_flex / (len(dates)/7) if len(dates)>0 else 0
weekly_reg_total = weekly_template_df[["regular_morning","regular_evening","regular_night"]].sum().sum()
weekly_reg_cost = weekly_reg_total * weekly_cost_per_nurse
weekly_flex_cost = avg_flex_week * weekly_cost_per_nurse * FLEX_COST_MULT
print(f"Flexible deployments total: {total_flex} (avg/week: {avg_flex_week:.2f})")
print(f"Weekly regular cost (reduced): {weekly_reg_cost:.2f}, weekly flex cost est: {weekly_flex_cost:.2f}")

# Overutilization comparison (approx)
fixed_overutil = validation_df["overutilization"].sum()
val_adj = []
for _, r in sim_df.iterrows():
    for (req_k, assigned_k) in [("req_m","assigned_m"),("req_e","assigned_e"),("req_n","assigned_n")]:
        req_n = r[req_k]; assigned = r[assigned_k]; val_adj.append({"ward":r["ward"],"date":r["date"],"req":req_n,"assigned":assigned,"over": req_n>assigned})
val_adj_df = pd.DataFrame(val_adj)
flex_overutil = val_adj_df["over"].sum()
print(f"Fixed overutil incidents: {fixed_overutil}, Flexible approx overutil: {flex_overutil}")

print("\nScript finished. Adjust assumptions (nursing_hours parsing, shift candidates, cost params) in the script as needed.")
