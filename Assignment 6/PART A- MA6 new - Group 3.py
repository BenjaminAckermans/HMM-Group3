#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 17:17:24 2025

@author: ariannaperini
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
import matplotlib.pyplot as plt
from tabulate import tabulate  # for clean console tables


XLS_PATH = "Data for Assignment 6.xlsx"
HOURS_WINDOW = list(range(8, 21))  # 08:00–21:00
HOUR_LABELS = [f"{h:02d}:00-{h+1:02d}:00" for h in HOURS_WINDOW]
MIN_NURSES = 1
DEFAULT_COST_4H = 120
DEFAULT_COST_8H = 220
PATIENT_NURSE_HOUR_PER_HOUR = 0.25  # 15 minutes per patient per hour

# FUNCTIONS

def parse_admit_dt(or_date, ready_for_ward):
    if pd.isna(or_date) or pd.isna(ready_for_ward):
        return pd.NaT
    try:
        if isinstance(ready_for_ward, pd.Timestamp):
            t = ready_for_ward.time()
        else:
            t = pd.to_datetime(str(ready_for_ward)).time()
        return pd.Timestamp(datetime.combine(or_date.date(), t))
    except Exception:
        return pd.NaT

def parse_daycare_shift_costs_robust(nurse_costs_df):
    cost_4h = DEFAULT_COST_4H
    cost_8h = DEFAULT_COST_8H
    try:
        first_col = nurse_costs_df.iloc[:, 0].astype(str).str.lower()
        if first_col.str.contains('daycare', na=False).any():
            idx = first_col[first_col.str.contains('daycare', na=False)].index[0]
            for i in range(max(0, idx - 6), min(len(nurse_costs_df), idx + 7)):
                row_text = ' '.join(nurse_costs_df.iloc[i].astype(str).values).lower()
                for hours, var in [('4h', 'cost_4h'), ('8h', 'cost_8h')]:
                    if hours in row_text and 'shift' in row_text:
                        for c in nurse_costs_df.columns:
                            val = nurse_costs_df.at[i, c]
                            try:
                                num = float(val)
                                if 5 <= num <= 2000:
                                    if hours == '4h': cost_4h = int(num)
                                    else: cost_8h = int(num)
                                    break
                            except: continue
    except Exception:
        pass
    return cost_4h, cost_8h

def load_and_filter_daycare(xls_path):
    xls = pd.ExcelFile(xls_path)
    patients = pd.read_excel(xls, sheet_name='patients')
    nurse_costs = pd.read_excel(xls, sheet_name='nurse costs').fillna('')
    daycare_mask = (patients['LOS'] == 0) | (patients['group'].astype(str).str.lower().str.contains('daycare', na=False))
    daycare = patients[daycare_mask].copy()
    # find ready column
    ready_col = next((c for c in patients.columns if 'ready' in str(c).lower()), None)
    if ready_col is None:
        raise ValueError("Could not find 'ready for ward' column.")
    daycare['admit_dt'] = daycare.apply(lambda r: parse_admit_dt(r['OR date'], r[ready_col]), axis=1)
    daycare = daycare.dropna(subset=['admit_dt'])
    # enforce empty at 08:00
    daycare = daycare[daycare['admit_dt'].dt.time >= time(8, 0)].copy()
    daycare['end_dt'] = daycare['admit_dt'] + pd.Timedelta(hours=4)
    daycare['date'] = daycare['admit_dt'].dt.date
    daycare['weekday'] = daycare['admit_dt'].dt.weekday
    return daycare, nurse_costs

def compute_average_pattern(daycare):
    working = daycare[daycare['weekday'] < 5].copy()
    if working.empty:
        raise ValueError("No working-day daycare records after filtering.")
    dates = sorted(working['date'].unique())
    rows = []
    for d in dates:
        df_day = working[working['date'] == d]
        counts = []
        for h in HOURS_WINDOW:
            slot_start = pd.Timestamp(datetime.combine(d, time(h, 0)))
            slot_end = slot_start + pd.Timedelta(hours=1)
            cnt = int(((df_day['admit_dt'] < slot_end) & (df_day['end_dt'] > slot_start)).sum())
            counts.append(cnt)
        rows.append(counts)
    df_counts = pd.DataFrame(rows, columns=HOUR_LABELS, index=pd.to_datetime(dates))
    avg_patients = df_counts.mean(axis=0)
    avg_required_nurse_hours = avg_patients * PATIENT_NURSE_HOUR_PER_HOUR
    required_nurses = np.maximum(np.ceil(avg_required_nurse_hours).astype(int), MIN_NURSES)
    return df_counts, avg_patients, avg_required_nurse_hours, required_nurses

def build_shift_candidates():
    shift_lengths = {'4h': 4, '8h': 8}
    candidates = []
    coverage = {}
    for typ, length in shift_lengths.items():
        for s in range(8, 21):
            cover = {h: (1 if (s <= h < s + length) else 0) for h in HOURS_WINDOW}
            if sum(cover.values()) > 0:
                candidates.append((typ, s))
                coverage[(typ, s)] = cover
    return candidates, coverage

def greedy_heuristic(candidates, coverage, required_nurses, cost_4h, cost_8h):
    remaining = {h: int(required_nurses[HOUR_LABELS.index(f"{h:02d}:00-{h+1:02d}:00")]) for h in HOURS_WINDOW}
    selected = {}
    while any(remaining[h] > 0 for h in HOURS_WINDOW):
        best = None
        best_metric = 1e9
        for k in candidates:
            new_units = sum(1 for h in HOURS_WINDOW if coverage[k][h] == 1 and remaining[h] > 0)
            if new_units == 0: continue
            cost = cost_4h if k[0] == '4h' else cost_8h
            metric = cost / new_units
            if metric < best_metric:
                best_metric = metric
                best = k
        if best is None: break
        selected[best] = selected.get(best, 0) + 1
        for h in HOURS_WINDOW:
            if coverage[best][h] == 1 and remaining[h] > 0:
                remaining[h] -= 1
    return selected

def compute_staffing_from_solution(solution, coverage):
    staffing = {h: 0 for h in HOURS_WINDOW}
    for k, cnt in solution.items():
        for h in HOURS_WINDOW:
            staffing[h] += coverage[k][h] * cnt
    return pd.Series([staffing[h] for h in HOURS_WINDOW], index=HOUR_LABELS)

def compute_utilization_table(avg_required_nurse_hours, staffed_series):
    util = []
    for i, hlabel in enumerate(HOUR_LABELS):
        required = float(avg_required_nurse_hours.iloc[i])
        staffed = float(staffed_series.iloc[i])
        util.append(required / staffed if staffed > 0 else 0.0)
    return pd.DataFrame({
        'Hour': HOUR_LABELS,
        'Avg Required Nurse-Hours': avg_required_nurse_hours.values,
        'Staffed Nurses': staffed_series.values,
        'Utilization (%)': np.array(util) * 100
    })


# MAIN
def main():
    daycare, nurse_costs = load_and_filter_daycare(XLS_PATH)
    df_counts, avg_patients, avg_required_nurse_hours, required_nurses = compute_average_pattern(daycare)
    cost_4h, cost_8h = parse_daycare_shift_costs_robust(nurse_costs)

    print("\n=== DAYCARE WARD STAFFING ANALYSIS (PART A) ===")
    print(f"Loaded {len(daycare)} daycare patient records.")
    print(f"Parsed per-shift costs → 4h: €{cost_4h}, 8h: €{cost_8h}")

    candidates, coverage = build_shift_candidates()
    solution = greedy_heuristic(candidates, coverage, required_nurses, cost_4h, cost_8h)

    # Shifts table
    shift_rows = []
    for (typ, s), cnt in sorted(solution.items()):
        length = 4 if typ == '4h' else 8
        cost_each = cost_4h if typ == '4h' else cost_8h
        shift_rows.append([typ, f"{s:02d}:00–{s+length:02d}:00", cnt, f"€{cost_each}", f"€{cnt*cost_each}"])
    shift_df = pd.DataFrame(shift_rows, columns=['Type', 'Shift', 'Count', 'Cost Each', 'Total'])
    total_daily_cost = sum(cnt * (cost_4h if typ == '4h' else cost_8h) for (typ, _), cnt in solution.items())

    # Utilization table
    staffed_series = compute_staffing_from_solution(solution, coverage)
    util_table = compute_utilization_table(avg_required_nurse_hours, staffed_series)

    # Print clean tables
    print("\n--- Average Hourly Requirements ---")
    avg_table = pd.DataFrame({
        'Hour': HOUR_LABELS,
        'Avg Patients': avg_patients.values,
        'Req. Nurse-Hours': avg_required_nurse_hours.values,
        'Required Nurses': required_nurses
    })
    print(tabulate(avg_table, headers='keys', tablefmt='rounded_grid', showindex=False, floatfmt=".3f"))

    print("\n--- Optimal Shift Allocation ---")
    print(tabulate(shift_df, headers='keys', tablefmt='rounded_grid', showindex=False))

    print(f"\nTotal Daily Cost: €{total_daily_cost:.2f}")

    print("\n--- Utilization by Hour ---")
    print(tabulate(util_table, headers='keys', tablefmt='rounded_grid', showindex=False, floatfmt=".2f"))

    avg_util = util_table['Utilization (%)'][util_table['Staffed Nurses'] > 0].mean()
    peak_staffed = util_table['Staffed Nurses'].max()
    print(f"\nAverage Nurse Utilization: {avg_util:.1f}%")
    print(f"Peak Staffed Nurses: {peak_staffed}")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(HOUR_LABELS, avg_required_nurse_hours.values, marker='o', label='Avg required nurse-hours')
    plt.plot(HOUR_LABELS, staffed_series.values, marker='s', label='Staffed nurses')
    plt.xticks(rotation=45)
    plt.ylabel('Nurses')
    plt.title('Daycare Ward: Demand vs Staffing (08:00–21:00)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
