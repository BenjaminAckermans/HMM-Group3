#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 13:31:26 2025

@author: ariannaperini
"""

#!/usr/bin/env python3
"""
Assignment 6 - Part A: Daycare Ward Planning
No files are saved — outputs are printed and plotted inline.
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# === File path ===
EXCEL_PATH = "Data for Assignment 6.xlsx"

# ------------------ Functions ------------------

def load_data(path):
    xl = pd.ExcelFile(path)
    patients = xl.parse('patients')
    nurse_costs = xl.parse('nurse costs', header=None)
    return patients, nurse_costs

def extract_daycare_patients(patients_df):
    df = patients_df.copy()
    df.columns = [c.strip() for c in df.columns]
    spec_col = next((c for c in df.columns if c.lower().startswith('specil')), 'specilism')
    los_col = next((c for c in df.columns if c.lower() == 'los'), 'LOS')
    ready_col = next((c for c in df.columns if 'ready' in c.lower()), 'ready for ward')
    or_date_col = next((c for c in df.columns if c.lower().startswith('or ')), 'OR date')

    # Filter daycare patients
    daycare = df[(df[spec_col].astype(str).str.lower() == 'daycare') & (df[los_col] == 0)].copy()

    daycare[ready_col] = pd.to_datetime(daycare[ready_col])
    daycare[or_date_col] = pd.to_datetime(daycare[or_date_col])
    daycare = daycare.rename(columns={ready_col: 'admission', or_date_col: 'or_date'})

    # 4-hour stay
    daycare['departure'] = daycare['admission'] + pd.Timedelta(hours=4)
    daycare['admission_date'] = daycare['admission'].dt.date
    return daycare

def compute_hourly_average(daycare_df, from_hour=8, to_hour=21):
    days = sorted(daycare_df['admission_date'].unique())
    hours = list(range(from_hour, to_hour))
    records = []

    for d in days:
        for h in hours:
            hour_start = pd.Timestamp(pd.to_datetime(d)) + pd.Timedelta(hours=h)
            hour_end = hour_start + pd.Timedelta(hours=1)
            mask = (
                (daycare_df['admission'] < hour_end)
                & (daycare_df['departure'] > hour_start)
                & (daycare_df['admission_date'] == d)
            )
            overlaps = daycare_df.loc[mask, ['admission','departure']]
            overlap_hours = 0.0
            if not overlaps.empty:
                starts = overlaps['admission'].apply(lambda t: max(t, hour_start))
                ends = overlaps['departure'].apply(lambda t: min(t, hour_end))
                overlap_hours = ((ends - starts).dt.total_seconds() / 3600).sum()
            patient_equiv = overlap_hours
            required_nurse_hours = patient_equiv * 0.25
            records.append({'date': d, 'hour': h, 'patient_equiv': patient_equiv, 'required_nurse_hours': required_nurse_hours})

    df_hours = pd.DataFrame(records)
    avg = df_hours.groupby('hour').agg(
        avg_patient_equiv=('patient_equiv','mean'),
        avg_required_nurse_hours=('required_nurse_hours','mean'),
    ).reset_index()
    avg['required_nurses_decimal'] = avg['avg_required_nurse_hours']
    avg['required_nurses'] = avg['avg_required_nurse_hours'].apply(lambda x: max(1, math.ceil(x)))
    return avg

def parse_daycare_shift_costs(nurse_costs):
    cost_4h = None
    cost_8h = None
    for _, row in nurse_costs.iterrows():
        joined = ' '.join([str(x).strip() for x in row.dropna().astype(str).values]).lower()
        if 'daycare' in joined and '4h' in joined:
            digits = [int(s) for s in joined.split() if s.isdigit()]
            if digits: cost_4h = digits[0]
        if 'daycare' in joined and '8h' in joined:
            digits = [int(s) for s in joined.split() if s.isdigit()]
            if digits: cost_8h = digits[0]
    if cost_4h is None: cost_4h = 120
    if cost_8h is None: cost_8h = 220
    return cost_4h, cost_8h

def build_shifts(cost_4h, cost_8h):
    return [
        {'name':'4h_08_12','start':8,'end':12,'cost':cost_4h},
        {'name':'4h_12_16','start':12,'end':16,'cost':cost_4h},
        {'name':'4h_16_20','start':16,'end':20,'cost':cost_4h},
        {'name':'4h_17_21','start':17,'end':21,'cost':cost_4h},
        {'name':'8h_08_16','start':8,'end':16,'cost':cost_8h},
        {'name':'8h_13_21','start':13,'end':21,'cost':cost_8h},
    ]

def solve_shift_allocation(avg_by_hour, shifts):
    from itertools import product
    hours = avg_by_hour['hour'].tolist()
    req = dict(zip(hours, avg_by_hour['required_nurses']))
    coverage = {s['name']: [1 if (h >= s['start'] and h < s['end']) else 0 for h in hours] for s in shifts}
    shift_names = [s['name'] for s in shifts]

    best_cost = float('inf')
    best_sol = None

    for combo in product(range(3), repeat=len(shifts)):  # allow up to 2 nurses per shift type
        cost = sum(combo[i]*shifts[i]['cost'] for i in range(len(shifts)))
        if cost >= best_cost: continue
        ok = True
        for i, h in enumerate(hours):
            cover = sum(combo[j]*coverage[shifts[j]['name']][i] for j in range(len(shifts)))
            if cover < req[h]:
                ok = False; break
        if ok:
            best_cost = cost
            best_sol = dict(zip(shift_names, combo))

    if not best_sol:
        raise ValueError("No feasible shift allocation found.")
    return best_sol, best_cost

def compute_utilization(avg_by_hour, solution, shifts):
    hours = avg_by_hour['hour'].tolist()
    coverage = {s['name']: [1 if (h>=s['start'] and h<s['end']) else 0 for h in hours] for s in shifts}
    staffed = []
    for i, h in enumerate(hours):
        staffed.append(sum(solution[s['name']] * coverage[s['name']][i] for s in shifts))
    avg_by_hour['staffed_nurses'] = staffed
    avg_by_hour['utilization'] = avg_by_hour.apply(
        lambda r: r['avg_required_nurse_hours']/r['staffed_nurses'] if r['staffed_nurses']>0 else 0, axis=1
    )
    return avg_by_hour

# ------------------ Main ------------------

patients, nurse_costs = load_data(EXCEL_PATH)
daycare = extract_daycare_patients(patients)

print(f"Loaded {len(daycare)} daycare patients.")

avg_by_hour = compute_hourly_average(daycare)
cost_4h, cost_8h = parse_daycare_shift_costs(nurse_costs)
shifts = build_shifts(cost_4h, cost_8h)

solution, total_cost = solve_shift_allocation(avg_by_hour, shifts)
avg_by_hour = compute_utilization(avg_by_hour, solution, shifts)

# Print results
print("\n=== Average Hourly Care Requirements ===")
print(avg_by_hour[['hour','avg_patient_equiv','avg_required_nurse_hours','required_nurses']])

print("\n=== Optimal Shift Allocation ===")
for s in shifts:
    print(f"{s['name']:10s} {s['start']:02d}:00-{s['end']:02d}:00  → {solution[s['name']]} nurse(s), cost €{s['cost']} each")
print(f"Total daily cost: €{total_cost}")

avg_util = avg_by_hour[avg_by_hour['staffed_nurses']>0]['utilization'].mean()
print(f"\nAverage nurse utilization: {avg_util:.2%}")
print(f"Peak staffed nurses: {avg_by_hour['staffed_nurses'].max()}")

# Plot
plt.figure(figsize=(10,4))
plt.plot(avg_by_hour['hour'], avg_by_hour['required_nurses_decimal'], marker='o', label='Avg required nurses (decimal)')
plt.plot(avg_by_hour['hour'], avg_by_hour['staffed_nurses'], marker='s', label='Staffed nurses (schedule)')
plt.title("Daycare Ward: Demand vs Staffing (08:00–21:00)")
plt.xlabel("Hour of day")
plt.ylabel("Nurses")
plt.grid(True)
plt.legend()
plt.show()
