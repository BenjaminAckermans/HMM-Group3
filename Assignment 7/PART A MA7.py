#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "/Users/ariannaperini/Downloads/Data for Assignment 7.xlsx" #change this

# Read sheets
patients = pd.read_excel(file_path, sheet_name='Patients')
surgery_queue = pd.read_excel(file_path, sheet_name='Surgery Queue')
in_ward = pd.read_excel(file_path, sheet_name='In Ward')
surgeryLOS = pd.read_excel(file_path, sheet_name='surgeryLOS')

# Parameters (change N to match the number of specialists you want to assume)
start_time = 1040
end_time = 1040 + 52 * 40  # = 3120
N = 10  # number of medical specialists (change as needed)

# Build a lookup dictionary for surgery time (minutes) and LOS (days)
surgery_lookup = {}
for _, row in surgeryLOS.iterrows():
    key = (int(row['subspec']), int(row['subsub']))
    surgery_lookup[key] = {
        'surgery_min': float(row['surgery time(min)']) if not pd.isna(row['surgery time(min)']) else 0.0,
        'LOS_days': float(row['LOS(DAYS)']) if not pd.isna(row['LOS(DAYS)']) else 0.0
    }

# Helper: count weekdays in a block of consecutive days starting at a given day index.
# day_index 0 => Monday
def count_weekdays(start_day_index, num_days):
    cnt = 0
    for d in range(num_days):
        dow = (start_day_index + d) % 7  # 0=Mon ... 6=Sun
        if dow < 5:
            cnt += 1
    return cnt

# 1) OUTPATIENT: consider all patients with arrival (h) <= end_time
outpatient_df = patients[patients['arrival (h)'] <= end_time].copy()
outpatient_by_sub = outpatient_df.groupby('subspec')['timeod (min)'].sum().to_dict()
total_outpatient_min = outpatient_df['timeod (min)'].sum()

# 2) SURGERIES / OR: consider surgery queue entries with due-date op in (start_time, end_time]
surg_df = surgery_queue[(surgery_queue['due-date op (h)'] > start_time) & (surgery_queue['due-date op (h)'] <= end_time)].copy()

def get_surgery_min(row):
    key = (int(row['subspec']), int(row['subsubsp']))
    return surgery_lookup.get(key, {}).get('surgery_min', 0.0)

surg_df['surgery_min'] = surg_df.apply(get_surgery_min, axis=1)
total_or_min = surg_df['surgery_min'].sum()
or_by_subspec = surg_df.groupby('subspec')['surgery_min'].sum().to_dict()
or_by_subsub = surg_df.groupby('subsubsp')['surgery_min'].sum().to_dict()

# 3) WARD visits (15 min per visit, weekdays only)
ward_minutes_by_sub = {}
total_ward_min = 0.0

# a) current in-ward patients: remaining LOS starting day 0
for _, row in in_ward.iterrows():
    subs = int(row['subspec'])
    remaining_LOS = int(row['remaining LOS'])
    start_day_index = 0  # patient is already in ward at simulation day 0
    num_days = max(0, remaining_LOS)
    wkdays = count_weekdays(start_day_index, num_days)
    visits = wkdays + 1  # 1 visit more than LOS
    mins = visits * 15
    ward_minutes_by_sub[subs] = ward_minutes_by_sub.get(subs, 0) + mins
    total_ward_min += mins

# b) post-op ward visits for surgeries occurring this year
for _, row in surg_df.iterrows():
    subs = int(row['subspec'])
    subsub = int(row['subsubsp'])
    due_hour = int(row['due-date op (h)'])
    day_index = int(math.floor((due_hour - start_time) / 24.0))
    key = (subs, subsub)
    LOS = int(round(surgery_lookup.get(key, {}).get('LOS_days', 0.0)))
    wkdays = count_weekdays(day_index, LOS)
    visits = wkdays + 1  # 1 visit more than LOS
    mins = visits * 15
    ward_minutes_by_sub[subs] = ward_minutes_by_sub.get(subs, 0) + mins
    total_ward_min += mins

# 4) ADMIN: 5 min per outpatient, 10 min per surgery
admin_min_from_outpatients = len(outpatient_df) * 5
admin_min_from_surgeries = len(surg_df) * 10
total_admin_min = admin_min_from_outpatients + admin_min_from_surgeries

# Admin by subspecialism: allocate the 5/10 minutes to the patient's subspecialism
admin_by_sub = {}
out_count_by_sub = outpatient_df.groupby('subspec').size().to_dict()
for subs, cnt in out_count_by_sub.items():
    admin_by_sub[subs] = admin_by_sub.get(subs, 0) + cnt * 5
surg_count_by_sub = surg_df.groupby('subspec').size().to_dict()
for subs, cnt in surg_count_by_sub.items():
    admin_by_sub[subs] = admin_by_sub.get(subs, 0) + cnt * 10

# Convert minutes to hours
total_outpatient_hours = total_outpatient_min / 60.0
outpatient_hours_by_sub = {k: v / 60.0 for k, v in outpatient_by_sub.items()}

total_or_hours = total_or_min / 60.0
or_hours_by_subspec = {k: v / 60.0 for k, v in or_by_subspec.items()}
or_hours_by_subsub = {k: v / 60.0 for k, v in or_by_subsub.items()}

total_ward_hours = total_ward_min / 60.0
ward_hours_by_sub = {k: v / 60.0 for k, v in ward_minutes_by_sub.items()}

total_admin_hours = total_admin_min / 60.0
admin_hours_by_sub = {k: v / 60.0 for k, v in admin_by_sub.items()}

# 5) Education and 6) Conferences (scale with N specialists)
education_hours_total = N * 80.0
conference_hours_total = N * (2 * 5 * 8 + 24)  # 2 weeks * 40h + 24h

# Print summary
print("\n Part A: Minimum required hours")
print(f"Ward_hours_total: {total_ward_hours:.2f} hours")
print(f"Outpatient_hours_total: {total_outpatient_hours:.2f} hours")
print(f"OR_hours_total: {total_or_hours:.2f} hours")
print(f"Admin_hours_total: {total_admin_hours:.2f} hours")
print(f"Education_hours_total (N specialists): {education_hours_total:.2f} hours (N={N})")
print(f"Conference_hours_total (N specialists): {conference_hours_total:.2f} hours (N={N})")
print(f"Number_of_outpatients_considered: {len(outpatient_df)}")
print(f"Number_of_surgeries_considered: {len(surg_df)}")
print(f"Number_of_in_ward_patients: {len(in_ward)}")

# Breakdown table by subspecialism (first 4 tasks)
all_subs = sorted(set(list(ward_hours_by_sub.keys()) + list(outpatient_hours_by_sub.keys()) +
                      list(or_hours_by_subspec.keys()) + list(admin_hours_by_sub.keys())))
break_rows = []
for subs in all_subs:
    w = ward_hours_by_sub.get(subs, 0.0)
    o = outpatient_hours_by_sub.get(subs, 0.0)
    orh = or_hours_by_subspec.get(subs, 0.0)
    a = admin_hours_by_sub.get(subs, 0.0)
    break_rows.append({'subspec': subs, 'Ward_h': w, 'Outpatient_h': o, 'OR_h': orh, 'Admin_h': a})

break_df = pd.DataFrame(break_rows).sort_values('subspec').reset_index(drop=True)
print("\n Breakdown by subspecialism (hours/year)")
print(break_df.to_string(index=False))

# Bar charts for visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

# Ward
break_df.plot.bar(x='subspec', y='Ward_h', ax=axes[0], legend=False)
axes[0].set_title('Ward hours by subspecialism')
axes[0].set_xlabel('Subspecialism')
axes[0].set_ylabel('Hours (year)')

# Outpatient
break_df.plot.bar(x='subspec', y='Outpatient_h', ax=axes[1], legend=False)
axes[1].set_title('Outpatient hours by subspecialism')
axes[1].set_xlabel('Subspecialism')
axes[1].set_ylabel('Hours (year)')

# OR
break_df.plot.bar(x='subspec', y='OR_h', ax=axes[2], legend=False)
axes[2].set_title('Operating Room hours by subspecialism')
axes[2].set_xlabel('Subspecialism')
axes[2].set_ylabel('Hours (year)')

# Admin
break_df.plot.bar(x='subspec', y='Admin_h', ax=axes[3], legend=False)
axes[3].set_title('Administration hours by subspecialism')
axes[3].set_xlabel('Subspecialism')
axes[3].set_ylabel('Hours (year)')

plt.tight_layout()
plt.show()

# Totals table
totals_table = pd.DataFrame([
    ['Ward', total_ward_hours],
    ['Outpatient', total_outpatient_hours],
    ['Operating Room', total_or_hours],
    ['Administration', total_admin_hours],
    ['Education (total, N specialists)', education_hours_total],
    ['Conferences (total, N specialists)', conference_hours_total]
], columns=['Task', 'Required hours (year)'])
print("\n Totals (hours/year) ")
print(totals_table.to_string(index=False))

# Print the assumptions summary for clarity
assumptions_text = f"""
ASSUMPTIONS (explicit):
- Time window: start_time = {start_time}, end_time = {end_time}.
- Outpatient consultations considered: all patients in 'Patients' with arrival (h) <= end_time.  => {len(outpatient_df)} patients.
- Surgeries considered: all rows in 'Surgery Queue' with due-date op (h) in (start_time, end_time]. => {len(surg_df)} surgeries.
- Patients in 'In Ward' are included for remaining LOS starting at day 0. => {len(in_ward)} patients.
- Ward visits count only weekdays (Mon-Fri). The simulation start day is taken as Monday.
- Ward visits per patient: number of weekdays during LOS plus 1 extra visit.
- Ward visit duration: 15 minutes per visit.
- Admin times: 5 min per outpatient, 10 min per surgery.
- Education: 80 hours per specialist per year (parameter N = {N} specialists).
- Conferences: 2 weeks (2*5 workdays) + 24 hours per specialist => {2*5*8 + 24} hours per specialist => total {conference_hours_total} hours for N={N}.
"""
print(assumptions_text)
