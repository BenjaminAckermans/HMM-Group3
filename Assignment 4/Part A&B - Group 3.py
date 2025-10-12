"""
Script for Part A & B

"""

import pandas as pd
import numpy as np
from datetime import timedelta

# Load and Prepare the Data

# Load the Excel dataset
path = "Data for Assignment 4.xlsx"
df = pd.read_excel(path, sheet_name=0)

# Clean column names for easier reference
df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

# Convert arrival date to datetime
df['Arrival data'] = pd.to_datetime(df['Arrival data'], dayfirst=True, errors='coerce')

# Convert "Expected surgery time" to minutes
def to_minutes(x):
    if pd.isna(x):
        return np.nan
    try:
        td = pd.to_timedelta(str(x))
        return td.total_seconds() / 60
    except Exception:
        try:
            return float(x)
        except Exception:
            return np.nan

df['expected_minutes'] = df['Expected surgery time'].apply(to_minutes)


# PART A: Patient Grouping by Ward (3 patient groups)

# We group patients into 3 groups based on ward: aos, vts, trs
ward_groups = df.groupby('ward')

# Calculate key descriptive statistics for each group
summary = ward_groups['expected_minutes'].agg(
    count='count',
    mean_minutes='mean',
    median_minutes='median',
    total_minutes='sum'
).reset_index()

# Calculate number of 8-hour sessions required (480 minutes/session)
summary['sessions_needed'] = summary['total_minutes'] / 480

# Distribution of urgency per ward
urgency_dist = df.groupby(['ward', 'Urgency']).size().unstack(fill_value=0)

# Distribution of ASA classification per ward
asa_dist = df.groupby(['ward', 'ASAClass']).size().unstack(fill_value=0)

# Print analysis results for Part A
print("Part A: Patient Group Summary by Ward ")
print(summary.round(2))
print("\n Urgency Distribution ")
print(urgency_dist)
print("\n ASA Distribution ")
print(asa_dist)

# Choose cycle length: 4 weeks (28 days) – common in master OR scheduling
cycle_length_days = 28
print("\nChosen cycle length: {} days (4 weeks)".format(cycle_length_days))


# PART B: OR Schedule Implementation (with key constraints)

# Step 1: Assign due-dates based on urgency
# Mapping urgency levels to weeks until due
urg_map = {1:2, 2:6, 3:12, 4:26}

def urgency_to_weeks(u):
    """
    Map urgency codes to the number of weeks until surgery is due.
    If urgency missing, assume 12 weeks.
    """
    try:
        if pd.isna(u):
            return 12
        ui = int(float(u))
        return urg_map.get(ui, 12)
    except Exception:
        return 12

df['target_weeks'] = df['Urgency'].apply(urgency_to_weeks)
df['due_date'] = df['Arrival data'] + pd.to_timedelta(df['target_weeks']*7, unit='D')

# Convert due-dates to due-week (Monday of the week)
def week_start(dt):
    if pd.isna(dt):
        return pd.NaT
    return (dt - pd.Timedelta(days=dt.weekday())).normalize()

df['due_week_start'] = df['due_date'].apply(week_start)

# Step 2: Define planning horizon
plan_start = pd.Timestamp('2010-02-01')
plan_end = pd.Timestamp('2013-01-31')

# Keep only patients within planning horizon
df = df[df['Arrival data'].notna() & (df['Arrival data'] <= plan_end)].copy()

# Compute arrival week for each patient
def week_of(dt):
    return (dt - pd.Timedelta(days=dt.weekday())).normalize()

df['arrival_week_start'] = df['Arrival data'].apply(week_of)


# --- Step 3: Calculate sessions per week for each ward ---
weekly_demand = df.groupby(['arrival_week_start', 'ward'])['expected_minutes'].sum().reset_index()
all_weeks = pd.date_range(start=week_of(plan_start), end=week_of(plan_end), freq='7D')
wards = df['ward'].dropna().unique()

# Average weekly demand → sessions per ward
mean_weekly = {}
for w in wards:
    s = weekly_demand[weekly_demand['ward']==w].set_index('arrival_week_start').reindex(all_weeks, fill_value=0)['expected_minutes']
    mean_weekly[w] = s.mean()

# Number of sessions per week per ward
sessions_per_week = {w: int(np.ceil(max(1, mean_weekly[w]/480))) for w in wards}


# Step 4: Build OR sessions (capacity = 480 min)
session_length = 480
sessions = []
for week_start in all_weeks:
    for w in wards:
        for sidx in range(sessions_per_week[w]):
            sessions.append({
                'session_id': f"{week_start.date()}_{w}_{sidx}",
                'ward': w,
                'week_start': week_start,
                'remaining_minutes': session_length,
                'scheduled_minutes': 0,
                'patients': []
            })
sessions_df = pd.DataFrame(sessions).reset_index(drop=True)


# Step 5: Patient selection methodology
# Patients sorted by arrival date → schedule in order of arrival
patients = df.sort_values('Arrival data').copy()
patients['scheduled'] = False
patients['scheduled_session'] = None
patients['scheduled_date'] = pd.NaT

# Key constraint: if due-date < 6 weeks, cannot schedule more than 1 week before due-date week
def compute_earliest(pat):
    arr_week = pat['arrival_week_start']
    if pd.isna(pat['due_date']) or pd.isna(pat['due_week_start']):
        return arr_week
    days_to_due = (pat['due_date'] - pat['arrival_week_start']).days
    if days_to_due < 42:  # less than 6 weeks
        return max(arr_week, pat['due_week_start'] - pd.Timedelta(days=7))
    return arr_week

patients['earliest_week'] = patients.apply(compute_earliest, axis=1)

# Build quick lookup of sessions by ward + week
sessions_index = {}
for idx, row in sessions_df.iterrows():
    sessions_index.setdefault((row['ward'], row['week_start']), []).append(idx)


# Step 6: Scheduling patients into sessions
for idx, pat in patients.iterrows():
    ward = pat['ward']
    if pd.isna(ward):
        continue
    earliest = pat['earliest_week']
    allocated = False
    
    # First try: schedule within normal session capacity (480 min)
    for week in all_weeks[all_weeks >= earliest]:
        key = (ward, week)
        if key not in sessions_index:
            continue
        for sidx in sessions_index[key]:
            if sessions_df.at[sidx, 'remaining_minutes'] >= pat['expected_minutes']:
                # Allocate patient
                sessions_df.at[sidx, 'remaining_minutes'] -= pat['expected_minutes']
                sessions_df.at[sidx, 'scheduled_minutes'] += pat['expected_minutes']
                sessions_df.at[sidx, 'patients'].append(pat['patientnr'])
                patients.at[idx, 'scheduled'] = True
                patients.at[idx, 'scheduled_session'] = sessions_df.at[sidx, 'session_id']
                patients.at[idx, 'scheduled_date'] = sessions_df.at[sidx, 'week_start']
                allocated = True
                break
        if allocated:
            break
    
    # Second try: allow up to 120 minutes overtime if no space found
    if not allocated:
        for week in all_weeks[all_weeks >= earliest]:
            key = (ward, week)
            if key not in sessions_index:
                continue
            for sidx in sessions_index[key]:
                if sessions_df.at[sidx, 'scheduled_minutes'] + pat['expected_minutes'] <= session_length + 120:
                    sessions_df.at[sidx, 'scheduled_minutes'] += pat['expected_minutes']
                    sessions_df.at[sidx, 'patients'].append(pat['patientnr'])
                    patients.at[idx, 'scheduled'] = True
                    patients.at[idx, 'scheduled_session'] = sessions_df.at[sidx, 'session_id']
                    patients.at[idx, 'scheduled_date'] = sessions_df.at[sidx, 'week_start']
                    allocated = True
                    break
            if allocated:
                break


# Step 7: Calculate performance metrics 
sessions_df['utilization'] = sessions_df['scheduled_minutes'] / session_length
sessions_df['overtime'] = sessions_df['scheduled_minutes'].apply(lambda x: max(0, x - session_length))

# Global averages
avg_util = sessions_df['utilization'].mean()
avg_overtime = sessions_df['overtime'].mean()

# Identify late patients (scheduled after due-week)
scheduled_pats = patients[patients['scheduled']==True].copy()
scheduled_pats['late'] = False
scheduled_pats.loc[scheduled_pats['scheduled_date'].notna(), 'late'] = (
    scheduled_pats['scheduled_date'] > scheduled_pats['due_week_start']
)

num_late = int(scheduled_pats['late'].sum())
num_unscheduled = int((patients['scheduled']==False).sum())


# Step 8: Report results 
print("\n=== Part B: OR Scheduling Summary ===")
print("Sessions per week by ward:")
for w, s in sessions_per_week.items():
    print(f"  Ward {w}: {s} sessions/week")

print(f"\nAverage session utilization across all sessions: {avg_util:.3f}")
print(f"Average overtime per session (minutes): {avg_overtime:.1f}")
print(f"Number of operations scheduled late: {num_late}")
print(f"Number of operations not scheduled (resource shortage): {num_unscheduled}")

# Ward-level breakdowns
util_by_ward = sessions_df.groupby('ward')['utilization'].mean()
overtime_by_ward = sessions_df.groupby('ward')['overtime'].mean()
late_by_ward = scheduled_pats.groupby('ward')['late'].sum()

print("\n=== Ward-level Metrics ===")
print("Average utilization by ward:")
print(util_by_ward.round(3))
print("\nAverage overtime (minutes) by ward:")
print(overtime_by_ward.round(1))
print("\nNumber of late patients by ward:")
print(late_by_ward)

# Preview of scheduled patients
print("\nFirst 10 scheduled patients (preview):")
print(scheduled_pats[['patientnr','ward','Arrival data','due_date','scheduled_date','scheduled_session','expected_minutes','late']].head(10))
