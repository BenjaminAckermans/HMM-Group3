"""
Script for Part A & B

"""

import pandas as pd
import numpy as np

# --- Load data ---
path = "Data for Assignment 4.xlsx"
df = pd.read_excel(path, sheet_name=0)

# Clean column names
df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

# Parse dates
df['Arrival data'] = pd.to_datetime(df['Arrival data'], dayfirst=True, errors='coerce')

# Convert expected surgery time to minutes (numeric)
def to_minutes(x):
    if pd.isna(x):
        return np.nan
    try:
        td = pd.to_timedelta(str(x))
        return td.total_seconds() / 60
    except Exception:
        return np.nan

df['expected_minutes'] = df['Expected surgery time'].apply(to_minutes)

# --- Group by Ward ---
ward_groups = df.groupby('ward')

summary = ward_groups['expected_minutes'].agg(
    count='count',
    mean_minutes='mean',
    median_minutes='median',
    total_minutes='sum'
).reset_index()

# Calculate 8-hour (480 min) OR sessions required
summary['sessions_needed'] = summary['total_minutes'] / 480

# Urgency distribution per ward
urgency_dist = df.groupby(['ward', 'Urgency']).size().unstack(fill_value=0)

# ASA distribution per ward
asa_dist = df.groupby(['ward', 'ASAClass']).size().unstack(fill_value=0)

print("=== Patient Group Summary by Ward ===")
print(summary.round(2))

print("\n=== Urgency Distribution ===")
print(urgency_dist)

print("\n=== ASA Distribution ===")
print(asa_dist)

# --- Suggested cycle length ---
# Here we use 4 weeks (20 working days) as a practical cycle length for master surgery schedules
cycle_length_days = 28
print("\nChosen cycle length: {} days (4 weeks)".format(cycle_length_days))
