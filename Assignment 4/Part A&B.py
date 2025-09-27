"""
Script for Part A & B

"""
import pandas as pd
import numpy as np

# === Load Data ===
file_path = "Data for Assignment 4.xlsx"  # adjust if needed
df = pd.read_excel(file_path, sheet_name="Blad1")

# Clean column names
df.columns = [c.strip() for c in df.columns]

# Convert arrival date
df['Arrival data'] = pd.to_datetime(df['Arrival data'], errors='coerce')

# Convert Expected Surgery Time (hh:mm:ss) → minutes
def to_minutes_safe(val):
    try:
        parts = str(val).split(":")
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return h*60 + m + s/60
        return np.nan
    except:
        return np.nan

df['Expected_minutes'] = df['Expected surgery time'].apply(to_minutes_safe)

# === PART A ANALYSIS ===

# Patient counts per ward
ward_counts = df['ward'].value_counts()

# Urgency distribution per ward
urgency_dist = df.groupby(['ward', 'Urgency']).size().unstack(fill_value=0)

# Ward-level summary
ward_summary = df.groupby('ward').agg(
    patients=('patientnr', 'count'),
    total_expected_minutes=('Expected_minutes', 'sum'),
    avg_expected_minutes=('Expected_minutes', 'mean'),
    median_expected_minutes=('Expected_minutes', 'median'),
    avg_los=('LOS Ward', 'mean'),
    median_los=('LOS Ward', 'median')
).reset_index()

# Estimate sessions needed
# One session = 8 hours = 480 minutes
ward_summary['sessions_needed_total'] = ward_summary['total_expected_minutes'] / 480

# Planning window = Feb 1, 2010 – Jan 31, 2013 (~156 weeks)
planning_start = pd.Timestamp("2010-02-01")
planning_end = pd.Timestamp("2013-01-31")
weeks_in_window = ((planning_end - planning_start).days + 1) / 7

ward_summary['sessions_per_week'] = ward_summary['sessions_needed_total'] / weeks_in_window

# Round for readability
ward_summary = ward_summary.round({
    'avg_expected_minutes': 1,
    'median_expected_minutes': 1,
    'avg_los': 1,
    'median_los': 1,
    'sessions_per_week': 2
})

# pd.set_option("display.max_columns", None)  # show all columns
# pd.set_option("display.max_rows", None)     # show all rows
# pd.set_option("display.width", 0)           # don't wrap lines

# # === OUTPUTS === 
print("\nPatient Counts per Ward:")
print(ward_counts)

print("\nUrgency Distribution per Ward:")
print(urgency_dist)

print("\nWard Summary (Part A):")
print(ward_summary.to_string(index=False))


