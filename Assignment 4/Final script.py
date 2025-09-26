# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd

# Load the Excel file
df = pd.read_excel("Data for Assignment 4.xlsx", engine="openpyxl")

# Convert 'Expected surgery time' and 'actual surgery time' from datetime.time to total minutes
def time_to_minutes(t):
    if pd.isnull(t):
        return None
    return t.hour * 60 + t.minute + t.second / 60

df["expected_minutes"] = df["Expected surgery time"].apply(time_to_minutes)
df["actual_minutes"] = df["actual surgery time"].apply(time_to_minutes)

# Group by 'ward' and calculate average durations and total patients
ward_summary = df.groupby("ward").agg(
    avg_expected_duration_min=("expected_minutes", "mean"),
    avg_actual_duration_min=("actual_minutes", "mean"),
    total_patients=("patientnr", "count")
).reset_index()

# Count number of patients per specialty group within each ward
group_counts = df.groupby(["ward", "GROUP"]).size().reset_index(name="patient_count")

# Display results
print("Ward Summary:")
print(ward_summary)

print("\nPatient Counts per Specialty Group in Each Ward:")
print(group_counts)
