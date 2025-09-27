"""
Script for Part C, D & E

"""
import pandas as pd
import numpy as np
from datetime import timedelta

# --- Load the dataset ---
df = pd.read_excel("Data for Assignment 4.xlsx", engine="openpyxl")

# --- Convert surgery times to minutes ---
def time_to_minutes(t):
    if pd.isnull(t):
        return np.nan
    return t.hour * 60 + t.minute + t.second / 60

df["Expected_minutes"] = df["Expected surgery time"].apply(time_to_minutes)
df["Actual_minutes"] = df["actual surgery time"].apply(time_to_minutes)

# Fill missing values sensibly for simulation:
median_actual = df["Actual_minutes"].median()
df["Actual_minutes_filled"] = df["Actual_minutes"].fillna(df["Expected_minutes"]).fillna(median_actual)
df["Expected_minutes_filled"] = df["Expected_minutes"].fillna(df["Actual_minutes_filled"])

# --- Define 12 patient groups based on Dutch keywords ---
group_keywords = {
    "Colorectale resecties": ["colectomie", "hemicolectomie", "rectum"],
    "Hepatobiliair & pancreas": ["lever", "pancreas", "whipple"],
    "Maag / slokdarm": ["gastrectomie", "oesophagus"],
    "Breast / mamma": ["mamma", "mastectomie", "lumpectomie"],
    "Endocrien / schildklier": ["schildklier", "thyroid"],
    "Proctologie / perianaal": ["fistel", "fissuur", "hemorroïden"],
    "Hernia / buikwand": ["liesbreuk", "inguinalis", "littekebreuk", "lap chole"],
    "Vasculair": ["fem-pop", "aneurysma", "carotis", "shunt"],
    "Sarcomen / weke delen": ["sarcoom", "tumor", "lipoom"],
    "Trauma / algemeen": ["trauma", "laparotomie", "osteosynthese"],
    "Orthopedie klein volume": ["prothese", "pseudo-artrose"],
    "Overige laag volume": []
}

def assign_group(description):
    if pd.isnull(description):
        return "Overige laag volume"
    desc = str(description).lower()
    for group, keywords in group_keywords.items():
        if any(keyword in desc for keyword in keywords):
            return group
    return "Overige laag volume"

df["Surgery Group"] = df["COMBI CTG (description in Dutch)"].apply(assign_group)

# --- PART C: Summary based on expected surgery time ---
session_duration = 480  # minutes per OR session

summary_c = df.groupby("Surgery Group").agg(
    Patient_Count=("patientnr", "count"),
    Median_Expected_Time=("Expected_minutes", "median"),
    Total_Expected_Minutes=("Expected_minutes", "sum")
).reset_index()

summary_c["Sessions_per_4wk_cycle"] = (summary_c["Total_Expected_Minutes"] / session_duration).round()
summary_c["Typical_Patients_per_Session"] = (session_duration / summary_c["Median_Expected_Time"]).round()

print("=== PART C: Patient Group Summary ===")
for _, row in summary_c.iterrows():
    print(f"Group: {row['Surgery Group']}")
    print(f"  Patient Count: {row['Patient_Count']}")
    print(f"  Median Expected Time (min): {row['Median_Expected_Time']:.1f}")
    print(f"  Total Expected Minutes: {row['Total_Expected_Minutes']:.1f}")
    print(f"  Estimated Sessions per 4-week Cycle (480 min each): {row['Sessions_per_4wk_cycle']}")
    print(f"  Typical Patients per 480-min Session: {row['Typical_Patients_per_Session']}")
    print("-" * 60)

# --- PART D: OR Scheduling Simulation using actual surgery time (no postponement) ---
df["Arrival data"] = pd.to_datetime(df["Arrival data"], errors='coerce')

urgency_weeks = {
    "< 2 weeks": 2,
    "< 6 weeks": 6,
    "< 12 weeks": 12,
    "> 12 weeks": 52
}
df["Due Weeks"] = df["Urgency"].map(urgency_weeks).fillna(12)
df["Due Date"] = df["Arrival data"] + df["Due Weeks"].apply(lambda x: timedelta(weeks=int(x)))
df["Due Week"] = df["Due Date"].dt.isocalendar().week
df["Scheduled Week"] = df["Arrival data"].dt.isocalendar().week

group_summary_d = []

# Use filled actual minutes for computing totals
for group, group_df in df.groupby("Surgery Group"):
    total_minutes = group_df["Actual_minutes_filled"].sum()
    sessions = int(np.ceil(total_minutes / session_duration)) if total_minutes > 0 else 0
    utilization = (total_minutes / (sessions * session_duration)) * 100 if sessions > 0 else 0
    overtime = max(0, total_minutes - (sessions * session_duration))
    # late ops approximation used previously: compare Scheduled Week (arrival) with Due Week
    late_ops = (group_df["Scheduled Week"] > group_df["Due Week"]).sum()

    group_summary_d.append({
        "Group": group,
        "Sessions": sessions,
        "Utilization (%)": round(utilization, 2),
        "Overtime (min)": round(overtime, 2),
        "Late Operations": int(late_ops),
        "Total_Actual_Minutes": total_minutes
    })

print("\n=== PART D: OR Scheduling Metrics (NO postponement) ===")
total_sessions_d = 0
total_minutes_d = 0
total_overtime_d = 0
total_late_ops_d = 0

for summary in group_summary_d:
    print(f"Group: {summary['Group']}")
    print(f"  Sessions: {summary['Sessions']}")
    print(f"  Utilization (%): {summary['Utilization (%)']}")
    print(f"  Overtime (min): {summary['Overtime (min)']}")
    print(f"  Late Operations: {summary['Late Operations']}")
    print("-" * 60)
    total_sessions_d += summary['Sessions']
    total_minutes_d += summary['Sessions'] * session_duration
    total_overtime_d += summary['Overtime (min)']
    total_late_ops_d += summary['Late Operations']

print("=== OVERALL METRICS (D) ===")
print(f"  Total Sessions: {total_sessions_d}")
print(f"  Total Scheduled Time (min): {total_minutes_d}")
print(f"  Total Overtime (min): {round(total_overtime_d, 2)}")
print(f"  Total Late Operations: {total_late_ops_d}")

# --- PART E: Postponement rule simulation ---
# Rules implemented:
#  - Do not start another surgery in the session if expected overtime > 30 minutes
#  - First operation in a session is always started
#  - Expected overtime is computed using Expected_minutes (fallback to Actual if missing)
#  - Postponed operations are moved to the next session for that patient group
#  - We assume one session per week per group. The first session's week = min Scheduled Week for that group.

def simulate_postponement_for_group(group_df, session_duration=480, max_week_mod=52):
    """
    Simulate sequential sessions for a single group using the postponement rule.
    Returns dict with sessions, utilization (%), total_overtime (min), late_operations_count, total_actual_minutes.
    """
    # Sort queue by Scheduled Week (arrival), then arrival date to preserve intended order
    q = group_df.sort_values(["Scheduled Week", "Arrival data", "patientnr"]).copy()
    # build list of operations (dictionaries) in queue order
    queue = []
    for _, r in q.iterrows():
        queue.append({
            "patientnr": r.get("patientnr"),
            "Expected": r.get("Expected_minutes_filled", 0),
            "Actual": r.get("Actual_minutes_filled", 0),
            "Due_Week": int(r.get("Due Week")) if not pd.isnull(r.get("Due Week")) else np.nan,
            "Orig_Scheduled_Week": int(r.get("Scheduled Week")) if not pd.isnull(r.get("Scheduled Week")) else np.nan,
            "Arrival": r.get("Arrival data")
        })

    if len(queue) == 0:
        return {"Sessions": 0, "Utilization (%)": 0.0, "Overtime (min)": 0.0, "Late Operations": 0, "Total_Actual_Minutes": 0.0}

    initial_week = int(q["Scheduled Week"].min()) if not q["Scheduled Week"].isnull().all() else 1

    session_index = 0
    scheduled_ops = []  # store ops with assigned session & week
    total_actual_minutes = 0.0
    total_overtime = 0.0

    # Process until queue empty
    while len(queue) > 0:
        session_index += 1
        # map session index to week number (wrapping mod max_week_mod)
        assigned_week = ((initial_week - 1 + session_index - 1) % max_week_mod) + 1
        current_time = 0.0
        first_in_session = True
        session_actual_sum = 0.0

        # Attempt to schedule as many front-of-queue operations as allowed by rule
        while len(queue) > 0:
            op = queue[0]  # peek at first queued operation
            if first_in_session:
                # always start first op in session
                actual = op["Actual"]
                current_time += actual
                session_actual_sum += actual
                total_actual_minutes += actual

                # assign and remove
                op_assigned = op.copy()
                op_assigned.update({"Assigned_Session": session_index, "Assigned_Week": int(assigned_week)})
                scheduled_ops.append(op_assigned)
                queue.pop(0)
                first_in_session = False
                # continue to consider next op in same session
            else:
                # compute expected overtime if we start this op now (use Expected)
                expected_end = current_time + op["Expected"]
                expected_overtime = max(0.0, expected_end - session_duration)
                if expected_overtime > 30.0:
                    # postpone this op (and therefore everything after it stays in queue)
                    break
                else:
                    # start it
                    actual = op["Actual"]
                    current_time += actual
                    session_actual_sum += actual
                    total_actual_minutes += actual

                    op_assigned = op.copy()
                    op_assigned.update({"Assigned_Session": session_index, "Assigned_Week": int(assigned_week)})
                    scheduled_ops.append(op_assigned)
                    queue.pop(0)
                    # continue to next queued op

        # end of session: compute overtime for this session based on actual total
        overtime_this_session = max(0.0, session_actual_sum - session_duration)
        total_overtime += overtime_this_session

    # After all scheduled, compute utilization across all sessions
    sessions = session_index
    utilization = (total_actual_minutes / (sessions * session_duration)) * 100 if sessions > 0 else 0.0

    # compute late operations: assigned week > due week
    late_ops = 0
    for o in scheduled_ops:
        if not pd.isnull(o.get("Due_Week")) and o.get("Assigned_Week") > o.get("Due_Week"):
            late_ops += 1

    return {
        "Sessions": sessions,
        "Utilization (%)": round(utilization, 2),
        "Overtime (min)": round(total_overtime, 2),
        "Late Operations": int(late_ops),
        "Total_Actual_Minutes": total_actual_minutes
    }

# Run simulation for each group and collect results
group_summary_e = []
for group, group_df in df.groupby("Surgery Group"):
    sim = simulate_postponement_for_group(group_df, session_duration=session_duration)
    sim_row = {"Group": group}
    sim_row.update(sim)
    group_summary_e.append(sim_row)

print("\n=== PART E: OR Scheduling Metrics (WITH postponement rule) ===")
total_sessions_e = 0
total_minutes_e = 0
total_overtime_e = 0
total_late_ops_e = 0

for summary in group_summary_e:
    print(f"Group: {summary['Group']}")
    print(f"  Sessions: {summary['Sessions']}")
    print(f"  Utilization (%): {summary['Utilization (%)']}")
    print(f"  Overtime (min): {summary['Overtime (min)']}")
    print(f"  Late Operations: {summary['Late Operations']}")
    print("-" * 60)
    total_sessions_e += summary['Sessions']
    total_minutes_e += summary['Sessions'] * session_duration
    total_overtime_e += summary['Overtime (min)']
    total_late_ops_e += summary['Late Operations']

print("=== OVERALL METRICS (E) ===")
print(f"  Total Sessions: {total_sessions_e}")
print(f"  Total Scheduled Time (min): {total_minutes_e}")
print(f"  Total Overtime (min): {round(total_overtime_e, 2)}")
print(f"  Total Late Operations: {total_late_ops_e}")

# --- Comparison between D and E ---
df_d = pd.DataFrame(group_summary_d).set_index("Group")
df_e = pd.DataFrame(group_summary_e).set_index("Group")

# align indices (ensure same groups present)
all_groups = sorted(set(df_d.index).union(df_e.index))
df_compare = pd.DataFrame(index=all_groups)
for col in ["Sessions", "Utilization (%)", "Overtime (min)", "Late Operations", "Total_Actual_Minutes"]:
    df_compare[f"D_{col}"] = df_d[col] if col in df_d.columns else 0
    df_compare[f"E_{col}"] = df_e[col] if col in df_e.columns else 0

# compute differences E - D
df_compare["Delta_Sessions"] = df_compare["E_Sessions"] - df_compare["D_Sessions"]
df_compare["Delta_Utilization_pct"] = df_compare["E_Utilization (%)"] - df_compare["D_Utilization (%)"]
df_compare["Delta_Overtime_min"] = df_compare["E_Overtime (min)"] - df_compare["D_Overtime (min)"]
df_compare["Delta_LateOps"] = df_compare["E_Late Operations"] - df_compare["D_Late Operations"]

pd.set_option("display.float_format", "{:,.2f}".format)
print("\n=== PER-GROUP COMPARISON (E minus D) ===")
print(df_compare[[
    "D_Sessions","E_Sessions","Delta_Sessions",
    "D_Utilization (%)","E_Utilization (%)","Delta_Utilization_pct",
    "D_Overtime (min)","E_Overtime (min)","Delta_Overtime_min",
    "D_Late Operations","E_Late Operations","Delta_LateOps"
]])

# Print high-level totals comparison
print("\n=== TOTALS COMPARISON ===")
print(f"Total Sessions: D={total_sessions_d} -> E={total_sessions_e} (delta {total_sessions_e - total_sessions_d})")
print(f"Total Scheduled Time (min): D={total_minutes_d} -> E={total_minutes_e} (delta {total_minutes_e - total_minutes_d})")
print(f"Total Overtime (min): D={round(total_overtime_d,2)} -> E={round(total_overtime_e,2)} (delta {round(total_overtime_e - total_overtime_d,2)})")
print(f"Total Late Operations: D={total_late_ops_d} -> E={total_late_ops_e} (delta {total_late_ops_e - total_late_ops_d})")
