#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 14:13:35 2025

@author: ariannaperini
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
import matplotlib.pyplot as plt
import math


XLS_PATH = "/Users/ariannaperini/Downloads/Data for Assignment 6.xlsx" #change this if needed
HOURS_WINDOW = list(range(8, 21))  # 08:00 through 20:00 -> slots 08-09 ... 20-21
HOUR_LABELS = [f"{h:02d}:00-{h+1:02d}:00" for h in HOURS_WINDOW]
MIN_NURSES = 1  # minimum staffing between 08:00-21:00
DEFAULT_COST_4H = 120
DEFAULT_COST_8H = 220
PATIENT_NURSE_HOUR_PER_HOUR = 0.25  # 15 minutes per patient per hour


# Utilities

def parse_admit_dt(or_date, ready_for_ward):
    """Combine OR date and ready-for-ward time into a pd.Timestamp.
       Return pd.NaT if not parseable."""
    if pd.isna(or_date) or pd.isna(ready_for_ward):
        return pd.NaT
    # ready_for_ward can be timestamp, time, or string
    try:
        if isinstance(ready_for_ward, pd.Timestamp):
            t = ready_for_ward.time()
        else:
            t = pd.to_datetime(str(ready_for_ward)).time()
    except Exception:
        return pd.NaT
    try:
        return pd.Timestamp(datetime.combine(or_date.date(), t))
    except Exception:
        # fallback: if 'or_date' is string-like
        try:
            datepart = pd.to_datetime(or_date).normalize()
            return datepart + pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
        except Exception:
            return pd.NaT

# -------------------------
# 1) Load data and demand analysis
# -------------------------
def load_and_filter_daycare(xls_path):
    xls = pd.ExcelFile(xls_path)
    patients = pd.read_excel(xls, sheet_name='patients')
    nurse_costs = pd.read_excel(xls, sheet_name='nurse costs').fillna('')
    # Identify daycare rows
    if 'LOS' not in patients.columns or 'group' not in patients.columns:
        raise ValueError("patients sheet must contain columns 'LOS' and 'group'.")
    daycare_mask = (patients['LOS'] == 0) | (patients['group'].astype(str).str.lower().str.contains('daycare', na=False))
    daycare = patients[daycare_mask].copy()
    # Parse admit datetime
    # 'OR date' is assumed to be a date-like column, 'ready for ward' contains time or datetime
    if 'OR date' not in daycare.columns:
        raise ValueError("patients sheet must contain column 'OR date'.")
    # Some files may label the time column differently — we try common names
    ready_col_candidates = ['ready for ward', 'ready_for_ward', 'ready_for_ward_time', 'ready']
    ready_col = None
    for c in ready_col_candidates:
        if c in daycare.columns:
            ready_col = c
            break
    # If not found, try to find any column with 'ready' in name
    if ready_col is None:
        for c in daycare.columns:
            if 'ready' in str(c).lower():
                ready_col = c
                break
    if ready_col is None:
        raise ValueError("Could not find 'ready for ward' column in patients sheet (tried several candidates).")
    # compute admit_dt
    daycare['admit_dt'] = daycare.apply(lambda r: parse_admit_dt(r['OR date'], r[ready_col]), axis=1)
    daycare = daycare.dropna(subset=['admit_dt']).copy()
    # enforce assumption: empty ward at 08:00 daily -> remove any record admitted before 08:00 on that day
    def admitted_before_0800(row):
        dt = row['admit_dt']
        return (dt.time() < time(8, 0))
    daycare['admitted_before_08'] = daycare.apply(admitted_before_0800, axis=1)
    # Keep only records admitted at or after 08:00 on that date
    daycare = daycare[~daycare['admitted_before_08']].copy()
    # compute end datetime (4-hour stay)
    daycare['end_dt'] = daycare['admit_dt'] + pd.Timedelta(hours=4)
    daycare['date'] = daycare['admit_dt'].dt.date
    daycare['weekday'] = daycare['admit_dt'].dt.weekday  # Monday=0
    return daycare, nurse_costs

def compute_average_pattern(daycare):
    # consider working days only (Mon-Fri)
    working = daycare[daycare['weekday'] < 5].copy()
    if working.empty:
        raise ValueError("No daycare records found for working days after applying 08:00-empty assumption.")
    dates = sorted(working['date'].unique())
    per_day_counts = []
    for d in dates:
        df_day = working[working['date'] == d]
        counts = []
        for h in HOURS_WINDOW:
            slot_start = pd.Timestamp(datetime.combine(d, time(h, 0)))
            slot_end = slot_start + pd.Timedelta(hours=1)
            # patient present if interval intersects
            cnt = int(((df_day['admit_dt'] < slot_end) & (df_day['end_dt'] > slot_start)).sum())
            counts.append(cnt)
        per_day_counts.append(counts)
    df_counts = pd.DataFrame(per_day_counts, columns=HOUR_LABELS, index=pd.to_datetime(dates))
    avg_patients = df_counts.mean(axis=0)
    avg_required_nurse_hours = avg_patients * PATIENT_NURSE_HOUR_PER_HOUR
    # For guaranteeing average required care is always met, we will translate required nurse-hours to an integer number of nurses per hour:
    # required_nurses_hourly = ceil(avg_required_nurse_hours), but also respect minimum staffing of 1.
    required_nurses = np.ceil(avg_required_nurse_hours).astype(int)
    required_nurses = np.maximum(required_nurses, MIN_NURSES)
    return df_counts, avg_patients, avg_required_nurse_hours, required_nurses

# -------------------------
# 2) Shift optimization model (mathematical formulation and solver attempts)
# -------------------------
def parse_daycare_shift_costs(nurse_costs):
    """Try to parse the daycare 4h and 8h shift costs from nurse_costs sheet, otherwise return defaults."""
    cost_4h = DEFAULT_COST_4H
    cost_8h = DEFAULT_COST_8H
    # naive parsing: find row containing 'daycare' then find lines near it containing '4h shift' and '8h shift'
    first_col = nurse_costs.iloc[:, 0].astype(str).str.lower()
    mask = first_col.str.contains('daycare', na=False)
    if mask.any():
        idx = mask[mask].index[0]
        # look at following few rows
        for i in range(idx, min(idx + 8, len(nurse_costs))):
            rowtext = ' '.join(nurse_costs.iloc[i].astype(str).values).lower()
            if '4h' in rowtext and 'shift' in rowtext:
                # try to find numeric in row
                for c in nurse_costs.columns:
                    try:
                        val = float(nurse_costs.at[i, c])
                        cost_4h = int(val)
                        break
                    except Exception:
                        continue
            if '8h' in rowtext and 'shift' in rowtext:
                for c in nurse_costs.columns:
                    try:
                        val = float(nurse_costs.at[i, c])
                        cost_8h = int(val)
                        break
                    except Exception:
                        continue
    return cost_4h, cost_8h

def build_shift_candidates():
    # shift types and lengths
    shift_lengths = {'4h': 4, '8h': 8}
    candidates = []
    coverage = {}
    for typ, length in shift_lengths.items():
        for s in range(0, 24):
            # coverage over HOURS_WINDOW
            cover = {h: (1 if (s <= h < s + length) else 0) for h in HOURS_WINDOW}
            if sum(cover.values()) > 0:
                candidates.append((typ, s))
                coverage[(typ, s)] = cover
    return candidates, coverage

def print_mathematical_model_description(HOUR_LABELS, required_nurse_hours, cost_4h, cost_8h):
    print("\nMathematical formulation (mixed-integer):")
    print("Decision variables:")
    print("  x_{t,s} = integer # of shifts of type t (4h or 8h) starting at hour s (s in 0..23).")
    print("Parameters:")
    print("  coverage_{t,s,h} = 1 if shift (t,s) covers hour-slot h, else 0.")
    print("  cost_t = cost of a shift of type t (4h -> €{}, 8h -> €{})".format(cost_4h, cost_8h))
    print("  required_nurses[h] = integer number of nurses required in hour h (we use ceil of average required nurse-hours, and minimum 1).")
    print("Objective:")
    print("  minimize sum_{t,s} cost_t * x_{t,s}")
    print("Constraints (for each hour h in our window):")
    print("  sum_{t,s} coverage_{t,s,h} * x_{t,s} >= required_nurses[h]")
    print("and x_{t,s} are nonnegative integers.")
    print("\nNote: required_nurses[h] is computed from average required nurse-hours as ceil(avg_nurse_hours) to guarantee integer nurses.\n")

def solve_with_pulp(candidates, coverage, required_nurses, cost_4h, cost_8h):
    try:
        import pulp
    except Exception:
        return None
    prob = pulp.LpProblem("DaycareShiftMinCost", pulp.LpMinimize)
    x = {k: pulp.LpVariable(f"x_{k[0]}_{k[1]}", lowBound=0, cat='Integer') for k in candidates}
    # objective
    prob += pulp.lpSum([ x[k] * (cost_4h if k[0] == '4h' else cost_8h) for k in candidates ])
    # constraints per hour
    for h in HOURS_WINDOW:
        prob += pulp.lpSum([ coverage[k][h] * x[k] for k in candidates ]) >= int(required_nurses[HOUR_LABELS.index(f"{h:02d}:00-{h+1:02d}:00")])
    # solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    # extract solution
    sol = {k: int(pulp.value(x[k])) for k in candidates if pulp.value(x[k]) is not None and pulp.value(x[k]) > 0.5}
    return sol

def solve_with_ortools(candidates, coverage, required_nurses, cost_4h, cost_8h):
    try:
        from ortools.sat.python import cp_model
    except Exception:
        return None
    model = cp_model.CpModel()
    vars_x = {}
    max_shifts_per_type = 20  # reasonable upper bound
    for k in candidates:
        vars_x[k] = model.NewIntVar(0, max_shifts_per_type, f"x_{k[0]}_{k[1]}")
    # constraints
    for idx, h in enumerate(HOURS_WINDOW):
        req = int(required_nurses[idx])
        model.Add(sum(coverage[k][h] * vars_x[k] for k in candidates) >= req)
    # objective: linear minimize costs
    objective_terms = []
    for k in candidates:
        cost = cost_4h if k[0] == '4h' else cost_8h
        objective_terms.append(cost * vars_x[k])
    model.Minimize(sum(objective_terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    sol = {}
    for k in candidates:
        val = int(solver.Value(vars_x[k]))
        if val > 0:
            sol[k] = val
    return sol

def greedy_heuristic(candidates, coverage, required_nurses, cost_4h, cost_8h):
    # deterministic greedy: choose shift with minimal cost per newly-covered-needed nurse (units of nurse count)
    remaining = {h: int(required_nurses[HOUR_LABELS.index(f"{h:02d}:00-{h+1:02d}:00")]) for h in HOURS_WINDOW}
    selected = {}
    iter_count = 0
    while any(remaining[h] > 0 for h in HOURS_WINDOW):
        iter_count += 1
        best_k = None
        best_metric = float('inf')
        for k in candidates:
            newly_covered = sum(1 for h in HOURS_WINDOW if coverage[k][h] == 1 and remaining[h] > 0)
            if newly_covered == 0:
                continue
            cost = cost_4h if k[0] == '4h' else cost_8h
            metric = cost / newly_covered
            if metric < best_metric:
                best_metric = metric
                best_k = k
        if best_k is None:
            # can't cover remaining (shouldn't happen)
            break
        selected[best_k] = selected.get(best_k, 0) + 1
        for h in HOURS_WINDOW:
            if coverage[best_k][h] == 1 and remaining[h] > 0:
                remaining[h] -= 1
        if iter_count > 1000:
            break
    return selected

# -------------------------
# 3) Resource utilization & reporting
# -------------------------
def compute_staffing_from_solution(solution, coverage):
    staffing = {h: 0 for h in HOURS_WINDOW}
    for k, cnt in solution.items():
        for h in HOURS_WINDOW:
            staffing[h] += coverage[k][h] * cnt
    staffed_series = pd.Series([staffing[h] for h in HOURS_WINDOW], index=HOUR_LABELS)
    return staffed_series

def compute_utilization_table(avg_required_nurse_hours, staffed_series):
    util = []
    for i, hlabel in enumerate(HOUR_LABELS):
        required = float(avg_required_nurse_hours.iloc[i])
        staffed = float(staffed_series.iloc[i])
        if staffed <= 0:
            util.append(0.0)
        else:
            u = required / staffed
            if u > 1.0:
                # if requirement > staff, utilization > 1 -> indicates understaffing, but we'll report >1
                util.append(u)
            else:
                util.append(u)
    util_series = pd.Series(util, index=HOUR_LABELS)
    table = pd.DataFrame({
        'hour_slot': HOUR_LABELS,
        'avg_required_nurse_hours': avg_required_nurse_hours.values,
        'staffed_nurses': staffed_series.values,
        'utilization_fraction': util_series.values
    })
    return table

# -------------------------
# Main orchestration
# -------------------------
def main():
    print("Loading data and filtering daycare patients (LOS==0 or group contains 'daycare') ...")
    daycare, nurse_costs = load_and_filter_daycare(XLS_PATH)
    print(f"Total daycare records after filtering and enforcing empty-at-08: {len(daycare)}")
    # Demand analysis
    df_counts, avg_patients, avg_required_nurse_hours, required_nurses = compute_average_pattern(daycare)
    print("\nTask 1: DEMAND ANALYSIS (fixed pattern per working day, assuming empty at 08:00 daily)\n")
    avg_table = pd.DataFrame({
        'hour_slot': HOUR_LABELS,
        'avg_patients': avg_patients.values,
        'avg_required_nurse_hours': avg_required_nurse_hours.values
    })
    # Display the table
    pd.set_option('display.precision', 4)
    print("Average hourly care requirements (08:00-21:00) -- nurse-hours are per hour:")
    print(avg_table.to_string(index=False))

    # Provide required nurses (ceil) used for optimization and explain
    print("\nWe convert average required nurse-hours to integer required nurses by taking ceil(avg_required_nurse_hours).")
    print("This guarantees that the scheduled integer nurses supply at least the average required nurse-hours each hour.")
    print("We also enforce a minimum of 1 nurse between 08:00 and 21:00 as required.\n")
    req_df = pd.DataFrame({'hour_slot': HOUR_LABELS, 'avg_required_nurse_hours': avg_required_nurse_hours.values, 'required_nurses_ceiled': required_nurses.values})
    print(req_df.to_string(index=False))

    # Shift costs
    cost_4h, cost_8h = parse_daycare_shift_costs(nurse_costs)
    print(f"\nDaycare shift costs used in optimization: 4h = €{cost_4h}, 8h = €{cost_8h}")

    # Build candidates and coverage
    candidates, coverage = build_shift_candidates()

    # Print mathematical model
    print_mathematical_model_description(HOUR_LABELS, avg_required_nurse_hours, cost_4h, cost_8h)

    # Try solvers in order: pulp -> ortools -> greedy
    solution = None
    print("Attempting exact integer optimization using pulp (if installed)...")
    solution = solve_with_pulp(candidates, coverage, required_nurses, cost_4h, cost_8h)
    if solution is not None and len(solution) > 0:
        print("Exact MILP (pulp) produced a solution.")
    else:
        print("pulp not available or no solution returned. Trying ortools CP-SAT (if installed)...")
        solution = solve_with_ortools(candidates, coverage, required_nurses, cost_4h, cost_8h)
        if solution is not None and len(solution) > 0:
            print("ortools CP-SAT produced a solution (may be optimal or feasible depending on time limit).")
        else:
            print("Exact solvers unavailable or failed. Falling back to greedy heuristic.")
            solution = greedy_heuristic(candidates, coverage, required_nurses, cost_4h, cost_8h)
            print("Greedy heuristic solution produced.")

    # Present shift schedule (deliverable)
    shift_rows = []
    for (typ, s), cnt in sorted(solution.items(), key=lambda x: (x[0][0], x[0][1])):
        length = 4 if typ == '4h' else 8
        shift_rows.append({'type': typ, 'start_hour': s, 'end_hour': s + length, 'count': cnt, 'cost_each': (cost_4h if typ == '4h' else cost_8h), 'total_cost': cnt * (cost_4h if typ == '4h' else cost_8h)})
    shift_df = pd.DataFrame(shift_rows)
    if shift_df.empty:
        print("\nWARNING: No shifts selected by the solver/heuristic (unexpected).")
    else:
        print("\nTask 2: SHIFT OPTIMIZATION - Selected shift mix (4h and 8h allocations):")
        print(shift_df.to_string(index=False))

    # Compute staffing by hour and utilization
    staffed_series = compute_staffing_from_solution(solution, coverage)
    util_table = compute_utilization_table(avg_required_nurse_hours, staffed_series)
    total_daily_cost = shift_df['total_cost'].sum() if not shift_df.empty else 0.0

    # Task 3 deliverables
    print("\nTask 3: RESOURCE UTILISATION ANALYSIS\n")
    print(f"Daily cost of selected shifts: €{total_daily_cost:.2f}\n")
    print("Hourly utilization and staffing (utilization >1 indicates understaffing relative to averaged required nurse-hours):")
    print(util_table.to_string(index=False))

    # Plot: Demand (nurse-hours) vs Staffing (nurses available)
    plt.figure(figsize=(11,4))
    plt.plot(HOUR_LABELS, avg_required_nurse_hours.values, marker='o', label='Avg required nurse-hours (demand)')
    plt.plot(HOUR_LABELS, staffed_series.values, marker='s', label='Staffed nurses (supply in nurse-hours)')
    plt.xticks(rotation=45)
    plt.ylabel('Nurse-hours / Nurses')
    plt.title('Daycare - Demand vs Staffing Coverage (working-day average, empty at 08:00 assumption)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Final summary printed for clarity
    print("\nDELIVERABLES SUMMARY:")
    print("- Table: average hourly care requirements -> printed above as 'Average hourly care requirements'.")
    print("- Shift schedule: printed above as selected shift mix (type, start, end, count, cost).")
    print("- Utilization metrics & cost summary: printed above (util_table and total cost).")

if __name__ == "__main__":
    main()
