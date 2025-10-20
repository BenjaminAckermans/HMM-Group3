#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Assignment 7 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings('ignore')
#!pip install pulp
# solver
try:
    import pulp
except Exception:
    raise ImportError("Please install 'pulp' (pip install pulp) to run LP/MILP parts.")

# file path
XLS_PATH = r"Data for Assignment 7.xlsx" #change this for your laptop

# constants
WEEKS_PER_YEAR = 52
WORK_DAYS_PER_WEEK = 5
HOURS_PER_DAY = 8
HOURS_PER_WEEK = WORK_DAYS_PER_WEEK * HOURS_PER_DAY  # 40
HOURS_PER_YEAR = WEEKS_PER_YEAR * HOURS_PER_WEEK  # 2080
WARD_VISIT_MIN = 15
ADMIN_MIN_PER_OP = 5
ADMIN_MIN_PER_SURG = 10
EDU_HOURS_PER_SPEC = 80
CONF_WEEKS_PER_SPEC = 2
CONF_EXTRA_HOURS_PER_SPEC = 24

# helper: find sheet by keywords
def find_sheet_name(xls, keywords):
    keys = [k.lower() for k in keywords]
    for s in xls.sheet_names:
        ls = s.lower()
        if all(k in ls for k in keys):
            return s
    # fallback: try any sheet that contains any keyword
    for s in xls.sheet_names:
        ls = s.lower()
        for k in keys:
            if k in ls:
                return s
    return None

# read excel
xls = pd.ExcelFile(XLS_PATH)
# map sheets according to your descriptions
sheet_pat = find_sheet_name(xls, ["patient"]) or find_sheet_name(xls, ["patients"]) or xls.sheet_names[0]
sheet_surg_q = find_sheet_name(xls, ["surgery", "queue"]) or find_sheet_name(xls, ["surgery queue"])
sheet_in_ward = find_sheet_name(xls, ["in ward"]) or find_sheet_name(xls, ["ward"])
sheet_surglos = find_sheet_name(xls, ["surgerylos", "surgerylos"]) or find_sheet_name(xls, ["surgerylos", "surgerylos"])
sheet_spec = xls.sheet_names[-1]  # as you said last one

df_pat = xls.parse(sheet_pat)
df_surg_q = xls.parse(sheet_surg_q)
df_in_ward = xls.parse(sheet_in_ward)
df_surglos = xls.parse(sheet_surglos)
df_spec_raw = xls.parse(sheet_spec, header=None)

# normalize column names (lowercase, strip)
def norm_cols(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

df_pat = norm_cols(df_pat)
df_surg_q = norm_cols(df_surg_q)
df_in_ward = norm_cols(df_in_ward)
df_surglos = norm_cols(df_surglos)

# expected column names (as you provided)
# patient data columns: 'nr', 'arrival (h)', 'timeod (min)', 'subspec', 'subsubsp'
# surgery queue: 'nr','subspec','subsubsp','due-date op (h)'
# in ward: 'nr','subspec','remaining LOS'
# surgeryLOS: 'subspec','subsub','#patients','surgery','surgery time(min)','LOS(days)'

# tidy patient dataframe (drop rows with missing minimal info)
pat_cols = [c for c in df_pat.columns]
# ensure columns accessible
# rename common variations
df_pat = df_pat.rename(columns={
    col: col.replace(' ', '').replace('(', '').replace(')', '').replace('-', '').lower()
    for col in df_pat.columns
})
df_surg_q = df_surg_q.rename(columns={
    col: col.replace(' ', '').replace('(', '').replace(')', '').replace('-', '').lower()
    for col in df_surg_q.columns
})
df_in_ward = df_in_ward.rename(columns={
    col: col.replace(' ', '').replace('(', '').replace(')', '').replace('-', '').lower()
    for col in df_in_ward.columns
})
df_surglos = df_surglos.rename(columns={
    col: col.replace(' ', '').replace('(', '').replace(')', '').replace('-', '').lower()
    for col in df_surglos.columns
})

# convenience names
# try to locate key colnames flexibly
def find_col(df, candidates):
    cols = df.columns
    for c in candidates:
        for col in cols:
            if c in col:
                return col
    return None

pat_arr_col = find_col(df_pat, ['arrival', 'arrivalh', 'arrival(h)', 'arrival (h)'])
pat_time_col = find_col(df_pat, ['timeod', 'time', 'consult', 'timeod(min)', 'timeodmin', 'timeod (min)'])
pat_sub_col = find_col(df_pat, ['subspec','subspecial'])
pat_subsub_col = find_col(df_pat, ['subsub','subsubsp'])

surgq_due_col = find_col(df_surg_q, ['due','due-date','duedate','due-date op','due-date op (h)'])
surgq_sub_col = find_col(df_surg_q, ['subspec','subspecial'])
surgq_subsub_col = find_col(df_surg_q, ['subsub','subsubsp'])

ward_sub_col = find_col(df_in_ward, ['subspec','subspecial'])
ward_los_col = find_col(df_in_ward, ['remaining','los','remaininglos'])

surglos_subspec_col = find_col(df_surglos, ['subspec'])
surglos_subsub_col = find_col(df_surglos, ['subsub'])
surglos_time_col = find_col(df_surglos, ['surgerytime','surgerytime(min)','surgerytime(min)','surgerytime(min)'.lower()]) or find_col(df_surglos, ['surgerytime','surgerytime(min)'])
surglos_time_col = surglos_time_col or find_col(df_surglos, ['surgerytime','surgerytime(min)','surgerytime(min)']) or find_col(df_surglos, ['surgerytime','surgery time(min)','surgery_time_min','surgerytime(min)'])
surglos_time_col = surglos_time_col or find_col(df_surglos, ['surgery','surgery time','operation time','surgerytime(min)'])
surglos_los_col = find_col(df_surglos, ['los','los(days)','losdays'])

# if anything missing, try heuristic names
if pat_arr_col is None: pat_arr_col = find_col(df_pat, ['arrival'])
if pat_time_col is None: pat_time_col = find_col(df_pat, ['time','min'])
if pat_sub_col is None: pat_sub_col = find_col(df_pat, ['subspec','subspecial'])
if pat_subsub_col is None: pat_subsub_col = find_col(df_pat, ['subsub'])

if surgq_due_col is None: surgq_due_col = find_col(df_surg_q, ['due'])
if surgq_sub_col is None: surgq_sub_col = find_col(df_surg_q, ['subspec'])
if surgq_subsub_col is None: surgq_subsub_col = find_col(df_surg_q, ['subsub'])

if ward_los_col is None: ward_los_col = find_col(df_in_ward, ['remaining','los'])
if ward_sub_col is None: ward_sub_col = find_col(df_in_ward, ['subspec'])

# clean numeric columns
def to_numeric_safe(s):
    return pd.to_numeric(s, errors='coerce')

if pat_arr_col: df_pat[pat_arr_col] = to_numeric_safe(df_pat[pat_arr_col])
if pat_time_col: df_pat[pat_time_col] = to_numeric_safe(df_pat[pat_time_col])
if pat_sub_col: df_pat[pat_sub_col] = to_numeric_safe(df_pat[pat_sub_col])
if pat_subsub_col: df_pat[pat_subsub_col] = to_numeric_safe(df_pat[pat_subsub_col])

if surgq_due_col: df_surg_q[surgq_due_col] = to_numeric_safe(df_surg_q[surgq_due_col])
if surgq_sub_col: df_surg_q[surgq_sub_col] = to_numeric_safe(df_surg_q[surgq_sub_col])
if surgq_subsub_col: df_surg_q[surgq_subsub_col] = to_numeric_safe(df_surg_q[surgq_subsub_col])

if ward_los_col: df_in_ward[ward_los_col] = to_numeric_safe(df_in_ward[ward_los_col])
if ward_sub_col: df_in_ward[ward_sub_col] = to_numeric_safe(df_in_ward[ward_sub_col])

# parse specialist sheet (matrix + FTE + preferences)
spec_table = df_spec_raw.copy()
spec_table = spec_table.replace('', np.nan)

# find row index where first column contains 'skill' (case-ins)
first_col = spec_table.iloc[:,0].astype(str).str.strip().str.lower()
r_skill = None
for idx, v in first_col.items():
    if isinstance(v, str) and 'skill' in v:
        r_skill = idx
        break
if r_skill is None:
    # fallback to row 0
    r_skill = 0

# header columns (specialist ids) assumed to be columns from col 1 onward in row r_skill
spec_ids = spec_table.iloc[r_skill, 1:].dropna().astype(str).tolist()
n_specs = len(spec_ids)
# find 'matrix' row: the next row after 'skill' with 'matrix' label in first col
r_matrix = None
for idx in range(r_skill+1, r_skill+12):
    if idx in spec_table.index:
        v = str(spec_table.iloc[idx,0]).strip().lower()
        if 'matrix' in v or v=='1' or v=='0':
            r_matrix = idx
            break
# more robust: find block of numeric matrix rows — assume subspecialisms 1..10 rows follow
# locate FTE row index by searching for 'fte' or 'f t e' in first col lower
r_fte = None
for idx, v in first_col.items():
    if isinstance(v, str) and ('fte' in v.lower()):
        r_fte = idx
        break

# assume matrix rows are between r_matrix (or r_skill+1) and r_fte-1
start_matrix = r_skill+1
end_matrix = r_fte-1 if r_fte is not None else r_skill+10
# extract matrix as numeric values from columns 1..n_specs
matrix_block = spec_table.iloc[start_matrix:end_matrix+1, 1:1+n_specs]
# try to coerce to numbers (0/1)
matrix_block = matrix_block.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
# ensure rows correspond to subspecialisms 1..matrix_block.shape[0]
subspec_list = list(range(1, matrix_block.shape[0]+1))
matrix_block.index = subspec_list
# build S matrix: spec_id -> dict of subspecialism->1/0
S = {}
for j, spec_id in enumerate(spec_ids):
    col = matrix_block.iloc[:, j] if j < matrix_block.shape[1] else pd.Series([0]*matrix_block.shape[0])
    S[spec_id] = {sub: int(col.iloc[i]) for i, sub in enumerate(matrix_block.index)}

# FTE row values
fte_vals = None
if r_fte is not None:
    fte_vals = spec_table.iloc[r_fte, 1:1+n_specs].apply(pd.to_numeric, errors='coerce').fillna(1.0).astype(float).tolist()
else:
    fte_vals = [1.0]*n_specs
specs = []
for i, spec_id in enumerate(spec_ids):
    specs.append({
        "spec_id": spec_id,
        "fte": float(fte_vals[i]) if i < len(fte_vals) else 1.0
    })
df_specs = pd.DataFrame(specs)

# preferences: find rows labeled 'preference','points','holidays','conferences'
row_labels = first_col
def find_row_label_matches(labels):
    mapping = {}
    for label in labels:
        idxs = [idx for idx, v in row_labels.items() if isinstance(v, str) and label in v.lower()]
        mapping[label] = idxs[0] if idxs else None
    return mapping

labels_map = find_row_label_matches(['preference','points','holidays','conferences'])
pref_row = labels_map.get('preference')
points_row = labels_map.get('points')
holidays_row = labels_map.get('holidays')
confs_row = labels_map.get('conferences')

prefs = {}
points = {}
holidays = {}
confs = {}

# simple parse: take the single row values for preference & points
if pref_row is not None:
    vals = spec_table.iloc[pref_row, 1:1+n_specs].tolist()
    for i, sid in enumerate(spec_ids):
        prefs[sid] = [] if pd.isna(vals[i]) else [int(vals[i])] if np.isfinite(vals[i]) else []
if points_row is not None:
    vals = spec_table.iloc[points_row, 1:1+n_specs].tolist()
    for i, sid in enumerate(spec_ids):
        points[sid] = float(vals[i]) if (i < len(vals) and not pd.isna(vals[i])) else 0.0

# holidays and conferences might be multi-row blocks below their header
def collect_block(start_row):
    if start_row is None:
        return {sid: [] for sid in spec_ids}
    out = {sid: [] for sid in spec_ids}
    r = start_row
    # collect until next blank-first-col or until 20 rows
    for rr in range(r, r+20):
        if rr not in spec_table.index:
            break
        first = str(spec_table.iloc[rr,0]).strip().lower()
        if first in ['', 'nan', 'preference', 'points', 'conferences', 'holidays', 'skill', 'matrix', 'fte']:
            # if this is the first row and label equals header, skip; but if blank -> stop
            if rr==r:
                continue
            else:
                break
        rowvals = spec_table.iloc[rr, 1:1+n_specs].tolist()
        for i, sid in enumerate(spec_ids):
            v = rowvals[i] if i < len(rowvals) else np.nan
            if not (v is None or (isinstance(v, float) and np.isnan(v))):
                try:
                    out[sid].append(int(v))
                except Exception:
                    # ignore non-integers
                    pass
    return out

hol_block = collect_block(holidays_row)
conf_block = collect_block(confs_row)
# fallback if empty: give default holiday weeks (1..8) and conferences (1..2)
for sid in spec_ids:
    holidays[sid] = hol_block.get(sid, []) if sum(len(v) for v in hol_block.values())>0 else list(range(1,9))
    confs[sid] = conf_block.get(sid, []) if sum(len(v) for v in conf_block.values())>0 else [1,2]
    prefs.setdefault(sid, [])
    points.setdefault(sid, 0.0)

# ---- Part A: compute required hours ----
# 1) Ward visits (remaining LOS): (remaining_los + 1) visits * 15 minutes
if ward_los_col is None:
    df_in_ward = df_in_ward.copy()
    df_in_ward['remaininglos'] = df_in_ward.iloc[:, -1]
    ward_los_col = 'remaininglos'
ward_total_minutes = ((df_in_ward[ward_los_col].fillna(0) + 1) * WARD_VISIT_MIN).sum()
ward_hours = ward_total_minutes / 60.0

# ward by subspecialism
ward_sub_col = ward_sub_col or df_in_ward.columns[-2]
ward_by_sub = df_in_ward.groupby(ward_sub_col)[ward_los_col].apply(lambda s: (((s+1)*WARD_VISIT_MIN).sum()/60.0)).to_dict()

# 2) OPD: consult durations in df_pat (timeod min)
opd_df = df_pat.copy()
time_col = pat_time_col or df_pat.columns[0]
opd_total_hours = opd_df[time_col].fillna(0).sum() / 60.0
opd_by_sub = opd_df.groupby(pat_sub_col)[time_col].sum().apply(lambda m: m/60.0).to_dict()

# 3) OR: from surgery queue and from surgeryLOS aggregated
# use df_surg_q and df_surglos mapping for typical surgery durations
surg_q = df_surg_q.copy()
# try to infer oper_minutes for surgery queue using surgeryLOS mapping
# build mapping from surglos: (subspec, subsub) -> surgery time (min)
surglos_map = {}
if surglos_subspec_col and surglos_subsub_col and surglos_time_col:
    for _, r in df_surglos.iterrows():
        k = (int(r[surglos_subspec_col]) if not pd.isna(r[surglos_subspec_col]) else None,
             int(r[surglos_subsub_col]) if not pd.isna(r[surglos_subsub_col]) else None)
        t = pd.to_numeric(r.get(surglos_time_col), errors='coerce')
        if not pd.isna(t):
            surglos_map[k] = float(t)

# try to attach oper minutes to queue
oper_minutes_list = []
for _, r in surg_q.iterrows():
    s = r.get(surgq_sub_col)
    ss = r.get(surgq_subsub_col)
    key = (int(s) if not pd.isna(s) else None, int(ss) if not pd.isna(ss) else None)
    t = surglos_map.get(key, np.nan)
    oper_minutes_list.append(t)
surg_q['oper_minutes'] = oper_minutes_list
# if some missing, fill with median
if surg_q['oper_minutes'].isna().any():
    med = np.nanmedian(list(surglos_map.values())) if len(surglos_map)>0 else 90.0
    surg_q['oper_minutes'] = surg_q['oper_minutes'].fillna(med)
or_total_hours = surg_q['oper_minutes'].sum() / 60.0
or_by_subsub = surg_q.groupby(surgq_subsub_col)['oper_minutes'].sum().apply(lambda m: m/60.0).to_dict()

# 4) Admin
admin_minutes = len(opd_df) * ADMIN_MIN_PER_OP + len(surg_q) * ADMIN_MIN_PER_SURG
admin_hours = admin_minutes / 60.0

# 5) Education & conferences
education_hours_total = EDU_HOURS_PER_SPEC * len(spec_ids)
conf_hours_total = (CONF_WEEKS_PER_SPEC * HOURS_PER_WEEK + CONF_EXTRA_HOURS_PER_SPEC) * len(spec_ids)

# print Part A summary
tasks_summary = {
    "WARD_hours": ward_hours,
    "OPD_hours": opd_total_hours,
    "OR_hours": or_total_hours,
    "ADMIN_hours": admin_hours,
    "EDU_hours": education_hours_total,
    "CONF_hours": conf_hours_total
}
print("PART A — required annual hours per task:")
for k,v in tasks_summary.items():
    print(f"  {k}: {v:.1f} hours")

# small bar chart
plt.figure(figsize=(8,4))
plt.bar(list(tasks_summary.keys()), list(tasks_summary.values()))
plt.xticks(rotation=20)
plt.title("Part A: required annual hours")
plt.tight_layout()
plt.show()

# ---- Part B: LP relaxed (continuous X_{i,t}) ----
# define tasks: WARD_1..WARD_10, OPD_1..OPD_10, OR_7..OR_10, ADMIN, EDU, CONF
TASKS = []
for s in range(1, 11): TASKS.append(f"WARD_{s}")
for s in range(1, 11): TASKS.append(f"OPD_{s}")
for ss in [7,8,9,10]: TASKS.append(f"OR_{ss}")
TASKS += ["ADMIN", "EDU", "CONF"]

# build task requirements map
task_req = {}
for s in range(1,11):
    task_req[f"WARD_{s}"] = ward_by_sub.get(s, 0.0)
    task_req[f"OPD_{s}"] = opd_by_sub.get(s, 0.0)
for ss in [7,8,9,10]:
    task_req[f"OR_{ss}"] = or_by_subsub.get(ss, 0.0)
task_req["ADMIN"] = admin_hours
task_req["EDU"] = education_hours_total
task_req["CONF"] = conf_hours_total

# build LP
prob = pulp.LpProblem("CapacityPlanning_LP", pulp.LpMinimize)
X = {}
for _, row in df_specs.iterrows():
    sid = row['spec_id']
    for t in TASKS:
        # can assign only if specialist has skill for that subspecialism task or for generic tasks
        allowed = True
        if t.startswith("WARD_") or t.startswith("OPD_"):
            sub = int(t.split("_")[1])
            allowed = bool(S.get(sid, {}).get(sub, 0))
        if t.startswith("OR_"):
            # allow OR if specialist has any skill (conservative)
            allowed = (sum(S.get(sid, {}).values())>0)
        if allowed:
            X[(sid,t)] = pulp.LpVariable(f"X_{sid}_{t}", lowBound=0)

# objective: minimize total assigned hours
prob += pulp.lpSum([X[(sid,t)] for (sid,t) in X]), "MinTotalHours"
# demand constraints
for t in TASKS:
    prob += pulp.lpSum([X[(sid,t)] for (sid,tt) in X if tt==t]) >= task_req.get(t,0.0)
# capacity per specialist (annual hours = FTE * HOURS_PER_YEAR)
for _, row in df_specs.iterrows():
    sid = row['spec_id']
    fte = float(row.get('fte', 1.0))
    annual = HOURS_PER_YEAR * fte
    prob += pulp.lpSum([X[(sid,t)] for (sid_,t) in X if sid_==sid]) <= annual

prob.solve(pulp.PULP_CBC_CMD(msg=False))
print("PART B LP status:", pulp.LpStatus[prob.status])
lp_assigned = {}
for (sid,t), var in X.items():
    if var.varValue and var.varValue>0:
        lp_assigned.setdefault(sid, {})[t] = var.varValue
lp_total_hours = sum(var.varValue for var in prob.variables() if var.name.startswith('X_'))
print(f"  LP assigned total hours = {lp_total_hours:.1f}")

print("\nDetailed LP results per task:")
for t in TASKS:
    total = sum(var.varValue for (sid,tt), var in X.items() if tt == t)
    required = task_req.get(t, 0.0)
    print(f"  {t}: required = {required:.1f} h, assigned = {total:.1f} h, diff = {total - required:.1f}")

# ---- Part C: MILP with binary hires Z_i ----
prob2 = pulp.LpProblem("CapacityPlanning_MILP", pulp.LpMinimize)
X2 = {}
Z = {}
for _, row in df_specs.iterrows():
    sid = row['spec_id']
    Z[sid] = pulp.LpVariable(f"Z_{sid}", lowBound=0, upBound=1, cat='Binary')
    fte = float(row.get('fte', 1.0))
    for t in TASKS:
        allowed = True
        if t.startswith("WARD_") or t.startswith("OPD_"):
            sub = int(t.split("_")[1])
            allowed = bool(S.get(sid, {}).get(sub, 0))
        if t.startswith("OR_"):
            allowed = (sum(S.get(sid, {}).values())>0)
        if allowed:
            X2[(sid,t)] = pulp.LpVariable(f"X2_{sid}_{t}", lowBound=0)
# objective: minimize number of hired specialists weighted by FTE (proxy)
prob2 += pulp.lpSum([float(df_specs.loc[df_specs['spec_id']==sid,'fte'].iloc[0]) * Z[sid] for sid in Z]) + 1e-3 * pulp.lpSum([X2[(sid,t)] for (sid,t) in X2])
# demand constraints
for t in TASKS:
    prob2 += pulp.lpSum([X2[(sid,t)] for (sid,tt) in X2 if tt==t]) >= task_req.get(t,0.0)
# capacity constraints: X2 sum <= annual_hours * Z[sid]
for _, row in df_specs.iterrows():
    sid = row['spec_id']
    fte = float(row.get('fte', 1.0))
    annual = HOURS_PER_YEAR * fte
    prob2 += pulp.lpSum([X2[(sid,t)] for (sid_,t) in X2 if sid_==sid]) <= annual * Z[sid]

prob2.solve(pulp.PULP_CBC_CMD(msg=False))
print("PART C MILP status:", pulp.LpStatus[prob2.status])
hired = {sid: int(Zvar.varValue or 0) for sid, Zvar in Z.items()}
print("  specialists hired (count):", sum(hired.values()))
milp_assigned = {}
for (sid,t), var in X2.items():
    if var.varValue and var.varValue>0.0:
        milp_assigned.setdefault(sid, {})[t] = var.varValue

# ---- Part D: MILP with multiple slack levels ----
for SLACK in [0.10, 0.15, 0.20]:
    print(f"\n=== Solving MILP with {int(SLACK*100)}% slack ===")

    task_req_slack = {t: task_req.get(t,0.0) * (1 + SLACK) for t in TASKS}
    prob_slack = pulp.LpProblem(f"CapacityPlanning_MILP_Slack_{int(SLACK*100)}", pulp.LpMinimize)

    Xs = {}
    Zs = {}

    # variable creation
    for _, row in df_specs.iterrows():
        sid = row['spec_id']
        Zs[sid] = pulp.LpVariable(f"Z_{int(SLACK*100)}_{sid}", lowBound=0, upBound=1, cat='Binary')
        fte = float(row.get('fte', 1.0))
        for t in TASKS:
            allowed = True
            if t.startswith("WARD_") or t.startswith("OPD_"):
                sub = int(t.split("_")[1])
                allowed = bool(S.get(sid, {}).get(sub, 0))
            if t.startswith("OR_"):
                allowed = (sum(S.get(sid, {}).values()) > 0)
            if allowed:
                Xs[(sid,t)] = pulp.LpVariable(f"X_{int(SLACK*100)}_{sid}_{t}", lowBound=0)

    # objective
    prob_slack += pulp.lpSum([
        float(df_specs.loc[df_specs['spec_id']==sid,'fte'].iloc[0]) * Zs[sid] for sid in Zs
    ]) + 1e-3 * pulp.lpSum([Xs[(sid,t)] for (sid,t) in Xs])

    # demand constraints
    for t in TASKS:
        prob_slack += pulp.lpSum([Xs[(sid,t)] for (sid,tt) in Xs if tt==t]) >= task_req_slack.get(t,0.0)

    # capacity constraints
    for _, row in df_specs.iterrows():
        sid = row['spec_id']
        fte = float(row.get('fte', 1.0))
        annual = HOURS_PER_YEAR * fte
        prob_slack += pulp.lpSum([Xs[(sid,t)] for (sid_,t) in Xs if sid_==sid]) <= annual * Zs[sid]

    # solve
    prob_slack.solve(pulp.PULP_CBC_CMD(msg=False))
    print(f"  Status: {pulp.LpStatus[prob_slack.status]}")

    hired_slack = {sid: int(Zvar.varValue or 0) for sid, Zvar in Zs.items()}
    print(f"  Specialists hired: {sum(hired_slack.values())}")

    assigned_with_slack = {}
    for (sid,t), var in Xs.items():
        if var.varValue and var.varValue > 0:
            assigned_with_slack.setdefault(sid, {})[t] = var.varValue

# ---- Part E: weekly schedule heuristic and simulation ----
# use 15% slack hires from Part D (hired_slack) and assigned_with_slack hours
if len(hired_specs)==0:
    # fallback: pick top FTE specialists
    hired_specs = df_specs.sort_values('fte', ascending=False)['spec_id'].tolist()[:max(1, math.ceil(len(spec_ids)/2))]

# determine holiday and conference weeks per spec (use parsed preferences if available)
spec_hol_weeks = {}
spec_conf_weeks = {}
HOL_WEEKS_REQ = 8
CONF_WEEKS_REQ = 2
for sid in spec_ids:
    hw = holidays.get(sid, list(range(1, HOL_WEEKS_REQ+1)))
    cw = confs.get(sid, [1,2])
    # trim or pad
    spec_hol_weeks[sid] = hw[:HOL_WEEKS_REQ] if len(hw)>=HOL_WEEKS_REQ else hw + [w for w in range(1,53) if w not in hw][:HOL_WEEKS_REQ-len(hw)]
    spec_conf_weeks[sid] = cw[:CONF_WEEKS_REQ] if len(cw)>=CONF_WEEKS_REQ else cw + [w for w in range(1,53) if w not in cw][:CONF_WEEKS_REQ-len(cw)]

# build per-spec remaining hours to schedule (from assigned_with_slack)
per_spec_remaining = {}
for sid in hired_specs:
    per_spec_remaining[sid] = assigned_with_slack.get(sid, {})
    # if spec has no assigned tasks (rare), create zero map
    if per_spec_remaining[sid] == {}:
        per_spec_remaining[sid] = {}

# weekly schedule: for weeks 1..52, allocate each spec's assigned hours evenly across non-holiday/non-conf weeks
weekly_schedule = {w: [] for w in range(1, WEEKS_PER_YEAR+1)}
for sid in hired_specs:
    unavailable = set(spec_hol_weeks.get(sid, [])) | set(spec_conf_weeks.get(sid, []))
    working_weeks = [w for w in range(1, WEEKS_PER_YEAR+1) if w not in unavailable]
    if len(working_weeks)==0:
        continue
    # get tasks & hours
    tasks_hours = per_spec_remaining.get(sid, {})
    total_hours = sum(tasks_hours.values()) if tasks_hours else 0.0
    # split each task across working weeks evenly
    for t,hours in tasks_hours.items():
        per_week = hours / len(working_weeks)
        for w in working_weeks:
            weekly_schedule[w].append((sid, t, per_week))

# compute weekly utilization summary
weekly_metrics = []
for w in range(1, WEEKS_PER_YEAR+1):
    assigned = weekly_schedule[w]
    per_spec_week = {}
    for sid,t,h in assigned:
        per_spec_week[sid] = per_spec_week.get(sid, 0.0) + h
    # compute average utilization across hired specs
    utilizations = []
    for sid in hired_specs:
        fte = float(df_specs.loc[df_specs['spec_id']==sid,'fte'].iloc[0])
        avail = fte * HOURS_PER_WEEK
        used = per_spec_week.get(sid, 0.0)
        utilizations.append(used/avail if avail>0 else 0.0)
    weekly_metrics.append({"week": w, "avg_util": np.mean(utilizations) if len(utilizations)>0 else 0.0, "total_assigned": sum(per_spec_week.values())})
df_weekly_metrics = pd.DataFrame(weekly_metrics)

plt.figure(figsize=(10,4))
plt.plot(df_weekly_metrics['week'], df_weekly_metrics['avg_util'], label='avg utilization')
plt.xlabel('Week')
plt.ylabel('Average utilization (fraction)')
plt.title('Part E: weekly average utilization (heuristic)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Simple weekly simulation of patient flow using weekly OPD and OR capacity from schedule
def simulate_weekly(pat_df, surg_df, weekly_schedule, weeks=52):
    opd_queue = pat_df.copy().sort_values(by=pat_arr_col).reset_index(drop=True)
    surg_queue = surg_df.copy().sort_values(by=surgq_due_col).reset_index(drop=True)
    week_stats = []
    for w in range(1, weeks+1):
        assigns = weekly_schedule.get(w, [])
        opd_hours = sum(h for (sid,t,h) in assigns if t.startswith('OPD_'))
        or_hours = sum(h for (sid,t,h) in assigns if t.startswith('OR_'))
        # process OPD: number of patients = opd_hours / avg consult
        if len(opd_queue)>0:
            avg_consult_h = opd_queue[pat_time_col].mean()/60.0
        else:
            avg_consult_h = 0.333
        max_opd = int(opd_hours/avg_consult_h) if avg_consult_h>0 else 0
        processed_opd = opd_queue.head(max_opd)
        opd_queue = opd_queue.iloc[max_opd:].reset_index(drop=True)
        # process OR similarly using surg oper_minutes average
        if len(surg_queue)>0:
            avg_or_h = (surg_queue['oper_minutes'].mean()/60.0) if 'oper_minutes' in surg_queue.columns else 1.5
        else:
            avg_or_h = 1.5
        max_or = int(or_hours / avg_or_h) if avg_or_h>0 else 0
        processed_or = surg_queue.head(max_or)
        surg_queue = surg_queue.iloc[max_or:].reset_index(drop=True)
        week_stats.append({
            'week': w,
            'opd_processed': len(processed_opd),
            'surgeries_processed': len(processed_or),
            'opd_queue_length': len(opd_queue),
            'surg_queue_length': len(surg_queue)
        })
    return pd.DataFrame(week_stats)

sim_df = simulate_weekly(df_pat, surg_q, weekly_schedule, weeks=WEEKS_PER_YEAR)
print("PART E Simulation summary (first 8 weeks):")
print(sim_df.head(8))

# plot cumulative processed
sim_df['cum_opd'] = sim_df['opd_processed'].cumsum()
sim_df['cum_surg'] = sim_df['surgeries_processed'].cumsum()
plt.figure(figsize=(10,4))
plt.plot(sim_df['week'], sim_df['cum_opd'], label='OPD processed (cum)')
plt.plot(sim_df['week'], sim_df['cum_surg'], label='Surgery processed (cum)')
plt.legend()
plt.xlabel('Week')
plt.ylabel('Cumulative processed')
plt.title('Part E: cumulative processed patients')
plt.tight_layout()
plt.show()

# Final prints
print("\nKey outputs:")
print(f"  - Part A hours: ward {ward_hours:.1f}, OPD {opd_total_hours:.1f}, OR {or_total_hours:.1f}, admin {admin_hours:.1f}")
print(f"  - Part B LP assigned total hours: {lp_total_hours:.1f}")
print(f"  - Part C MILP hired count: {sum(hired.values())}")
print(f"  - Part D MILP-with-slack hired count: {sum(hired_slack.values())}")
print("  - Part E: weekly utilization (avg across weeks): {:.1%}".format(df_weekly_metrics['avg_util'].mean()))

# End of script
