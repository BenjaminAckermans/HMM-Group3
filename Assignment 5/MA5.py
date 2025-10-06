#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 14:13:58 2025

@author: ariannaperini
"""

# hospital_bed_planning.py
# Python 3.9+ recommended
# Uses: pandas, numpy, matplotlib
# IMPORTANT: No seaborn used (assignment requirement). Plots use matplotlib.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heapq
import math
from collections import Counter
from datetime import datetime, time

np.random.seed(0)  # reproducible sampling for initial seeding

DATA_PATH = "Data Assignment 5.xlsx"   # update path if necessary

# ---------------------------
# Load & preprocess dataset
# ---------------------------
df = pd.read_excel(DATA_PATH, sheet_name=0)
df.columns = [c.strip() for c in df.columns]

# arrival = OR ready + 1 hour (assignment)
df['arrival'] = df['OR ready'] + pd.Timedelta(hours=1)

# normalize text fields
df['subgroup ward'] = df['subgroup ward'].str.strip().str.lower()
df['ward group'] = df['ward group'].str.strip().str.lower()

# compute departure function (inpatients depart at 09:00 on arrival_date + LOS_days)
def compute_departure(arrival_ts, los_days):
    dep = arrival_ts + pd.Timedelta(days=int(los_days))
    return pd.Timestamp(datetime.combine(dep.date(), time(hour=9, minute=0)))

# daycare: stay = 4 hours (assignment)
def compute_row_departure(r):
    if (r['subgroup ward'] == 'daycare') and (r['LOS Ward'] == 0):
        return r['arrival'] + pd.Timedelta(hours=4)
    else:
        return compute_departure(r['arrival'], r['LOS Ward'])

df['departure'] = df.apply(compute_row_departure, axis=1)

# masks / subsets
DAYCARE_MASK = (df['subgroup ward'] == 'daycare') & (df['LOS Ward'] == 0)
daycare_df = df[DAYCARE_MASK].copy().sort_values('arrival')
ip_df = df[~DAYCARE_MASK].copy().sort_values('arrival')  # inpatients

# ---------------------------
# Helper utilities
# ---------------------------
def seed_initial_heap(num_init, arrivals_subset, min_los=1):
    """
    Return a heap (list) of departure datetimes representing seeded patients.
    Sampling uses empirical LOS from arrivals_subset (filtering los >= min_los).
    """
    heap = []
    if num_init <= 0:
        return heap
    los_vals = arrivals_subset['LOS Ward'].values
    candidates = los_vals[los_vals >= min_los] if len(los_vals) > 0 else np.array([min_los])
    if len(candidates) == 0:
        sampled = np.ones(num_init, dtype=int) * min_los
    else:
        sampled = np.random.choice(candidates, size=num_init, replace=True)
    start_dt = pd.Timestamp("2010-01-29 08:00:00")
    for s in sampled:
        dep = compute_departure(start_dt, int(s)).to_pydatetime()
        heapq.heappush(heap, dep)
    return heap

# ---------------------------
# PART A: Daycare Ward Planning
# ---------------------------
def part_a_daycare(daycare_df, target_direct_rate=0.98, verbose=True):
    # precompute arrivals grouped by date (daycare resets empty at 08:00 each day)
    daycare_df = daycare_df.sort_values('arrival')
    daycare_df['day'] = daycare_df['arrival'].dt.date
    arrivals_by_day = {d: list(g[['arrival','departure','patientnr']].itertuples(index=False, name=None))
                       for d,g in daycare_df.groupby('day')}

    def simulate_daycare(capacity):
        total=0; direct=0; hourly_occ = Counter(); daily_last_dep=[]
        for day, arrs in arrivals_by_day.items():
            occ_heap = []
            # process arrivals chronologically for that day
            for arrival, departure, pid in sorted(arrs, key=lambda x: x[0]):
                # free beds that departed before arrival
                while occ_heap and occ_heap[0] <= arrival:
                    heapq.heappop(occ_heap)
                total += 1
                if len(occ_heap) < capacity:
                    direct += 1
                    heapq.heappush(occ_heap, departure)
                # count hourly occupancy (approx)
                t = arrival
                while t < departure:
                    hourly_occ[(day, t.hour)] += 1
                    t += pd.Timedelta(hours=1)
            # compute last departure among admitted on this day
            occ_heap2 = []
            admitted_deps=[]
            for arrival, departure, pid in sorted(arrs, key=lambda x: x[0]):
                while occ_heap2 and occ_heap2[0] <= arrival:
                    heapq.heappop(occ_heap2)
                if len(occ_heap2) < capacity:
                    heapq.heappush(occ_heap2, departure)
                    admitted_deps.append(departure)
            daily_last_dep.append(max(admitted_deps) if admitted_deps else pd.Timestamp(datetime.combine(day, time(8,0))))
        direct_rate = direct/total if total>0 else 1.0
        # hourly aggregated utilization
        hourly_df = pd.DataFrame([{'date':d,'hour':h,'occ':occ} for (d,h),occ in hourly_occ.items()]) if hourly_occ else pd.DataFrame(columns=['date','hour','occ'])
        hourly_agg = hourly_df.groupby('hour')['occ'].mean().reset_index() if not hourly_df.empty else pd.DataFrame(columns=['hour','occ'])
        hourly_agg['util'] = hourly_agg['occ'] / capacity if not hourly_agg.empty else hourly_agg.get('occ', pd.Series(dtype=float))
        last_hours = [dt.hour + dt.minute/60.0 for dt in daily_last_dep] if daily_last_dep else [8.0]
        closing_hour_95 = float(np.percentile(last_hours, 95))
        return {
            'capacity': capacity,
            'direct_rate': direct_rate,
            'total_arrivals': total,
            'direct': direct,
            'hourly_agg': hourly_agg,
            'closing_hour_95pct': closing_hour_95
        }

    # estimate max simultaneous daycare demand to set search range
    max_sim = 0
    for day, arrs in arrivals_by_day.items():
        ev=[]
        for a,d,_ in arrs:
            ev.append((a,1)); ev.append((d,-1))
        ev.sort()
        cur = 0
        for _,delta in ev:
            cur += delta; max_sim = max(max_sim, cur)

    results=[]
    for capacity in range(1, max_sim+8):
        results.append(simulate_daycare(capacity))
    res_df = pd.DataFrame([{'capacity':r['capacity'],'direct_rate':r['direct_rate'],'closing_hour_95pct':r['closing_hour_95pct']} for r in results])

    feasible = res_df[res_df['direct_rate'] >= target_direct_rate]
    best_capacity = int(feasible.sort_values('capacity').iloc[0]['capacity']) if not feasible.empty else None

    if verbose:
        print("PART A — daycare capacity search done.")
        print("Suggested capacity (>= {:.2f} direct): {}".format(target_direct_rate, best_capacity))
        if best_capacity:
            chosen = [r for r in results if r['capacity']==best_capacity][0]
            print("95th percentile last-departure (closing) hour:", chosen['closing_hour_95pct'])
            # quick plot
            ha = chosen['hourly_agg']
            if not ha.empty:
                plt.figure(figsize=(8,4))
                plt.plot(ha['hour'], ha['util'], marker='o')
                plt.xlabel('Hour of day'); plt.ylabel('Average utilization'); plt.title(f'Daycare hourly mean util (capacity={best_capacity})')
                plt.grid(True); plt.show()
    return res_df, best_capacity

# ---------------------------
# PART B: Specialist wards (3 groups)
# ---------------------------
def part_b_specialist(ip_df, target_direct_rate=0.95, weekend_factor=0.8, verbose=True):
    specialties = sorted(ip_df['ward group'].unique())
    # helper to simulate a single specialty independently
    def simulate_specialty(cap_weekday, specialty):
        cap_weekend = max(1, math.ceil(cap_weekday * weekend_factor))
        sub = ip_df[ip_df['ward group'] == specialty].sort_values('arrival')
        total = len(sub)
        direct = 0
        # seed 50% occupancy of planned beds
        occ_heap = []
        num_init = int(round(0.5 * cap_weekday))
        if num_init > 0 and total>0:
            possible_los = sub['LOS Ward'].values[sub['LOS Ward'].values >= 1]
            if len(possible_los)==0:
                sampled = np.ones(num_init, dtype=int)
            else:
                sampled = np.random.choice(possible_los, size=num_init, replace=True)
            start_dt = pd.Timestamp("2010-01-29 08:00:00")
            for los in sampled:
                heapq.heappush(occ_heap, compute_departure(start_dt, int(los)).to_pydatetime())
        for _,row in sub.iterrows():
            arrival = row['arrival'].to_pydatetime()
            cap = cap_weekend if arrival.weekday()>=5 else cap_weekday
            while occ_heap and occ_heap[0] <= arrival:
                heapq.heappop(occ_heap)
            if len(occ_heap) < cap:
                direct += 1
                heapq.heappush(occ_heap, row['departure'].to_pydatetime())
        direct_rate = direct/total if total>0 else 1.0
        return direct_rate

    # determine a scan upper bound (peak concurrent + margin)
    upper_bounds = {}
    for s in specialties:
        sub = ip_df[ip_df['ward group'] == s]
        ev=[]; cur=0; peak=0
        for _,r in sub.iterrows():
            ev.append((r['arrival'].to_pydatetime(),1)); ev.append((r['departure'].to_pydatetime(),-1))
        ev.sort()
        for t,delta in ev:
            cur += delta; peak = max(peak, cur)
        upper_bounds[s] = max(5, int(peak + 10))

    results = {}
    for s in specialties:
        found = None
        for cap in range(1, upper_bounds[s] + 1):
            dr = simulate_specialty(cap, s)
            if dr >= target_direct_rate:
                found = cap
                break
        if found is None:
            # fallback: choose upper bound as conservative value
            found = upper_bounds[s]
        results[s] = found
    if verbose:
        print("PART B — suggested weekday capacities (target direct >= {:.2f}):".format(target_direct_rate))
        for s,v in results.items():
            print(f"  {s}: weekday capacity = {v} (weekend approx {math.ceil(v*weekend_factor)})")
    return results

# ---------------------------
# PART C: Flex ward (shared overflow)
# ---------------------------
def part_c_with_flex(ip_df, base_caps, weekend_factor=0.8,
                     target_specialty_admit=0.90, target_overall_admit=0.99,
                     max_inc=4, flex_max=20, verbose=True):
    # greedy search: increase base capacities together up to max_inc, test flex sizes 0..flex_max
    def simulate_with_flex(capacities, flex_capacity):
        wards = {w: {'cap_weekday': capacities[w], 'cap_weekend': max(1, math.ceil(capacities[w] * weekend_factor)), 'heap': []} for w in capacities}
        flex = {'cap_weekday': flex_capacity, 'cap_weekend': flex_capacity, 'heap': []}
        # seed initial occupancy in specialty wards (50%)
        start_dt = pd.Timestamp("2010-01-29 08:00:00")
        for w in wards:
            num_init = int(round(0.5 * wards[w]['cap_weekday']))
            sub = ip_df[ip_df['ward group'] == w]
            possible_los = sub['LOS Ward'].values[sub['LOS Ward'].values >= 1] if len(sub) > 0 else np.array([1])
            if len(possible_los) > 0:
                sampled = np.random.choice(possible_los, size=num_init, replace=True)
            else:
                sampled = np.ones(num_init, dtype=int)
            for los in sampled:
                heapq.heappush(wards[w]['heap'], compute_departure(start_dt, int(los)).to_pydatetime())
        total = len(ip_df)
        specialty_admit = 0; flex_admit = 0; rejected = 0
        for _,r in ip_df.iterrows():
            arrival = r['arrival'].to_pydatetime(); w = r['ward group']; los = int(r['LOS Ward']); departure = r['departure'].to_pydatetime()
            # release departures
            for ww in wards:
                while wards[ww]['heap'] and wards[ww]['heap'][0] <= arrival:
                    heapq.heappop(wards[ww]['heap'])
            while flex['heap'] and flex['heap'][0] <= arrival:
                heapq.heappop(flex['heap'])
            cap_today = wards[w]['cap_weekend'] if arrival.weekday()>=5 else wards[w]['cap_weekday']
            if len(wards[w]['heap']) < cap_today:
                heapq.heappush(wards[w]['heap'], departure); specialty_admit += 1
            else:
                if los >= 10:
                    rejected += 1
                else:
                    cap_flex_today = flex['cap_weekday'] if arrival.weekday()<5 else flex['cap_weekend']
                    if len(flex['heap']) < cap_flex_today:
                        heapq.heappush(flex['heap'], departure); flex_admit += 1
                    else:
                        rejected += 1
        return {
            'specialty_admit_rate': specialty_admit / total,
            'overall_admit_rate': (specialty_admit + flex_admit) / total,
            'specialty_admit': specialty_admit,
            'flex_admit': flex_admit,
            'rejected': rejected,
            'total': total
        }

    best_solution = None
    best_total = float('inf')
    # greedy: raise each specialty by same increment (simple approach)
    for inc in range(0, max_inc + 1):
        capacities_try = {w: max(1, base_caps[w] + inc) for w in base_caps}
        for flex in range(0, flex_max + 1):
            metrics = simulate_with_flex(capacities_try, flex)
            if metrics['specialty_admit_rate'] >= target_specialty_admit and metrics['overall_admit_rate'] >= target_overall_admit:
                total_beds = sum(capacities_try.values()) + flex
                if total_beds < best_total:
                    best_total = total_beds
                    best_solution = {
                        'capacities': capacities_try.copy(),
                        'flex': flex,
                        'metrics': metrics
                    }
        if best_solution:
            break  # found a feasible solution in the greedy neighborhood
    if verbose:
        if best_solution:
            print("PART C — found solution (greedy neighborhood):")
            print("  specialty capacities:", best_solution['capacities'])
            print("  flex beds:", best_solution['flex'])
            print("  metrics:", best_solution['metrics'])
            print("  total beds:", best_total)
        else:
            print("PART C — no feasible solution found in the searched neighborhood (increase search ranges).")
    return best_solution

# ---------------------------
# PART D: 10 subgroup wards (aos1-5, vts1-3, trs1-2)
# ---------------------------
def part_d_subgroups(df, target_own_admit=0.85, verbose=True):
    # standardize subgroup ward names (remove internal spaces and lowercase)
    df['subgroup ward'] = df['subgroup ward'].str.replace(r'\s+', '', regex=True).str.lower()

    # collect subgroup list excluding daycare
    all_subgroups = [s for s in df['subgroup ward'].unique() if s != 'daycare' and (s.startswith('aos') or s.startswith('vts') or s.startswith('trs'))]

    # limit to exactly 10 subgroups (top 10 by frequency)
    subgroup_counts = df['subgroup ward'].value_counts()
    subgroups = [s for s in subgroup_counts.index if s in all_subgroups][:10]
    subgroups = sorted(subgroups)  # keep consistent order

    subgroup_df = df[df['subgroup ward'].isin(subgroups)].sort_values('arrival').reset_index(drop=True)
    subgroup_to_specialty = {s: df[df['subgroup ward']==s]['ward group'].mode()[0] for s in subgroups}
    # initial capacities heuristic: mean_daily_arrivals * mean LOS
    capacities = {}
    for s in subgroups:
        sub = subgroup_df[subgroup_df['subgroup ward'] == s]
        if len(sub) == 0:
            capacities[s] = 1
        else:
            mean_daily = sub.groupby(sub['arrival'].dt.date).size().mean()
            mean_los = sub['LOS Ward'].mean()
            init = math.ceil(mean_daily * mean_los) if (mean_daily > 0 and mean_los > 0) else max(1, math.ceil(mean_daily))
            capacities[s] = max(1, init)
    # upper bound safety
    upper = {}
    for s in subgroups:
        sub = subgroup_df[subgroup_df['subgroup ward'] == s]
        ev=[]; cur=0; peak=0
        for _,r in sub.iterrows():
            ev.append((r['arrival'].to_pydatetime(),1)); ev.append((r['departure'].to_pydatetime(),-1))
        ev.sort()
        for t,delta in ev:
            cur += delta; peak = max(peak, cur)
        upper[s] = max(1, int(peak + 5))

    # simulation: allow intra-specialty alternatives (if own subgroup full)
    def simulate_subgroups(capac):
        heaps = {s: [] for s in capac}
        # seed 50% occupancy
        for s in capac:
            num_init = int(round(0.5 * capac[s]))
            sub = subgroup_df[subgroup_df['subgroup ward'] == s]
            possible_los = sub['LOS Ward'].values[sub['LOS Ward'].values >= 1] if len(sub) > 0 else np.array([1])
            sampled = np.random.choice(possible_los, size=num_init, replace=True) if len(possible_los) > 0 else np.ones(num_init, dtype=int)
            start_dt = pd.Timestamp("2010-01-29 08:00:00")
            for los in sampled:
                heapq.heappush(heaps[s], compute_departure(start_dt, int(los)).to_pydatetime())
        total_counts = subgroup_df.groupby('subgroup ward').size().to_dict()
        own_admit = Counter()
        rejected = 0
        for _,r in subgroup_df.iterrows():
            arrival = r['arrival'].to_pydatetime(); s = r['subgroup ward']; departure = r['departure'].to_pydatetime()
            for ss in heaps:
                while heaps[ss] and heaps[ss][0] <= arrival:
                    heapq.heappop(heaps[ss])
            if len(heaps[s]) < capac[s]:
                heapq.heappush(heaps[s], departure); own_admit[s] += 1
            else:
                # attempt other subgroups in same specialty
                candidates = [ss for ss in capac if subgroup_to_specialty[ss] == subgroup_to_specialty[s] and ss != s]
                free = [(capac[ss] - len(heaps[ss]), ss) for ss in candidates]
                free = [(fs, ss) for fs, ss in free if fs > 0]
                if free:
                    chosen = max(free)[1]; heapq.heappush(heaps[chosen], departure)
                else:
                    rejected += 1
        own_rate = {s: own_admit[s] / (total_counts[s] if total_counts.get(s,0) > 0 else 1) for s in capac}
        return own_rate, rejected

    # greedy incremental increase until own_rate >= target_own_admit
    iteration = 0
    while True:
        iteration += 1
        own_rate, rejected = simulate_subgroups(capacities)
        to_inc = [s for s in capacities if own_rate[s] < target_own_admit and capacities[s] < upper[s]]
        if not to_inc:
            break
        worst = min(to_inc, key=lambda s: own_rate[s])
        capacities[worst] += 1
        if iteration > 5000:
            print("Reached iteration cap in Part D greedy loop")
            break
    if verbose:
        print("PART D — subgroup capacities determined by greedy algorithm. Total beds:", sum(capacities.values()))
        for s in capacities:
            print(f"  {s}: {capacities[s]} (own-admit {own_rate[s]:.3f})")
    return capacities, own_rate

# ---------------------------
# PART E: Hybrid design (example implementation)
# ---------------------------
def part_e_hybrid(ip_df, weekend_factor=0.8, verbose=True):
    # Design:
    # - Pool short-stay (LOS <= 3) into a shared short-stay ward
    # - Long-stay (LOS >= 4) have dedicated specialty long-stay beds sized at 90th percentile concurrent long-stay demand
    # - small flex ward for overflow
    specialties = sorted(ip_df['ward group'].unique())
    long_demands = {}
    for s in specialties:
        sub = ip_df[(ip_df['ward group'] == s) & (ip_df['LOS Ward'] >= 4)]
        ev=[]; cur=0; timeline=[]
        for _,r in sub.iterrows():
            ev.append((r['arrival'].to_pydatetime(),1)); ev.append((r['departure'].to_pydatetime(),-1))
        ev.sort()
        for t,delta in ev:
            cur += delta; timeline.append(cur)
        long_demands[s] = max(1, int(np.percentile(timeline, 90))) if timeline else 1
    # pooled short stay size
    short_sub = ip_df[ip_df['LOS Ward'] <= 3]
    ev=[]; cur=0; timeline=[]
    for _,r in short_sub.iterrows():
        ev.append((r['arrival'].to_pydatetime(),1)); ev.append((r['departure'].to_pydatetime(),-1))
    ev.sort()
    for t,delta in ev:
        cur += delta; timeline.append(cur)
    avg_pooled = int(np.mean(timeline)) if timeline else 1
    short_pool_suggest = max(1, int(math.ceil(avg_pooled * 1.2)))
    flex_suggest = 5

    # simulation (similar to earlier hybrid)
    def simulate_hybrid(short_pool_beds, longbeds_dict, flex_beds):
        wards_long = {s: {'cap_weekday': longbeds_dict[s], 'heap': []} for s in longbeds_dict}
        short_pool = {'cap_weekday': short_pool_beds, 'heap': []}
        flex = {'cap_weekday': flex_beds, 'heap': []}
        # seed long wards
        start_dt = pd.Timestamp("2010-01-29 08:00:00")
        for s in wards_long:
            num_init = int(round(0.5 * wards_long[s]['cap_weekday']))
            sub = ip_df[ip_df['ward group'] == s]
            candidates = sub['LOS Ward'].values[sub['LOS Ward'].values >= 4] if len(sub) > 0 else np.array([4])
            sampled = np.random.choice(candidates, size=num_init, replace=True) if len(candidates) > 0 else np.ones(num_init, dtype=int) * 4
            for los in sampled:
                heapq.heappush(wards_long[s]['heap'], compute_departure(start_dt, int(los)).to_pydatetime())
        total = 0; admitted_specialty = 0; admitted_shortpool = 0; admitted_flex = 0; rejected = 0
        for _,r in ip_df.iterrows():
            arrival = r['arrival'].to_pydatetime(); s = r['ward group']; los = int(r['LOS Ward']); departure = r['departure'].to_pydatetime(); total += 1
            for ss in wards_long: 
                while wards_long[ss]['heap'] and wards_long[ss]['heap'][0] <= arrival:
                    heapq.heappop(wards_long[ss]['heap'])
            while short_pool['heap'] and short_pool['heap'][0] <= arrival:
                heapq.heappop(short_pool['heap'])
            while flex['heap'] and flex['heap'][0] <= arrival:
                heapq.heappop(flex['heap'])
            if los <= 3:
                if len(short_pool['heap']) < short_pool['cap_weekday']:
                    heapq.heappush(short_pool['heap'], departure); admitted_shortpool += 1
                else:
                    if len(flex['heap']) < flex['cap_weekday']:
                        heapq.heappush(flex['heap'], departure); admitted_flex += 1
                    else:
                        # last resort: try specialty long bed
                        if len(wards_long[s]['heap']) < wards_long[s]['cap_weekday']:
                            heapq.heappush(wards_long[s]['heap'], departure); admitted_specialty += 1
                        else:
                            rejected += 1
            else:
                if len(wards_long[s]['heap']) < wards_long[s]['cap_weekday']:
                    heapq.heappush(wards_long[s]['heap'], departure); admitted_specialty += 1
                else:
                    if los >= 10:
                        rejected += 1
                    else:
                        if len(flex['heap']) < flex['cap_weekday']:
                            heapq.heappush(flex['heap'], departure); admitted_flex += 1
                        else:
                            rejected += 1
        return {'total': total, 'admitted_specialty': admitted_specialty, 'admitted_shortpool': admitted_shortpool, 'admitted_flex': admitted_flex, 'rejected': rejected}

    hybrid_metrics = simulate_hybrid(short_pool_suggest, long_demands, flex_suggest)
    if verbose:
        print("PART E — hybrid design suggestion:")
        print("  long-stay beds (90th pct per specialty):", long_demands)
        print("  short pool suggested:", short_pool_suggest, "flex suggested:", flex_suggest)
        print("  hybrid simulation result:", hybrid_metrics)
    return {'long_demands': long_demands, 'short_pool': short_pool_suggest, 'flex': flex_suggest, 'metrics': hybrid_metrics}

# ---------------------------
# Run parts (example)
# ---------------------------
if __name__ == "__main__":
    print("Data loaded. Running Parts A-E (this will print/plot results).")
    # Part A
    a_df, daycare_capacity = part_a_daycare(daycare_df)
    # Part B
    partb_caps = part_b_specialist(ip_df)
    # Part C (greedy near Part B)
    c_solution = part_c_with_flex(ip_df, partb_caps)
    # Part D
    d_caps, d_own_rates = part_d_subgroups(df)
    # Part E
    e_design = part_e_hybrid(ip_df)

    # Summaries (print)
    print("\nSUMMARY (top-level):")
    print("Part A daycare capacity:", daycare_capacity)
    print("Part B capacities (weekday):", partb_caps, "total:", sum(partb_caps.values()))
    print("Part C solution (if any):", c_solution)
    print("Part D capacities (10 subgroup wards) total:", sum(d_caps.values()))
    print("Part E hybrid:", e_design)

    # Optionally save tables
    try:
        a_df.to_csv("part_a_daycare_summary.csv", index=False)
        pd.DataFrame([partb_caps]).to_csv("part_b_caps.csv", index=False)
        if c_solution:
            pd.DataFrame([c_solution['capacities']]).to_csv("part_c_caps.csv", index=False)
        pd.DataFrame([d_caps]).to_csv("part_d_caps.csv", index=False)
    except Exception:
        pass

