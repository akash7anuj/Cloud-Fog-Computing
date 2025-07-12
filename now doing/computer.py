
"""
balance_time_cost.py
────────────────────
Reads ExecutionTable and CostTable from an .xlsx workbook and assigns
each task to exactly one node so that the gap between the heaviest and
lightest node *under the chosen score* is minimised.

score(t, n) =
    • execution-time  if  --objective time   (default)
    • cost            if  --objective cost
    • α·time + β·cost if  --objective combo  (user supplies α, β)

After scheduling the script prints, for every node:
    total_execution_time  total_cost  and the chosen score total,
plus Tmax, Tmin, and the final gap.
"""

from __future__ import annotations
import argparse, random, time, math, sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# ──────────────────────────────────────────────────────────────
# 1.  Load ExecutionTable & CostTable
# ──────────────────────────────────────────────────────────────
def load_tables(xlsx: Path):
    """Return (tasks, nodes, exec_time, cost) as nested dicts."""
    def load_sheet(name):
        df = pd.read_excel(xlsx, sheet_name=name, engine="openpyxl")
        df = df.rename(columns={df.columns[0]: "Node"})
        df = df.dropna(subset=["Node"]).set_index("Node")
        df.index = df.index.astype(str).str.strip()
        df = df.apply(pd.to_numeric, errors="coerce")
        return df

    exec_df = load_sheet("ExecutionTable")
    cost_df = load_sheet("CostTable")

    # Ensure same shape / order
    if not exec_df.columns.equals(cost_df.columns) or not exec_df.index.equals(cost_df.index):
        sys.exit("ExecutionTable and CostTable must have identical task columns and node rows.")

    tasks = exec_df.columns.tolist()
    nodes = exec_df.index.tolist()

    exec_time: Dict[str, Dict[str, float]] = {t: {} for t in tasks}
    cost:      Dict[str, Dict[str, float]] = {t: {} for t in tasks}
    for t in tasks:
        for n in nodes:
            exec_time[t][n] = float(exec_df.at[n, t])
            cost[t][n]      = float(cost_df.at[n, t])

    return tasks, nodes, exec_time, cost


# ──────────────────────────────────────────────────────────────
# 2.  Helper   argmin without NumPy
# ──────────────────────────────────────────────────────────────
def argmin(items, key):
    best, best_val = None, math.inf
    for it in items:
        val = key(it)
        if val < best_val:
            best, best_val = it, val
    return best, best_val


# ──────────────────────────────────────────────────────────────
# 3.  Heuristic components (seed + local search)
# ──────────────────────────────────────────────────────────────
def lpt_seed(tasks, nodes, score):
    loads  = {n: 0.0 for n in nodes}
    assign = {n: []  for n in nodes}
    tasks_sorted = sorted(tasks, key=lambda t: min(score[t].values()), reverse=True)
    for t in tasks_sorted:
        n = min(nodes, key=loads.get)
        assign[n].append(t)
        loads[n] += score[t][n]
    return assign, loads


def local_search(assign, loads, score, nodes):
    improved = True
    while improved:
        improved = False
        heavy = max(loads, key=loads.get)
        light = min(loads, key=loads.get)
        gap0  = loads[heavy] - loads[light]
        best_delta, best_op = 0.0, None

        # single moves
        for t in assign[heavy]:
            new_h = loads[heavy] - score[t][heavy]
            new_l = loads[light] + score[t][light]
            new_gap = max(new_h, new_l, *[loads[n] for n in nodes if n not in (heavy, light)]) \
                    - min(new_h, new_l, *[loads[n] for n in nodes if n not in (heavy, light)])
            delta = gap0 - new_gap
            if delta > best_delta:
                best_delta, best_op = delta, ("move", heavy, light, t)

        # pair swaps
        for t_h in assign[heavy]:
            for t_l in assign[light]:
                new_h = loads[heavy] - score[t_h][heavy] + score[t_l][heavy]
                new_l = loads[light] - score[t_l][light] + score[t_h][light]
                new_gap = max(new_h, new_l, *[loads[n] for n in nodes if n not in (heavy, light)]) \
                        - min(new_h, new_l, *[loads[n] for n in nodes if n not in (heavy, light)])
                delta = gap0 - new_gap
                if delta > best_delta:
                    best_delta, best_op = delta, ("swap", heavy, light, t_h, t_l)

        # apply the best improvement
        if best_op:
            improved = True
            if best_op[0] == "move":
                _, src, dst, t = best_op
                assign[src].remove(t); assign[dst].append(t)
                loads[src] -= score[t][src]; loads[dst] += score[t][dst]
            else:
                _, h, l, t_h, t_l = best_op
                assign[h].remove(t_h); assign[h].append(t_l)
                assign[l].remove(t_l); assign[l].append(t_h)
                loads[h] += score[t_l][h] - score[t_h][h]
                loads[l] += score[t_h][l] - score[t_l][l]
    return assign, loads


def balance(tasks, nodes, score, restarts=20, seed=42):
    random.seed(seed)
    best_gap = math.inf
    best_assign = best_loads = None
    for _ in range(restarts):
        random.shuffle(tasks)
        assign, loads = lpt_seed(tasks, nodes, score)
        assign, loads = local_search(assign, loads, score, nodes)
        gap = max(loads.values()) - min(loads.values())
        if gap < best_gap:
            best_gap, best_assign, best_loads = gap, assign, loads
    return best_assign, best_loads, best_gap


# ──────────────────────────────────────────────────────────────
# 4.  Main CLI
# ──────────────────────────────────────────────────────────────
def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser(description="Balance tasks using time, cost, or a weighted combo.")
    p.add_argument("--file", default=r"C:\Users\Akash\Desktop\efficient computers\archive (2)\task120.xlsx", help="Workbook path (default: task120.xlsx)")
    p.add_argument("--objective", choices=["time", "cost", "combo"], default="time",
                   help="What to minimise the gap of (default: time)")
    p.add_argument("--alpha", type=float, default=0.5, help="Weight on time (combo mode)")
    p.add_argument("--beta",  type=float, default=0.5, help="Weight on cost (combo mode)")
    p.add_argument("--restarts", type=int, default=20, help="# random restarts (default: 20)")
    args = p.parse_args(argv)

    xlsx = Path(args.file).expanduser().resolve()
    if not xlsx.is_file():
        sys.exit(f"Workbook not found: {xlsx}")

    tasks, nodes, exec_time, cost = load_tables(xlsx)

    # choose score matrix
    if args.objective == "time":
        score = exec_time
        score_name = "execution time (s)"
    elif args.objective == "cost":
        score = cost
        score_name = "cost (currency units)"
    else:   # combo
        alpha, beta = args.alpha, args.beta
        score = {t: {n: alpha * exec_time[t][n] + beta * cost[t][n] for n in nodes} for t in tasks}
        score_name = f"combo score  (α={alpha}, β={beta})"

    t0 = time.time()
    assign, loads, gap = balance(tasks.copy(), nodes, score, restarts=args.restarts, seed=42)
    elapsed = time.time() - t0

    # compute plain totals for time & cost too
    time_tot = {n: sum(exec_time[t][n] for t in assign[n]) for n in nodes}
    cost_tot = {n: sum(cost[t][n]      for t in assign[n]) for n in nodes}

    print(f"\nBalanced on: {score_name}")
    print("Task → Node assignment")
    for n in nodes:
        print(f"{n}: {assign[n]}")

    print("\nPer-node totals")
    print("Node        Time(s)   Cost      Score")
    for n in nodes:
        print(f"{n:6}  {time_tot[n]:9.2f}  {cost_tot[n]:9.2f}  {loads[n]:9.2f}")

    print(f"\nTmax   : {max(loads.values()):.2f}")
    print(f"Tmin   : {min(loads.values()):.2f}")
    print(f"Gap Δ  : {gap:.2f}")
    print(f"\nFinished in {elapsed:.2f} s using {args.restarts} restarts.")

    # ──  WRITE RESULT TO EXCEL ────────────────────────────────────
    summary_rows = []
    for n in nodes:
        summary_rows.append(
            {
                "Node": n,
                "Assigned tasks": ", ".join(assign[n]),
                "Total exec time (s)": time_tot[n],
                "Total cost": cost_tot[n],
                "Score total": loads[n],
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    outfile = xlsx.with_name("Task_Node_Assignment.xlsx")  # same folder as input
    with pd.ExcelWriter(outfile, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="NodeSummary", index=False)

        # Optional: a 0/1 assignment matrix (node×task) on a second sheet
        matrix = pd.DataFrame(0, index=nodes, columns=tasks)
        for n in nodes:
            for t in assign[n]:
                matrix.at[n, t] = 1
        matrix.to_excel(writer, sheet_name="AssignmentMatrix")

    print(f"\nExcel report written to: {outfile}")
    # ────────────────────────────────────────────────────────────────────



if __name__ == "__main__":
    main()


# python balance_time_cost.py                           # minimise time gap
# python balance_time_cost.py --objective cost          # minimise cost gap
# python balance_time_cost.py --objective combo --alpha 0.7 --beta 0.3
