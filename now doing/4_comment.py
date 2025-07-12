"""
balance_time_cost.py
────────────────────
This script reads ExecutionTable and CostTable from an Excel file
and assigns each task to exactly one node, minimizing the gap
between the heaviest and lightest loaded nodes based on a chosen score.
"""

# Importing required libraries
from __future__ import annotations
import argparse, random, time, math, sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# ──────────────────────────────────────────────────────────────
# 1. Load ExecutionTable & CostTable
# ──────────────────────────────────────────────────────────────

def load_tables(xlsx: Path):
    """Loads tasks, nodes, execution time, and cost from an Excel file."""
    
    # Helper function to load a single sheet
    def load_sheet(name):
        df = pd.read_excel(xlsx, sheet_name=name, engine="openpyxl")  # Read the sheet
        df = df.rename(columns={df.columns[0]: "Node"})               # Rename first column to 'Node'
        df = df.dropna(subset=["Node"]).set_index("Node")             # Drop empty rows and set 'Node' as index
        df.index = df.index.astype(str).str.strip()                   # Clean up node names
        df = df.apply(pd.to_numeric, errors="coerce")                 # Convert all values to numbers
        return df

    # Load both Execution and Cost sheets
    exec_df = load_sheet("ExecutionTable")
    cost_df = load_sheet("CostTable")

    # Validate: both tables must match in size and labels
    if not exec_df.columns.equals(cost_df.columns) or not exec_df.index.equals(cost_df.index):
        sys.exit("ExecutionTable and CostTable must have identical task columns and node rows.")

    tasks = exec_df.columns.tolist()   # List of task names
    nodes = exec_df.index.tolist()     # List of node names

    # Initialize dictionaries for execution time and cost
    exec_time: Dict[str, Dict[str, float]] = {t: {} for t in tasks}
    cost: Dict[str, Dict[str, float]] = {t: {} for t in tasks}
    
    # Fill execution time and cost values
    for t in tasks:
        for n in nodes:
            exec_time[t][n] = float(exec_df.at[n, t])
            cost[t][n] = float(cost_df.at[n, t])

    return tasks, nodes, exec_time, cost  # Return everything

# ──────────────────────────────────────────────────────────────
# 2. Helper Function: argmin (find minimum without using numpy)
# ──────────────────────────────────────────────────────────────

def argmin(items, key):
    """Return the item with the minimum key value."""
    best, best_val = None, math.inf
    for it in items:
        val = key(it)
        if val < best_val:
            best, best_val = it, val
    return best, best_val

# ──────────────────────────────────────────────────────────────
# 3. Heuristic components (seed + local search)
# ──────────────────────────────────────────────────────────────

def lpt_seed(tasks, nodes, score):
    """Initial greedy assignment based on Longest Processing Time first."""
    
    loads = {n: 0.0 for n in nodes}  # Start load at 0 for all nodes
    assign = {n: [] for n in nodes}  # No task assigned initially

    # Sort tasks by their minimum score across nodes, largest first
    tasks_sorted = sorted(tasks, key=lambda t: min(score[t].values()), reverse=True)

    for t in tasks_sorted:
        n = min(nodes, key=loads.get)  # Find the node with minimum current load
        assign[n].append(t)            # Assign the task to this node
        loads[n] += score[t][n]         # Update the load for that node

    return assign, loads  # Return assignment and loads

def local_search(assign, loads, score, nodes):
    """Tries to improve the assignment by moving or swapping tasks."""
    
    improved = True  # Assume improvement is possible initially

    while improved:
        improved = False  # Set to False; will update if an improvement is found

        heavy = max(loads, key=loads.get)  # Node with maximum load
        light = min(loads, key=loads.get)  # Node with minimum load
        gap0 = loads[heavy] - loads[light]  # Current gap between max and min loads

        best_delta, best_op = 0.0, None  # Initialize best improvement

        # Try moving a task from heavy to light
        for t in assign[heavy]:
            new_h = loads[heavy] - score[t][heavy]  # Load on heavy if task is moved
            new_l = loads[light] + score[t][light]  # Load on light if task is moved
            new_gap = max(new_h, new_l, *[loads[n] for n in nodes if n not in (heavy, light)]) \
                      - min(new_h, new_l, *[loads[n] for n in nodes if n not in (heavy, light)])
            delta = gap0 - new_gap  # How much the gap improves

            if delta > best_delta:
                best_delta, best_op = delta, ("m56ove", heavy, light, t)  # Store best move

        # Try swapping a task between heavy and light nodes
        for t_h in assign[heavy]:
            for t_l in assign[light]:
                new_h = loads[heavy] - score[t_h][heavy] + score[t_l][heavy]
                new_l = loads[light] - score[t_l][light] + score[t_h][light]
                new_gap = max(new_h, new_l, *[loads[n] for n in nodes if n not in (heavy, light)]) \
                          - min(new_h, new_l, *[loads[n] for n in nodes if n not in (heavy, light)])
                delta = gap0 - new_gap

                if delta > best_delta:
                    best_delta, best_op = delta, ("swap", heavy, light, t_h, t_l)  # Store best swap

        # Apply the best operation found
        if best_op:
            improved = True  # Mark that an improvement was found

            if best_op[0] == "move":
                _, src, dst, t = best_op
                assign[src].remove(t)      # Remove task from source node
                assign[dst].append(t)       # Add task to destination node
                loads[src] -= score[t][src]  # Update loads
                loads[dst] += score[t][dst]
            else:
                _, h, l, t_h, t_l = best_op
                assign[h].remove(t_h); assign[h].append(t_l)
                assign[l].remove(t_l); assign[l].append(t_h)
                loads[h] += score[t_l][h] - score[t_h][h]
                loads[l] += score[t_h][l] - score[t_l][l]

    return assign, loads  # Return updated assignment and loads

def balance(tasks, nodes, score, restarts=20, seed=42):
    """Runs multiple random attempts to find the best possible assignment."""
    
    random.seed(seed)  # Fix random seed for repeatability

    best_gap = math.inf  # Start with worst possible gap
    best_assign = best_loads = None  # To store best result

    for _ in range(restarts):
        random.shuffle(tasks)  # Shuffle tasks randomly
        assign, loads = lpt_seed(tasks, nodes, score)  # Do initial assignment
        assign, loads = local_search(assign, loads, score, nodes)  # Try to improve it
        gap = max(loads.values()) - min(loads.values())  # Calculate the final gap

        if gap < best_gap:
            best_gap, best_assign, best_loads = gap, assign, loads  # Save best found assignment

    return best_assign, best_loads, best_gap  # Return best assignment

# ──────────────────────────────────────────────────────────────
# 4. Main CLI (Command Line Interface)
# ──────────────────────────────────────────────────────────────

def main(argv: List[str] | None = None):
    """Main function to run the task assignment balancing."""
    
    p = argparse.ArgumentParser(description="Balance tasks using time, cost, or a weighted combo.")

    # Define arguments user can pass from terminal
    p.add_argument("--file", default=r"C:\Users\Akash\Desktop\efficient computers\archive (2)\task120.xlsx", help="Workbook path")
    p.add_argument("--objective", choices=["time", "cost", "combo"], default="time", help="Which gap to minimize")
    p.add_argument("--alpha", type=float, default=0.5, help="Weight for time in combo mode")
    p.add_argument("--beta", type=float, default=0.5, help="Weight for cost in combo mode")
    p.add_argument("--restarts", type=int, default=20, help="How many random restarts to try")

    args = p.parse_args(argv)  # Parse arguments

    xlsx = Path(args.file).expanduser().resolve()  # Get absolute file path
    if not xlsx.is_file():
        sys.exit(f"Workbook not found: {xlsx}")  # Exit if file is missing

    tasks, nodes, exec_time, cost = load_tables(xlsx)  # Load Excel tables

    # Select the scoring system based on objective
    if args.objective == "time":
        score = exec_time
        score_name = "execution time (s)"
    elif args.objective == "cost":
        score = cost
        score_name = "cost (currency units)"
    else:  # Weighted combination of time and cost
        alpha, beta = args.alpha, args.beta
        score = {t: {n: alpha * exec_time[t][n] + beta * cost[t][n] for n in nodes} for t in tasks}
        score_name = f"combo score (α={alpha}, β={beta})"

    t0 = time.time()  # Start timing
    assign, loads, gap = balance(tasks.copy(), nodes, score, restarts=args.restarts, seed=42)  # Balance tasks
    elapsed = time.time() - t0  # Time taken

    # Calculate total execution time and cost per node
    time_tot = {n: sum(exec_time[t][n] for t in assign[n]) for n in nodes}
    cost_tot = {n: sum(cost[t][n] for t in assign[n]) for n in nodes}

    # Print the result summary
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

    # ── Write results into Excel file ────────────────────────────────
    summary_rows = []
    for n in nodes:
        summary_rows.append({
            "Node": n,
            "Assigned tasks": ", ".join(assign[n]),
            "Total exec time (s)": time_tot[n],
            "Total cost": cost_tot[n],
            "Score total": loads[n],
        })

    summary_df = pd.DataFrame(summary_rows)  # Create summary dataframe

    outfile = xlsx.with_name("Task_Node_Assignment.xlsx")  # Set output file name
    with pd.ExcelWriter(outfile, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="NodeSummary", index=False)  # Write summary sheet

        # Write 0/1 assignment matrix
        matrix = pd.DataFrame(0, index=nodes, columns=tasks)
        for n in nodes:
            for t in assign[n]:
                matrix.at[n, t] = 1
        matrix.to_excel(writer, sheet_name="AssignmentMatrix")  # Write matrix sheet

    print(f"\nExcel report written to: {outfile}")  # Notify user

# Run main function if this script is executed
if __name__ == "__main__":
    main()
