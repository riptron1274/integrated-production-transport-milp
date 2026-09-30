#!/usr/bin/env python3
"""
Model 1 — Deterministic Integrated Production–Transport–Rack–Site MILP
======================================================================

This is a standalone research prototype for the BYWall-style problem:

Production -> Rack assignment -> Transportation -> Site arrival -> Installation
                                      |
                                      v
                               Rack return/reuse

Main assumptions
----------------
1. One production line.
2. Each panel is produced exactly once.
3. Each panel is shipped exactly once.
4. One panel occupies one reusable rack per delivery cycle.
5. Racks are identical.
6. Trucks are identical.
7. A rack/truck is occupied during outbound travel + unloading + return.
8. Travel/production times are deterministic.
9. Installation cannot occur before panel arrival.
10. Installation cannot occur before the target site time.
11. Panels follow a fixed installation sequence.
12. Time is discretized.

Objective
---------
Minimize a weighted sum of:
- installation lateness,
- site waiting / early delivery,
- number of racks used,
- production makespan.

Dependencies
------------
pip install numpy pandas scipy matplotlib

Run
---
python model1_bywall_milp.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csc_matrix


# ============================================================
# 1. DATA
# ============================================================

@dataclass
class ProblemData:
    panels: List[str]

    # Production
    production_time: np.ndarray

    # Site demand
    target_install: np.ndarray

    # Logistics
    travel_out: np.ndarray
    unload_time: np.ndarray
    travel_back: np.ndarray

    # Resources
    num_candidate_racks: int
    num_trucks: int

    # Planning
    horizon: int
    install_gap: int = 1

    @property
    def n(self) -> int:
        return len(self.panels)

    @property
    def rack_cycle(self) -> np.ndarray:
        """
        Full time a rack/truck is unavailable after dispatch.
        """
        return self.travel_out + self.unload_time + self.travel_back


@dataclass
class ObjectiveWeights:
    lateness: float = 100.0
    site_wait: float = 1.0
    racks: float = 5.0
    makespan: float = 0.1


def create_example_instance() -> ProblemData:
    """
    Synthetic example only.
    Replace these values with real industrial data later.
    """
    return ProblemData(
        panels=["P1", "P2", "P3", "P4", "P5", "P6"],

        production_time=np.array(
            [3, 2, 3, 2, 2, 3],
            dtype=int
        ),

        target_install=np.array(
            [8, 10, 12, 14, 16, 18],
            dtype=int
        ),

        travel_out=np.array(
            [2, 2, 2, 2, 2, 2],
            dtype=int
        ),

        unload_time=np.array(
            [1, 1, 1, 1, 1, 1],
            dtype=int
        ),

        travel_back=np.array(
            [2, 2, 2, 2, 2, 2],
            dtype=int
        ),

        num_candidate_racks=4,
        num_trucks=2,

        horizon=50,
        install_gap=1,
    )


# ============================================================
# 2. MILP MODEL
# ============================================================

def solve_model(
    data: ProblemData,
    weights: ObjectiveWeights,
    max_racks: Optional[int] = None,
    time_limit: float = 30.0,
) -> Tuple[object, Optional[pd.DataFrame], Optional[Dict[str, float]]]:

    n = data.n
    R = data.num_candidate_racks
    T = list(range(data.horizon + 1))
    cycle = data.rack_cycle

    # --------------------------------------------------------
    # Variable indexing
    # --------------------------------------------------------
    names: List[str] = []

    def add_vars(prefix: str, count: int) -> np.ndarray:
        start = len(names)
        names.extend([f"{prefix}{k}" for k in range(count)])
        return np.arange(start, start + count)

    # x[i,t] = 1 if panel i starts production at time t
    x = np.empty((n, len(T)), dtype=int)
    for i in range(n):
        x[i, :] = add_vars(f"x_{i}_", len(T))

    # y[i,r,t] = 1 if panel i departs at time t using rack r
    y = np.empty((n, R, len(T)), dtype=int)
    for i in range(n):
        for r in range(R):
            y[i, r, :] = add_vars(f"y_{i}_{r}_", len(T))

    # u[r] = 1 if rack r is used at least once
    u = add_vars("u_", R)

    # Continuous variables
    install = add_vars("install_", n)
    lateness = add_vars("lateness_", n)
    site_wait = add_vars("site_wait_", n)
    cmax = add_vars("cmax_", 1)[0]

    N = len(names)

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------
    c = np.zeros(N)

    c[lateness] = weights.lateness
    c[site_wait] = weights.site_wait
    c[u] = weights.racks
    c[cmax] = weights.makespan

    # --------------------------------------------------------
    # Bounds and integrality
    # --------------------------------------------------------
    lb = np.zeros(N)
    ub = np.full(N, np.inf)
    integrality = np.zeros(N, dtype=int)

    # Binary variables
    integrality[x.ravel()] = 1
    ub[x.ravel()] = 1

    integrality[y.ravel()] = 1
    ub[y.ravel()] = 1

    integrality[u] = 1
    ub[u] = 1

    # Prevent starts/departures that exceed horizon
    for i in range(n):
        for ti, t in enumerate(T):

            if t + data.production_time[i] > data.horizon:
                ub[x[i, ti]] = 0

            if t + cycle[i] > data.horizon:
                for r in range(R):
                    ub[y[i, r, ti]] = 0

    # --------------------------------------------------------
    # Constraint builder
    # --------------------------------------------------------
    rows = []
    lows = []
    highs = []

    def add_constraint(
        coefficients: Dict[int, float],
        lower: float = -np.inf,
        upper: float = np.inf,
    ):
        rows.append(coefficients)
        lows.append(lower)
        highs.append(upper)

    # ========================================================
    # CONSTRAINT 1
    # Each panel starts production exactly once
    #
    # sum_t x[i,t] = 1
    # ========================================================
    for i in range(n):
        add_constraint(
            {x[i, ti]: 1.0 for ti in range(len(T))},
            lower=1,
            upper=1,
        )

    # ========================================================
    # CONSTRAINT 2
    # Each panel is dispatched exactly once using one rack
    #
    # sum_r sum_t y[i,r,t] = 1
    # ========================================================
    for i in range(n):
        coeff = {}

        for r in range(R):
            for ti in range(len(T)):
                coeff[y[i, r, ti]] = 1.0

        add_constraint(
            coeff,
            lower=1,
            upper=1,
        )

    # ========================================================
    # CONSTRAINT 3
    # Single production-line capacity
    #
    # At every time tau:
    # sum of active production operations <= 1
    # ========================================================
    for tau in T:
        coeff = {}

        for i in range(n):
            p_i = int(data.production_time[i])

            for ti, t in enumerate(T):
                if t <= tau < t + p_i:
                    coeff[x[i, ti]] = coeff.get(x[i, ti], 0.0) + 1.0

        add_constraint(
            coeff,
            upper=1,
        )

    # ========================================================
    # CONSTRAINT 4
    # Production must finish before shipment
    #
    # departure_i >= completion_i
    #
    # sum_r sum_t t*y[i,r,t]
    # >=
    # sum_t (t+p_i)*x[i,t]
    # ========================================================
    for i in range(n):
        coeff = {}

        # Departure time
        for r in range(R):
            for ti, t in enumerate(T):
                coeff[y[i, r, ti]] = coeff.get(y[i, r, ti], 0.0) + t

        # Production completion time
        for ti, t in enumerate(T):
            coeff[x[i, ti]] = (
                coeff.get(x[i, ti], 0.0)
                - (t + data.production_time[i])
            )

        add_constraint(
            coeff,
            lower=0,
        )

    # ========================================================
    # CONSTRAINT 5
    # Rack activation
    #
    # y[i,r,t] <= u[r]
    # ========================================================
    for i in range(n):
        for r in range(R):
            for ti in range(len(T)):
                add_constraint(
                    {
                        y[i, r, ti]: 1.0,
                        u[r]: -1.0,
                    },
                    upper=0,
                )

    # Symmetry breaking:
    # lower-index rack is activated before higher-index rack
    for r in range(R - 1):
        add_constraint(
            {
                u[r]: 1.0,
                u[r + 1]: -1.0,
            },
            lower=0,
        )

    # Optional maximum number of racks
    if max_racks is not None:
        add_constraint(
            {u[r]: 1.0 for r in range(R)},
            upper=max_racks,
        )

    # ========================================================
    # CONSTRAINT 6
    # Rack non-overlap / rack circulation
    #
    # A rack cannot be used by two active trips simultaneously.
    #
    # Rack occupied over:
    # [departure, departure + cycle_time)
    # ========================================================
    for r in range(R):
        for tau in T:
            coeff = {}

            for i in range(n):
                c_i = int(cycle[i])

                for ti, t in enumerate(T):
                    if t <= tau < t + c_i:
                        coeff[y[i, r, ti]] = (
                            coeff.get(y[i, r, ti], 0.0) + 1.0
                        )

            add_constraint(
                coeff,
                upper=1,
            )

    # ========================================================
    # CONSTRAINT 7
    # Truck fleet capacity
    #
    # Number of active deliveries at any time <= number of trucks
    # ========================================================
    for tau in T:
        coeff = {}

        for i in range(n):
            c_i = int(cycle[i])

            for r in range(R):
                for ti, t in enumerate(T):

                    if t <= tau < t + c_i:
                        coeff[y[i, r, ti]] = (
                            coeff.get(y[i, r, ti], 0.0) + 1.0
                        )

        add_constraint(
            coeff,
            upper=data.num_trucks,
        )

    # ========================================================
    # CONSTRAINT 8
    # Installation cannot occur before arrival
    #
    # install_i >= departure_i + travel_out_i
    # ========================================================
    for i in range(n):
        coeff = {
            install[i]: 1.0
        }

        for r in range(R):
            for ti, t in enumerate(T):
                coeff[y[i, r, ti]] = (
                    coeff.get(y[i, r, ti], 0.0)
                    - (t + data.travel_out[i])
                )

        add_constraint(
            coeff,
            lower=0,
        )

    # ========================================================
    # CONSTRAINT 9
    # Installation cannot occur before target time
    #
    # install_i >= d_i
    # ========================================================
    for i in range(n):
        add_constraint(
            {install[i]: 1.0},
            lower=data.target_install[i],
        )

    # ========================================================
    # CONSTRAINT 10
    # Lateness
    #
    # lateness_i >= install_i - target_i
    # ========================================================
    for i in range(n):
        add_constraint(
            {
                lateness[i]: 1.0,
                install[i]: -1.0,
            },
            lower=-data.target_install[i],
        )

    # ========================================================
    # CONSTRAINT 11
    # Site waiting
    #
    # wait_i >= install_i - arrival_i
    # ========================================================
    for i in range(n):
        coeff = {
            site_wait[i]: 1.0,
            install[i]: -1.0,
        }

        for r in range(R):
            for ti, t in enumerate(T):
                coeff[y[i, r, ti]] = (
                    coeff.get(y[i, r, ti], 0.0)
                    + (t + data.travel_out[i])
                )

        add_constraint(
            coeff,
            lower=0,
        )

    # ========================================================
    # CONSTRAINT 12
    # Fixed installation sequence
    #
    # install[i+1] >= install[i] + gap
    # ========================================================
    for i in range(n - 1):
        add_constraint(
            {
                install[i + 1]: 1.0,
                install[i]: -1.0,
            },
            lower=data.install_gap,
        )

    # ========================================================
    # CONSTRAINT 13
    # Production makespan
    #
    # Cmax >= completion_i
    # ========================================================
    for i in range(n):
        coeff = {
            cmax: 1.0
        }

        for ti, t in enumerate(T):
            coeff[x[i, ti]] = (
                coeff.get(x[i, ti], 0.0)
                - (t + data.production_time[i])
            )

        add_constraint(
            coeff,
            lower=0,
        )

    # --------------------------------------------------------
    # Build sparse constraint matrix
    # --------------------------------------------------------
    A = lil_matrix((len(rows), N))

    for row_idx, coeffs in enumerate(rows):
        for var_idx, value in coeffs.items():
            A[row_idx, var_idx] = value

    A = csc_matrix(A)

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------
    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(
            A,
            np.asarray(lows),
            np.asarray(highs),
        ),
        options={
            "time_limit": time_limit,
        },
    )

    if not result.success:
        return result, None, None

    solution = result.x

    # --------------------------------------------------------
    # Extract solution
    # --------------------------------------------------------
    records = []

    for i, panel in enumerate(data.panels):

        # Production start
        production_start = next(
            t
            for ti, t in enumerate(T)
            if solution[x[i, ti]] > 0.5
        )

        # Rack and departure time
        rack_id = None
        departure = None

        for r in range(R):
            for ti, t in enumerate(T):

                if solution[y[i, r, ti]] > 0.5:
                    rack_id = r + 1
                    departure = t
                    break

            if rack_id is not None:
                break

        arrival = departure + int(data.travel_out[i])

        rack_return = (
            departure
            + int(data.rack_cycle[i])
        )

        records.append({
            "Panel": panel,

            "Production start": int(production_start),

            "Production finish":
                int(production_start + data.production_time[i]),

            "Rack": int(rack_id),

            "Departure": int(departure),

            "Arrival": int(arrival),

            "Target install":
                int(data.target_install[i]),

            "Actual install":
                round(float(solution[install[i]]), 3),

            "Site waiting":
                round(float(solution[site_wait[i]]), 3),

            "Lateness":
                round(float(solution[lateness[i]]), 3),

            "Rack return":
                int(rack_return),
        })

    schedule = pd.DataFrame(records)

    summary = {
        "Status": result.message,
        "Objective value":
            round(float(result.fun), 3),

        "Racks used":
            int(round(solution[u].sum())),

        "Total lateness":
            round(float(solution[lateness].sum()), 3),

        "Total site waiting":
            round(float(solution[site_wait].sum()), 3),

        "Production makespan":
            round(float(solution[cmax]), 3),
    }

    return result, schedule, summary


# ============================================================
# 3. PLOT
# ============================================================

def plot_schedule(
    schedule: pd.DataFrame,
    output_path: Path
):
    """
    Simple visual:
    - thick bars = production
    - thin bars = outbound transport
    - x marker = installation
    """

    fig, ax = plt.subplots(figsize=(11, 5))

    for row_idx, row in schedule.iterrows():

        # Production
        ax.barh(
            row_idx,
            row["Production finish"]
            - row["Production start"],
            left=row["Production start"],
            height=0.30,
        )

        # Transportation
        ax.barh(
            row_idx,
            row["Arrival"] - row["Departure"],
            left=row["Departure"],
            height=0.16,
        )

        # Installation
        ax.scatter(
            row["Actual install"],
            row_idx,
            marker="x",
            s=60,
        )

    ax.set_yticks(range(len(schedule)))
    ax.set_yticklabels(schedule["Panel"])

    ax.set_xlabel("Time")
    ax.set_ylabel("Panel")

    ax.set_title(
        "Integrated production–transport–installation schedule"
    )

    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 4. EXPERIMENT: RACK FLEET SIZE
# ============================================================

def rack_sensitivity_experiment(
    data: ProblemData,
    weights: ObjectiveWeights,
    output_dir: Path,
) -> pd.DataFrame:

    rows = []

    for rack_limit in range(
        1,
        data.num_candidate_racks + 1
    ):

        result, schedule, summary = solve_model(
            data=data,
            weights=weights,
            max_racks=rack_limit,
        )

        if schedule is None:
            rows.append({
                "Rack limit": rack_limit,
                "Feasible": False,
                "Racks used": np.nan,
                "Total lateness": np.nan,
                "Total site waiting": np.nan,
                "Production makespan": np.nan,
                "Objective value": np.nan,
            })

        else:
            rows.append({
                "Rack limit": rack_limit,
                "Feasible": True,
                "Racks used":
                    summary["Racks used"],

                "Total lateness":
                    summary["Total lateness"],

                "Total site waiting":
                    summary["Total site waiting"],

                "Production makespan":
                    summary["Production makespan"],

                "Objective value":
                    summary["Objective value"],
            })

            schedule.to_csv(
                output_dir
                / f"schedule_rack_limit_{rack_limit}.csv",
                index=False,
            )

    df = pd.DataFrame(rows)

    df.to_csv(
        output_dir / "rack_sensitivity.csv",
        index=False,
    )

    return df


# ============================================================
# 5. MAIN
# ============================================================

def main():

    output_dir = Path("model1_outputs")
    output_dir.mkdir(exist_ok=True)

    data = create_example_instance()

    weights = ObjectiveWeights(
        lateness=100.0,
        site_wait=1.0,
        racks=5.0,
        makespan=0.1,
    )

    print("=" * 70)
    print("MODEL 1")
    print("Deterministic Integrated Production–Transport–Rack–Site MILP")
    print("=" * 70)

    result, schedule, summary = solve_model(
        data=data,
        weights=weights,
    )

    if schedule is None:
        print("No feasible solution.")
        print(result.message)
        return

    print("\nSUMMARY")
    print("-" * 70)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nDETAILED SCHEDULE")
    print("-" * 70)

    print(
        schedule.to_string(
            index=False
        )
    )

    # Save main schedule
    schedule.to_csv(
        output_dir / "model1_schedule.csv",
        index=False,
    )

    # Save summary
    pd.DataFrame(
        [summary]
    ).to_csv(
        output_dir / "model1_summary.csv",
        index=False,
    )

    # Plot
    plot_schedule(
        schedule,
        output_dir / "model1_schedule.png",
    )

    # Rack fleet sensitivity
    print("\nRACK SENSITIVITY")
    print("-" * 70)

    rack_results = rack_sensitivity_experiment(
        data=data,
        weights=weights,
        output_dir=output_dir,
    )

    print(
        rack_results.to_string(
            index=False
        )
    )

    print(
        f"\nFiles saved in: "
        f"{output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
