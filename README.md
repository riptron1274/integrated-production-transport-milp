# Integrated Production–Transport–Rack Scheduling MILP

This project contains an optimization model for coordinating production, transportation, reusable racks, and installation activities.

The basic process is:

**Production → Rack Assignment → Transportation → Site Arrival → Installation → Rack Return**

The model is formulated as a Mixed-Integer Linear Programming (MILP) problem. It schedules each panel(product) while considering production capacity, truck availability, rack reuse, delivery timing, and installation requirements.

## Objective

The model minimizes a weighted combination of:

- Installation lateness
- Waiting time at the installation site
- Number of racks used
- Production makespan

## Constraints

The model includes:

- One production line
- Each panel is produced and transported once
- Limited number of trucks
- Reusable racks with return times
- Production must finish before transportation
- Installation can only start after the panel arrives
- Fixed installation sequence

## Data

The current version uses a small synthetic dataset with six panels(product).
## Output

Running the model generates:

- Detailed production and transportation schedule
- Optimization summary
- Schedule visualization
- Rack fleet sensitivity analysis

Results are saved automatically in the `model1_outputs` folder.

## Requirements

```bash
pip install numpy pandas scipy matplotlib
```

## Run

```bash
python production-transport-rack-model1.py
```

