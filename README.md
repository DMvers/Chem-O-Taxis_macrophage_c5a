# Chem-O-Taxis: Simulating Self-Generated Gradient Chemotaxis

A hybrid agent-based simulation of macrophage chemotaxis in self-generated or imposed gradients of complement C5a, modelling cells migrating across the bridge of an Insall chamber.


## Overview

Cells are placed in a simulated Insall chamber filled with chemoattractant (C5a). Each cell independently senses the local attractant gradient, consumes attractant from its surroundings via Michaelis–Menten kinetics, and migrates up the resulting gradient. Attractant diffusion is solved on a 2D grid using a DuFort–Frankel finite difference scheme.

## Model Description

The simulation implements a hybrid agent-based / continuum model on a 2D grid representing the bridge of an Insall chamber:

**Attractant dynamics.** C5a concentration is represented on a grid and evolved using DuFort–Frankel diffusion (after a single Forward Euler initialisation step). Boundary conditions maintain a source of attractant at one end of the chamber and a calculated outer well value for the other end.

**Outer well depletion.** The model also tracks the attractant concentration in the simulated outer well, accounting for consumption by the cell population seeded there, which drives the formation of self-generated gradients along the bridge.

**Cell agents.** Each cell occupies a circular area on the grid. At every timestep, cells sense the attractant concentration across their occupied grid sites, compute a weighted-average direction toward higher attractant (with receptor-binding kinetics and configurable noise), blend this with their previous heading via a persistence parameter, and take a step. After sensing, each cell consumes attractant from its local environment using Michaelis–Menten kinetics, representing receptor-mediated endocytosis.


## Repository Structure

```
├── simulation_migration.py    # Main simulation entry point
├── cell.py                    # Cell agent: sensing, movement, division, collision
├── environment.py             # Ligand grid: diffusion, decay, consumption, production
├── mazelayouts.py             # Wall/boundary geometry definitions
├── collisionfunctions.py      # Cell–cell and cell–wall collision detection
├── diffusionfunctions.py      # Standalone diffusion solvers (Forward Euler & DuFort–Frankel)
├── datasaver.py               # Output: Apache Feather grids, cell positions, CSV tracking
├── run_experiments.sh          # Batch script to run all paper conditions (3 replicates each)
├── run_viz.r                  # R script to render simulation output as tracked-cell images
└── support.r                  # R helper functions for visualisation
```

## Requirements

### Simulation (Python ≥ 3.9)

- numpy
- numba
- matplotlib
- pyarrow

### Visualisation (R)

- arrow (for reading Feather files)
- colorspace

Install the Python dependencies into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy numba matplotlib pyarrow pandas
```

## Running the Simulation


The `run_experiments.sh` script launches all conditions used in the paper (3 replicates each, run in parallel):

```bash
bash run_experiments.sh
```
Each run creates a folder containing Feather files for the attractant grid and cell positions at regular intervals, plus a CSV file tracking all cell positions at every timestep.

## Output Format

Each run produces a folder containing:

- `attractantN` — Feather files with the flattened 2D attractant grid at timestep N
- `cellsN` — Feather files with cell x/y positions and IDs at timestep N
- `tracking.csv` — Comma-separated file with columns: timestep, cell ID, x position, y position (appended every timestep)

## Visualisation

After the simulation has completed, use the R scripts to generate tracked-cell images:

```r
source("run_viz.r")
```

This reads the Feather output files and produces PNG frames showing the attractant field overlaid with cell positions and migration tracks. The script iterates over all experimental conditions defined in `allexpnames` and renders each frame with fading track tails for visual clarity.



