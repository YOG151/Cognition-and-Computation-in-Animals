# Flocking and Predator Simulation

This repository contains a Python simulation and analysis of flocking behavior among agents and their interaction with a predator in a 2D environment using Pygame. The simulation explores how flocking behavior helps agents avoid predators and survive longer.

## Overview

The simulation involves the following main components:

- `Agent` class: Represents individual agents that exhibit flocking behavior, including alignment, cohesion, and separation. Agents also interact with a predator to avoid being caught.
- `Predator` class: Represents a predator that tries to catch agents by moving towards them. It interacts with agents and removes them upon catching.
- `sim` function: Simulates the flocking behavior and predator interaction for a specified number of rounds, collecting statistics such as predator catch rates, survival times, and average distances from the predator.
- `distance_analysis`, `catch_rate_analysis`, and `survival_time_analysis` functions: Analyze the collected statistics and visualize them using histograms, bar plots, and box plots.

## Prerequisites

- Python (>= 3.6)
- Pygame
- Numpy
- Pandas
- Seaborn
- Matplotlib

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/YOG151/flocking-predator-simulation.git

2. Navigate to the repository folder:
   ```bash
   cd flocking-predator-simulation
   
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the simulation and analysis script:
   ```bash
   python main.py

This will run the simulation and display various plots showcasing the impact of different traits on hive metrics.

# Read the final report for the full rundown on the logic of the simulations as well as the results and discussion.


   


