# Monte Carlo Tree Search with Velocity Obstacles for Safe and Efficient Motion Planning in Dynamic Environments

This repository contains the code to reproduce the experiments described in our paper on Monte Carlo Tree Search (MCTS) with Velocity Obstacles for motion planning in dynamic environments.

## Prerequisites

- Python 3.10+ (code tested on Python 3.10.12)

## Installation

1. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the maps data from [this link](https://mega.nz/file/yltCxDJZ#RTxOtTxkCOzULBTDmTQrorpa6CCqDPjmCaMrwCfAGFg).

4. Extract the data into `bettergym/environments`. After extraction, you should have the following structure:
   ```
   bettergym/environments/fixed_obs/
   ├── intention
   └── trefoil
   ```

5. Modify the `run.py` file to set the number of parallel experiments you want to run (adjust the `max_processes` variable).

## Usage

1. Run the experiments:
   ```
   python run.py
   ```

2. The output of the experiments will be saved in the `debug` directory, including:
   - CSV files containing experiment data
   - GIFs showing the planner trajectory

## Results

The results of the experiments can be found in the `debug` directory. The CSV files contain detailed metrics, while the GIFs provide a visual representation of the planner's performance.