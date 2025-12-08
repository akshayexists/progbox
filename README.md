# ProgBox - NOEYETEST Simulation Framework

ProgBox is a Monte Carlo simulation tool designed to project player progression in Basketball GM (BBGM) leagues. It takes a league export, identifies eligible players, simulates their development over thousands of iterations, and provides detailed statistical analysis of their potential outcomes.

## 🚀 Quick Start

1. **Prepare Data**:
   - Place your BBGM export file as `data/export.json`.
   - Ensure `data/teaminfo.json` maps Team IDs to Abbreviations (e.g., `{"0": "ATL", "1": "BOS"}`).
2. **Run the Simulation**:

   ```bash
   python benchmark.py
   ```

   Runs complete simulation with visualizations and Excel analysis. Creates timestamped output folder: `outputs/run_YYYYMMDD_HHMMSS/`

   **Alternative**: Use `python main.py` to run simulation with output to `outputs/raw/` and `outputs/analysis.xlsx`.

3. **View Results**:
   - Check timestamped folder in `outputs/run_*/` for all results including `analysis.xlsx`, plots, and raw data.

---

## 🎯 Running Benchmarks & Visualizations

### Full Benchmark Run

Run complete simulation with all outputs and visualizations:

```bash
python benchmark.py
```

This creates a timestamped output folder (`outputs/run_YYYYMMDD_HHMMSS/`) containing:

- `analysis.xlsx`: Aggregated Excel report with player statistics
- `outputs.csv`: Full simulation results
- `metadata.json`: League information and simulation parameters (including master seed)
- `raw/`: Input data and special logs
- `plots/`: Generated visualizations (4 PNG files)

### Generate Visualizations Only

To create plots from existing simulation data:

```bash
python visualize.py <path_to_outputs.csv> [output_dir]
```

Example:

```bash
python visualize.py outputs/run_20250107_143022/outputs.csv outputs/run_20250107_143022
```

If `output_dir` is omitted, plots are saved to the same directory as the CSV.

### Individual Commands

- **Simulation only**: Modify `benchmark.py` to comment out `create_plots()` and `generate_excel_analysis()` calls
- **Visualization only**: Use `visualize.py` with a CSV path (see above)
- **Alternative entry point**: Use `main.py` for simulation with fixed output paths (`outputs/raw/`, `outputs/analysis.xlsx`)

---

## ⚙️ Configuration

You can tweak the simulation parameters in `main.py` or `benchmark.py`:

- **`teams`**: List of team abbreviations to filter by (e.g., `['GSW', 'BOS']`). Leave empty `[]` to scan all teams.
- **`seed`**: Master seed for reproducibility (default: `69`).
- **`runs`**: Number of Monte Carlo iterations per player (default: `1000`).

### Reproducibility with Seeds

The simulation uses a **master seed** to ensure reproducible results. When you run a simulation with the same seed, you will get identical results across runs. This is essential for:
- Comparing different simulation parameters
- Debugging and validating changes
- Sharing and reproducing specific simulation outcomes

**How it works:**
- The master seed initializes the random number generator
- Each simulation run receives a derived seed generated from the master seed
- All random operations (progression rolls, attribute changes, etc.) are deterministic based on these seeds

**How to modify the master seed:**

1. **In `benchmark.py`** (recommended for full benchmark runs):
   ```python
   def run_benchmark(..., seed=69):  # Change 69 to your desired seed
   ```
   Or when calling:
   ```python
   run_benchmark(seed=12345)
   ```

2. **In `main.py`** (for simple runs):
   ```python
   seed = 69  # Change 69 to your desired seed
   ```

**Seed recording:**
The master seed used for each simulation is automatically recorded in `metadata.json` within the timestamped output directory (`outputs/run_YYYYMMDD_HHMMSS/metadata.json`). This allows you to verify which seed was used for any given simulation run.

---

## 📂 Project Structure

```text
progbox/
├── data/                  # Input data folder
│   ├── export.json        # BBGM League Export file
│   └── teaminfo.json      # Team ID to Abbreviation mapping
├── outputs/               # Simulation results
│   ├── analysis.xlsx      # Excel template (used by main.py, copied by benchmark.py)
│   ├── run_YYYYMMDD_HHMMSS/   # Timestamped benchmark runs
│   │   ├── analysis.xlsx      # Aggregated Excel report
│   │   ├── outputs.csv        # Full simulation results
│   │   ├── metadata.json      # League information and simulation parameters
│   │   ├── raw/               # Raw data files
│   │   │   ├── inputs.csv
│   │   │   ├── godprogs.json
│   │   │   └── superlucky.json
│   │   └── plots/             # Generated visualizations
│   │       ├── dist_ovr_change_by_age.png
│   │       ├── avg_progression_trend.png
│   │       ├── scatter_talent_vs_progression.png
│   │       └── box_talent_tier_progression.png
│   └── raw/               # Raw data (from main.py)
├── main.py                # Alternative entry point (fixed output paths)
├── benchmark.py            # Primary entry point (timestamped outputs + plots + Excel)
├── visualize.py            # Standalone visualization generator
├── runsim.py               # Core simulation logic (Monte Carlo wrapper)
├── exportcleaner.py        # Data parser for BBGM exports
└── progutils.py            # Progression logic/sandbox
```

---

## 📊 Data Formats

### 1. Inputs (`data/`)

- **`export.json`**: Standard Basketball GM league export file.
- **`teaminfo.json`**: A simple JSON object mapping numeric Team IDs (strings) to Team Abbreviations.

  ```json
  {
    "0": "ATL",
    "1": "BOS",
    ...
  }
  ```

### 2. Simulation Metadata (`metadata.json`)

Generated by `benchmark.py` in `outputs/run_*/metadata.json`. Contains league information and simulation parameters:

| Field | Description |
|-------|-------------|
| **league_name** | Name of the Basketball GM league |
| **season** | Season year |
| **starting_season** | Starting season identifier |
| **phase** | League phase number |
| **phase_text** | Description of league phase |
| **export_date** | Date of the export file |
| **master_seed** | Master seed used for this simulation (enables reproducibility) |
| **runs** | Number of Monte Carlo iterations per player |
| **timestamp** | Timestamp of the simulation run (format: `YYYYMMDD_HHMMSS`) |

### 3. Output Analysis (`analysis.xlsx`)

Generated by `benchmark.py` in `outputs/run_*/analysis.xlsx` (or `outputs/analysis.xlsx` when using `main.py`).

The `aggregated` sheet provides a statistical summary for each player:

| Column | Description |
|--------|-------------|
| **Name** | Player Name |
| **Team** | Team Abbreviation |
| **Age** | Player Age at start of sim |
| **Baseline** | Starting Overall (Ovr) rating |
| **Ovr** | **Mean** projected Overall rating after progression |
| **Delta** | Average change in Overall (+/-) |
| **PctChange** | Average percentage change in Overall |
| **Ovr_min/max** | Worst and Best case Overall observed |
| **Ovr_q10/25/75/90** | Percentiles for Overall (e.g., q90 = 90th percentile outcome) |
| **GodProg Average** | Average stats during "God Progression" events |
| **GodProg Chance** | Probability of a "God Progression" event |
| **GodProgCount** | Total number of God Progs observed for this player |
| **[Attributes]** | Mean projected values for specific stats (Spd, Jmp, 3Pt, etc.) |

### 4. Raw Output (`outputs/raw/outputs.csv`)

Contains the result of *every* single simulation run.

| Column | Description |
|--------|-------------|
| **Run** | Simulation run number (0 to `runs`-1) |
| **RunSeed** | Specific seed used for that run |
| **Name** | Player Name |
| **PlayerID** | Unique Player ID |
| **Baseline** | Starting Overall |
| **Value** | Simulated Overall for this run |
| **Delta** | Difference (Value - Baseline) |
| **AboveBaseline** | Boolean (True if player improved) |
| **[Attributes]** | Simulated values for all individual attributes (Hgt, Str, Spd, etc.) |

### 5. Benchmark Summary (`outputs/benchmark_summary.csv`)

Aggregated statistics grouped by age groups and talent tiers:

| Column | Description |
|--------|-------------|
| **Group** | Primary grouping: `Talent` or `Age_Group` |
| **SubGroup** | Specific category (e.g., `Talented`, `Average Talent`, `NET_Young`, `NET_Mid`, `NET_Old`) |
| **Count** | Number of simulation runs in this group |
| **Avg_Delta** | Average OVR change |
| **Median_Delta** | Median OVR change |
| **Std_Delta** | Standard deviation of OVR change |
| **Pct_Improvement** | Percentage of runs where player improved |
| **Pct_Regression** | Percentage of runs where player declined |
| **Pct_Static** | Percentage of runs with no change |
| **Avg_Pct_Change** | Average percentage change in OVR |

### 5. Visualizations (`outputs/run_*/plots/`)

Four PNG files analyzing progression patterns:

- **`dist_ovr_change_by_age.png`**: Box plot showing distribution of OVR change by age. Displays median, quartiles, and outliers for each age group.
- **`avg_progression_trend.png`**: Line plot with 95% confidence intervals showing average progression trend by age.
- **`scatter_talent_vs_progression.png`**: Scatter plot of baseline OVR (talent) vs. OVR change, colored by age. Shows relationship between starting talent and progression outcomes.
- **`box_talent_tier_progression.png`**: Box plot of progression distribution by talent tier (Low/Avg/Good/Star/God) and age. Shows how progression varies across talent levels.

### 7. Special Logs

- **`godprogs.json`**: Detailed log of every "God Progression" event (extremely high roll improvements), including the exact stats generated.
- **`superlucky.json`**: A summary count of how many times each player achieved a God Progression.
