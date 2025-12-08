# ProgBox - [NoEyeTest](https://github.com/fearandesire/NoEyeTest) Simulation Framework

Monte Carlo simulation tool for projecting player progression in Basketball GM leagues. Simulates player development over thousands of iterations and provides statistical analysis.

## Quick Start

1. **Setup**:
   - Place BBGM export as `data/export.json`
   - Add team mappings in `data/teaminfo.json` (format: `{"0": "ATL", "1": "BOS"}`)

2. **Run**:

   ```bash
   python benchmark.py
   ```

   Creates timestamped folder: `outputs/run_YYYYMMDD_HHMMSS/` with Excel analysis, plots, and raw data.

   **Alternative**: `python main.py` outputs to fixed paths (`outputs/raw/`, `outputs/analysis.xlsx`).

3. **Results**: Check `outputs/run_*/` for:
   - `analysis.xlsx`: Aggregated player statistics
   - `outputs.csv`: Full simulation results
   - `metadata.json`: League info and simulation parameters (includes master seed)
   - `plots/`: 4 visualization PNGs
   - `raw/`: Input data and special logs

## Configuration

Edit `benchmark.py` or `main.py`:

- **`teams`**: Filter by team abbreviations (e.g., `['GSW', 'BOS']`). Empty `[]` = all teams.
- **`seed`**: Master seed for reproducibility (default: `69`).
- **`runs`**: Monte Carlo iterations per player (default: `1000`).

### Reproducibility

The master seed ensures identical results across runs. Change it in:

- `benchmark.py`: `run_benchmark(seed=12345)`
- `main.py`: `seed = 12345`

All seeds are recorded in `metadata.json` for verification.

## Commands

**Full benchmark** (recommended):

```bash
python benchmark.py
```

**Visualizations only** (from existing data):

```bash
python visualize.py <path_to_outputs.csv> [output_dir]
```

## Project Structure

```txt
progbox/
├── data/
│   ├── export.json          # BBGM league export
│   └── teaminfo.json        # Team ID to abbreviation map
├── outputs/
│   ├── run_YYYYMMDD_HHMMSS/ # Timestamped benchmark runs
│   │   ├── analysis.xlsx
│   │   ├── outputs.csv
│   │   ├── metadata.json
│   │   ├── plots/           # 4 PNG visualizations
│   │   └── raw/             # Input data + special logs
│   └── raw/                 # Fixed output (main.py only)
├── benchmark.py              # Primary entry (timestamped outputs)
├── main.py                   # Alternative entry (fixed paths)
├── visualize.py              # Standalone plot generator
├── runsim.py                 # Monte Carlo simulation logic
├── exportcleaner.py          # BBGM export parser
└── progutils.py              # Progression logic
```

## Data Formats

### Metadata (`metadata.json`)

| Field | Description |
|-------|-------------|
| league_name | BBGM league name |
| season | Season year |
| starting_season | Starting season ID |
| phase | League phase number |
| phase_text | Phase description |
| export_date | Export file date |
| master_seed | Simulation seed (reproducibility) |
| runs | Monte Carlo iterations per player |
| timestamp | Run timestamp (YYYYMMDD_HHMMSS) |

### Analysis (`analysis.xlsx`)

Aggregated statistics per player:

| Column | Description |
|--------|-------------|
| Name, Team, Age | Player identifiers |
| Baseline | Starting Overall rating |
| Ovr | Mean projected Overall |
| Delta | Average Overall change |
| PctChange | Average % change |
| Ovr_min/max | Best/worst outcomes |
| Ovr_q10/25/75/90 | Outcome percentiles |
| GodProg Average | Stats during god progressions |
| GodProg Chance | Probability of god progression |
| GodProgCount | Total god progressions observed |
| [Attributes] | Mean projected values (Spd, Jmp, 3Pt, etc.) |

### Raw Output (`outputs.csv`)

Every simulation run:

| Column | Description |
|--------|-------------|
| Run | Simulation iteration (0 to runs-1) |
| RunSeed | Seed for this run |
| Name, PlayerID | Player identifiers |
| Baseline | Starting Overall |
| Value | Simulated Overall |
| Delta | Change (Value - Baseline) |
| AboveBaseline | Boolean (improved?) |
| [Attributes] | Simulated attribute values |

### Benchmark Summary (`benchmark_summary.csv`)

Aggregated by age groups and talent tiers:

| Column | Description |
|--------|-------------|
| Group | `Talent` or `Age_Group` |
| SubGroup | Category (Talented, Average, NET_Young, etc.) |
| Count | Number of runs |
| Avg_Delta, Median_Delta | OVR change statistics |
| Std_Delta | Standard deviation |
| Pct_Improvement/Regression/Static | Outcome percentages |
| Avg_Pct_Change | Average % change |

### Visualizations (`plots/`)

- **`dist_ovr_change_by_age.png`**: Box plot of OVR change distribution by age (median, quartiles, outliers).
- **`avg_progression_trend.png`**: Line plot with 95% CI showing progression trend by age.
- **`scatter_talent_vs_progression.png`**: Baseline OVR vs. OVR change, colored by age.
- **`box_talent_tier_progression.png`**: Progression by talent tier (Low/Avg/Good/Star/God) and age.

### Special Logs

- **`godprogs.json`**: Detailed log of all god progression events.
- **`superlucky.json`**: Count of god progressions per player.

## NET Tier Compliance

Uses fixed hard bounds from `NoEyeTest/tiers.md`, independent of PER:

- 25-30: [-2, 4]
- 31-34: [-10, 2]
- 35+: [-14, 0]

Players 80+ OVR: hard max 0 with tier-specific hard mins (25-30: -2, 31-34: -10, 35+: -14).
