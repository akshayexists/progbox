# NoEyeTest, Monte Carlo Progression Simulator

A parallelised Monte Carlo engine for simulating and analysing player progression in [BasketballGM](https://zengm.com/basketball/). Runs hundreds or thousands of independent progression passes over a roster, then produces aggregated statistics and tuner-focused diagnostic charts.

The progression scripts are the following:
- [NET v3.2 (main branch)](https://github.com/fearandesire/NoEyeTest/blob/dev/src/NoEyeTest.js)
- [NET v4](https://gist.github.com/fearandesire/fa7ddef9be41be66e1b9639b51bb88d6) (dev branch, slightly older code) 
- [modified for NET v4.1 (this branch)](https://github.com/shawnmalik1/NoEyeTest-v4/blob/main/noeyetest_progs_v4.js)

---

## Overview

Given a roster export, it runs N fully independent progression passes (each with its own RNG seed), then aggregates the results into per-player outcome distributions. This is useful. Well, should be.

---

## Project Structure

```
.
├── data/
│   ├── export.json          # BBGM league export
│   └── teaminfo.json        # Team metadata
│
├── outputs/
│   └── {RUN_TS}/            # One folder per run, named by timestamp
│       ├── raw/
│       │   ├── outputs.csv  # Full long-format simulation results
│       │   ├── godprogs.json
│       │   └── superlucky.json
│       ├── charts/          # diagnostic PNG charts
│       └── analysis.xlsx    # Aggregated workbook
│
├── exportcleaner.py         # Parses and normalises BBGM export
├── progutils.py             # Core progression engine + Config <----- pls modify
├── runsim.py                # Monte Carlo harness <------------------ do not modify
├── analysis.py              # Aggregated stats + chart generation <-- modify if you modify progutils to use a different prog script
└── main.py                  # Entry point <-------------------------- for setting sim settings
```

---

## Configuration Reference

All tunable parameters live in `Config` inside `progutils.py`. 
If you wish to change the prog script, you may do so, just ensure ProgressionSandbox class and its API hygine is maintained.

---

## Setup

```bash
pip install numpy pandas scipy matplotlib tqdm openpyxl
```

No other dependencies beyond the standard library. Python 3.10+ recommended.

Place your league export at `data/export.json` and team metadata at `data/teaminfo.json`.

---

## Running a Simulation
all you have to change is the config dict.

```python
    config = {
        "teams": [], # [] = all teams; e.g. ['GSW', 'BOS'] for specific teams
        "seed": 69,
        "runs": 1000,
        "n_workers": 12,
        "export_file": 'data/export.json',
        "teaminfo_file": 'data/teaminfo.json'
    }
```

**`n_workers`** defaults to `os.cpu_count()`. Set it to `1` to run single-process for debugging. On a 12-core machine, 1000 runs over a full 30-player roster typically completes in under a minute.

**`seed`** controls the master RNG. The master RNG derives each run's seed, so the same `seed` always produces the same set of simulations regardless of `n_workers`.

---

## Output Files

### `outputs/{RUN_TS}/raw/outputs.csv`

Long-format table w one row per player × run. Key columns:

| Column | Description |
|--------|-------------|
| `Run` | Run index (0 to N−1) |
| `RunSeed` | The RNG seed used for this run |
| `Name`, `Team`, `Age` | Player metadata |
| `PlayerID` | Stable identifier matching the input DataFrame index |
| `Baseline` | Player's OVR before any progression |
| `Ovr` | Simulated OVR after progression |
| `Delta` | `Ovr − Baseline` |
| `PctChange` | `Delta / Baseline` |
| `AboveBaseline` | Boolean, did this run produce a net gain? |
| `PER`, `DWS`, `EWA` | Input stats used to compute composite score |
| `dIQ` … `3Pt` | Final simulated attribute values |

### `outputs/{RUN_TS}/raw/godprogs.json`

Array of every god-progression event across all runs. Fields: `name`, `run_seed`, `age`, `ovr`, `bonus`, `chance`.

### `outputs/{RUN_TS}/raw/superlucky.json`

Object mapping player name → total god-prog count across all runs.

### `outputs/{RUN_TS}/analysis.xlsx`

---

## Analysis Charts

All thresholds, splits, and reference lines are derived from the dataset. Nothing is hard-coded.

### Chart 1 - Age Outcome Profiles
**Question: Are the age tiers producing meaningfully different outcome shapes?**

Three overlaid KDE density curves, one per age group, over the same x-axis. If the curves heavily overlap, the age transition has no real bite. A flat, wide curve means that age group's outcomes are dominated by RNG.

### Chart 2 - Progression Response Curve
**Question: What does the composite score → OVR delta mapping actually look like?**

Players are bucketed into equal-frequency composite score bins (deciles by default). Each bin shows mean delta as a bar, with IQR as error bars. A flat section is a dead zone where composite score changes do nothing. A smooth upward ramp is ideal.

### Chart 3 - Physical vs Skill Gate Check
**Question: Are the v4.1 age-gated physical decline rules doing anything beyond ordinary age decay?**

Grouped bars: for each age group, one bar for the mean change in physical attributes (Spd, Str, Jmp, End) and one for skill attributes. Physical bars should decline meaningfully faster than skill bars as age rises.

### Chart 4 - Attribute Delta Heatmap
**Question: What is every attribute doing, per age group, at a glance?**

A 14-row × 3-column grid. Rows are the 14 progressing attributes; columns are the three age groups. Cell colour: red = net decline, white = flat, green = net gain.

### Chart 5 - Player Outcome Certainty
**Question: Which players have settled outcomes vs RNG-dominated outcomes?**

One horizontal row per player, sorted by median delta. A dot marks the median; a line spans Q10 to Q90. Short lines mean the engine converges consistently on a result for that player, and their stat profile locks them in. Long lines mean the outcome is a lottery.

### Chart 6 - Stat Weight Reality Check
**Question: Are PER/DWS/EWA contributing to outcomes in proportion to their design weights?**

Two bars per stat: theoretical weight from `Config.COMPOSITE` (normalised to 100%) vs actual partial R² share (how much explained variance is lost when that stat is removed from an OLS regression, normalised to 100%). If the bars diverge, a stat is over- or under-driving outcomes relative to its design intent.

### Chart 7 - Age × Performance Tuning Matrix
**Question: How does performance level interact with age tier across all combinations?**

A 3×3 heatmap. Columns are age groups. Rows are performance tiers (Low / Mid / High), defined as composite score tertiles *within each age group*. Each cell shows mean OVR delta and player count.

---

## Module Reference

### `progutils.py`
The core engine. Key public symbols:

| Symbol | Description |
|--------|-------------|
| `Config` | All tunable parameters |
| `calcovr(attrs_dict)` | OVR from an attribute dictionary |
| `calcovr_from_array(attrs_array)` | OVR from an ALL_ATTRS-ordered numpy array (fast path) |
| `get_prog_range(per, dws, ewa, ovr, age)` | Returns `(lo, hi)` for a player |
| `progplayer(row, rng)` | Progresses one player row in-place; returns `(row, god_info)` |
| `ProgressionSandbox` | Owns a tracker; applies one full roster pass via `progress_roster()` |
| `ProgressionTracker` | Accumulates god-prog events |
| `SimAnalytics` | Built-in analytics; call `.print_report()` for a console summary |

this is all that needs to change for any new prog script!!

### `runsim.py`
Monte Carlo harness. Inherits `ProgressionSandbox`.

| Method | Description |
|--------|-------------|
| `runsim(seed)` | Constructor. Sets master RNG seed. |
| `.PROGEMUP(df, runs, output_dir, n_workers)` | Runs N parallel simulations. Returns long-format DataFrame. Populates `.analytics`. |
| `.analytics` | `SimAnalytics` instance populated after `PROGEMUP()` completes. |

Worker processes are initialised once per process via `_worker_init`, and the input DataFrame is serialised once, not once per task. Results are streamed back via `imap_unordered` and sorted by run index before assembly.
Parallelism is new.

### `analysis.py`

| Function | Description |
|----------|-------------|
| `generate_analysis(analysis_path)` | Main entry point. Reads `raw/outputs.csv`, writes `analysis.xlsx` and `charts/*.png`. |

---
