# NoEyeTest, Monte Carlo Progression Simulator

A parallelised Monte Carlo engine for simulating and analysing player progression in [BasketballGM](https://zengm.com/basketball/). Runs hundreds or thousands of independent progression passes over a roster, then produces aggregated statistics and tuner-focused diagnostic charts.

The progression scripts are the following:
- [NET v3.2 (main branch)](https://github.com/fearandesire/NoEyeTest/blob/dev/src/NoEyeTest.js)
- []
- [modified for NET v4.1 (this branch)](https://github.com/shawnmalik1/NoEyeTest-v4/blob/main/noeyetest_progs_v4.js)

---

## Table of Contents for this branch and this branch only

- [Overview](#overview)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
  - [The Composite Score](#the-composite-score)
  - [The Progression Range](#the-progression-range)
  - [Physical Decline Gates](#physical-decline-gates)
  - [God Progressions](#god-progressions)
  - [OVR Calculation](#ovr-calculation)
- [Configuration Reference](#configuration-reference)
- [Setup](#setup)
- [Running a Simulation](#running-a-simulation)
- [Output Files](#output-files)
- [Analysis Charts](#analysis-charts)
- [Module Reference](#module-reference)

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
│       ├── charts/          # Seven diagnostic PNG charts
│       └── analysis.xlsx    # Aggregated workbook
│
├── exportcleaner.py         # Parses and normalises ZengM export
├── progutils.py             # Core progression engine + Config
├── runsim.py                # Monte Carlo harness (parallel)
├── analysis.py              # Aggregated stats + chart generation
└── main.py                  # Entry point
```

---

## How It Works (from shawn's NETv4.1)

### The Composite Score

Every player's progression potential is reduced to a single number before any dice are rolled:

```
composite = PER × 0.70 + DWS × 5.0 × 0.20 + EWA × 3.0 × 0.10
          = PER × 0.70 + DWS × 1.00 + EWA × 0.30
```

The scalars on DWS and EWA exist because those stats live on a different numerical range than PER. The effective weight of each stat in the final formula is approximately **PER 70%, DWS 20%, EWA 10%**. Chart 6 lets you verify whether those weights are producing their intended relative influence in practice.

### The Progression Range

The composite score maps to a `[lo, hi]` delta range. Every progressing attribute then rolls a random integer in that range. The mapping differs by age group:

| Age group | Parameters | Hard max per-attr gain |
|-----------|-----------|------------------------|
| 26–30 | `lo = ceil(score / 5) − 7`, `hi = ceil(score / 4) − 2` | +4 |
| 31–34 | `lo = ceil(score / 6) − 7`, `hi = ceil(score / 4) − 3` | +2 |
| 35+   | `lo = ceil(score / 6) − 9`, `hi = 0` (never improves)  | 0  |

**Low-composite special case** (score ≤ 20 and age < 31): a wider, more volatile range is used: `lo = ceil(score/5) − 6`, `hi = ceil(score/4) − 1`. This gives young underperformers more variance.

**OVR hard cap**: no player may progress past OVR 80. If the range would breach the cap it is clipped. Players already at 80 receive `hi = 0` and an age-appropriate decline floor.

### Physical Decline Gates

v4.1 adds extra age-based restrictions on the four physical attributes `Spd`, `Str`, `Jmp`, `End`.

**Age 30+** (`OLD_AGE_PHYS`): when `hi > 0` (would-be improvement), a random check fires with approximately a 3.5% pass rate. If the check fails the attribute is left unchanged. When `hi ≤ 0` (decline path), the roll applies directly with no gate. Gains that do pass are capped at +3.

**Age 26–29** (`MID_AGE_PHYS`): the delta rolls normally, but any positive result is then subject to a linear fade -- 70% chance at age 26 dropping to 40% at age 29. If the fade check fails the attribute is left unchanged.

Chart 3 directly validates whether these gates are separating physical attrs from skill attrs across age groups.

### God Progressions

Once per player per run, a flat **2% chance** fires before the normal range is calculated. Eligibility: age < 30, OVR < 60. If triggered, `lo = hi = bonus` where bonus is a uniform draw from `[7, 10]`. The normal range is completely overridden for that run.

God progression events are logged to `godprogs.json` and `superlucky.json`, and appear as a separate sheet in `analysis.xlsx` if any fired.

---

## Configuration Reference

All tunable parameters live in `Config` inside `progutils.py`.

### Age Group Progression Parameters

```python
Config.AGE_GROUP_CONFIG = {
    '26-30': dict(min1=5, min2=7, max1=4, max2=2, hard_max=4),
    '31-34': dict(min1=6, min2=7, max1=4, max2=3, hard_max=2),
    '35+':   dict(min1=6, min2=9, max1=None, max2=None, hard_max=0),
}
```

- `min1`, `min2` - control the floor of the range: `lo = ceil(score / min1) − min2`
- `max1`, `max2` - control the ceiling: `hi = ceil(score / max1) − max2` (`None` = no upside)
- `hard_max`     - absolute per-attribute gain cap for this age group

Increasing `hard_max` allows older players to improve more. Lowering `min2` makes decline steeper. The 35+ group has `hard_max=0` and no upside formula, it can only produce flat or negative deltas.

### Composite Score Weights

```python
Config.COMPOSITE = dict(per_w=0.70, dws_scale=5.0, dws_w=0.20,
                        ewa_scale=3.0, ewa_w=0.10)
```

Effective weights: `composite = PER×per_w + DWS×(dws_scale×dws_w) + EWA×(ewa_scale×ewa_w)`. Chart 6 shows whether the actual influence of each stat in simulation outcomes matches these theoretical shares.

### God Progression

```python
Config.GOD_PROG = dict(max_chance=0.02, age_limit=30, ovr_limit=60,
                       bonus_min=7, bonus_max=10)
```

- `max_chance` - flat probability per eligible player per run (0.02 = 2%)
- `age_limit`  - players at this age or older are ineligible
- `ovr_limit`  - players at this OVR or above are ineligible
- `bonus_min`, `bonus_max` - uniform range for the god-prog delta

### Physical Decline Sets

```python
Config.OLD_AGE_PHYS = frozenset({'Spd', 'Str', 'Jmp', 'End'})  # gates at 30+
Config.MID_AGE_PHYS = frozenset({'Spd', 'Str', 'Jmp'})          # fade at 26–29
```

`End` is in `OLD_AGE_PHYS` but not `MID_AGE_PHYS`, endurance fades later than the pure physical trio.

### Other

```python
Config.OVR_HARD_CAP = 80   # no player may progress above this OVR
Config.MIN_AGE      = 26   # players younger than this are skipped entirely
```

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

| Sheet | Contents |
|-------|----------|
| `Aggregated` | Per-player: baseline, mean/std/min/max OVR, quartiles Q10/25/75/90, % runs positive, composite score, percentile rank |
| `AgeGroups` | Delta statistics (mean, median, std, IQR, % positive) split by 26-30 / 31-34 / 35+ |
| `StatDrivers` | Per-player averaged PER / DWS / EWA / composite score alongside mean delta |
| `Correlations` | Pearson r and p-value for each stat vs mean delta |
| `AttrSensitivity` | Per-attribute std across runs + coefficient of variation + physical flag |
| `GodProgs` | Only present if god progressions occurred |

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