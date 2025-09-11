from progutils import progsandbox
import numpy as np, pandas as pd, random

class runsim(progsandbox):
	"""Monte Carlo wrapper producing a single analysis-ready CSV with cumulative Ovr stats.	"""

	def __init__(self, seed):
		if seed is None:
			raise ValueError("runsim requires a seed")
		self.SEED = seed
		super().__init__(seed=self.SEED)

		# numpy generator tied to the master seed (not using global np RNG)
		self.np_rng = np.random.default_rng(self.SEED)
		self.totalsimcounts = 0
		self._sim = None
		self._delta = None
		self._attrs = None
		self._meta = None
		self._baseline = None
		self.export_ = pd.DataFrame()
		self.gpc = 0

	def _set_seed(self):
		"""Reinitialize master RNGs when seed changes. Do NOT mutate global RNGs."""
		self.master_rng = random.Random(self.SEED)
		self.np_rng = np.random.default_rng(self.SEED)

	def PROGEMUP(self, initial_df, runs=100, seed=None):
		"""
		Vectorized PROGEMUP (sequential runs).
		- Eliminates per-player/per-attribute loops when recording run outputs.
		- Produces the same tidy CSV as before but built from vectorized arrays.
		- Keeps reproducibility: per-run seeds derived from self.master_rng.
		"""
		if seed is not None:
			self.SEED = seed
		# reinitialize master RNGs derived from current seed
		self._set_seed()

		# players order and attributes list (Ovr first)
		players = list(initial_df.index)
		attrs = ["Ovr"] + list(self.ATTRS)
		n_players = len(players)
		n_attrs = len(attrs)

		# ---------------- Metadata ----------------
		meta = initial_df.reindex(players).copy()
		# ensure these metadata columns exist (as in original)
		for c in ("Name", "Pos", "Team", "Age"):
			if c not in meta.columns:
				meta[c] = ""
		# store for later use
		self._meta = meta[["Name", "Pos", "Team", "Age"]]

		# ---------------- Baseline ----------------
		# baseline: array shape (n_players, n_attrs), integers
		baseline = np.zeros((n_players, n_attrs), dtype=int)
		df_players = initial_df.reindex(players)

		# For non-Ovr attrs, copy values; for Ovr, compute with calcovr per-row (keeps original semantics)
		for j, a in enumerate(attrs):
			if a == "Ovr":
				# calcovr likely depends on self.ATTRS; keep per-player computation (fast relative to sim)
				ovr_vals = []
				for _, row in df_players.iterrows():
					d = {k: int(row.get(k, 0) or 0) for k in self.ATTRS}
					ovr_vals.append(int(self.calcovr(d)))
				baseline[:, j] = ovr_vals
			else:
				col = df_players.get(a, pd.Series(0, index=players)).fillna(0)
				baseline[:, j] = col.astype(int).values

		# cache baseline and attrs
		self._baseline = baseline
		self._attrs = attrs

		# make a DataFrame representation of baseline for convenient fillna with aligned index/cols
		baseline_df = pd.DataFrame(baseline, index=players, columns=attrs)

		# ---------------- Prepare simulation storage ----------------
		sim = np.empty((runs, n_players, n_attrs), dtype=int)
		# per-run seeds (deterministic given master_rng)
		self._run_seeds = [self.master_rng.randint(0, 2**63 - 1) for _ in range(runs)]

		# ---------------- Run simulations (sequential) ----------------
		# Each run produces a DataFrame; we align its index/columns once and dump the whole block into sim[r]
		for r in range(runs):
			run_seed = self._run_seeds[r]
			run_rng = random.Random(run_seed)

			# runoneprog should accept rng and return a DataFrame indexed by playerID and with attr columns
			after = self.runoneprog(initial_df, rng=run_rng)

			# align to players × attrs. reindex(index=players, columns=attrs) ensures missing rows/cols -> NaN
			after_aligned = after.reindex(index=players, columns=attrs)

			# fill missing values from the baseline_df and convert to integer numpy array
			# baseline_df has same index & columns so fillna will broadcast appropriately
			after_filled = after_aligned.fillna(baseline_df)
			# convert to numpy int array; this is the vectorized step that replaces per-cell .at calls
			after_arr = after_filled.to_numpy(dtype=int)

			# store the entire players×attrs block for run r
			sim[r, :, :] = after_arr

			# progress printing preserved (as original)
			if (r + 1) % 1000 == 0:
				print(f"Run {r} complete (seed={run_seed})")

		# total simulated "player runs"
		self.totalsimcounts = runs * n_players

		# keep sim and delta for later inspection
		self._sim = sim
		self._delta = sim - baseline[np.newaxis, :, :]

		
		from progutils import GodProgSystem
		try:
			print(f'GODPROG!! Count:{GodProgSystem.godProgCount}, Average God Prog: {np.mean(GodProgSystem.godprogs)}, '
				f'Max God Prog: {np.max(GodProgSystem.godprogs)}, Min God Prog: {np.min(GodProgSystem.godprogs)}')
			print(f'Max Age GODPROGGED: {GodProgSystem.maxagegp}')
		except:
			print('No GODPROGS :(')

		# ---------------- Build tidy CSV (vectorized) ----------------
		# We will create:
		#   1) a baseline block (Run=-1) with one row per player
		#   2) a simulation block with runs * players rows, arranged run-major:
		#         run 0 -> player 0..P-1, run 1 -> player 0..P-1, ...

		# --- baseline rows ---
		baseline_rows = {
			"Run": np.full(n_players, -1, dtype=int),
			"RunSeed": [None] * n_players,
			"PlayerID": np.array(players, dtype=object),
			"Baseline": baseline[:, 0].astype(int),
			"Value": baseline[:, 0].astype(int),   # baseline "value" equals baseline Ovr
			"Delta": np.zeros(n_players, dtype=int),
			"PctChange": np.zeros(n_players, dtype=float),
			"AboveBaseline": np.zeros(n_players, dtype=bool),
			"CumulativeMean": baseline[:, 0].astype(float),
			"CumulativeStd": np.zeros(n_players, dtype=float),
			"CumulativeMin": baseline[:, 0].astype(int),
			"CumulativeMax": baseline[:, 0].astype(int),
			# add meta columns (Name/Pos/Team/Age)
			"Name": self._meta["Name"].astype(object).values,
			"Pos": self._meta["Pos"].astype(object).values,
			"Team": self._meta["Team"].astype(object).values,
			"Age": self._meta["Age"].astype(object).values,
		}

		# include attribute columns for baseline (Ovr + other attrs)
		for j, a in enumerate(attrs):
			baseline_rows[a] = baseline[:, j].astype(int)

		baseline_df_export = pd.DataFrame(baseline_rows)

		# --- simulation rows (vectorized) ---
		# Extract Ovr values for all runs and players (shape: runs, n_players)
		ovr_vals = sim[:, :, 0].astype(int)

		# cumulative stats along the runs axis
		counts = np.arange(1, runs + 1, dtype=float)[:, None]      # shape (runs, 1)
		cumsum = np.cumsum(ovr_vals, axis=0).astype(float)         # shape (runs, n_players)
		cum_mean = cumsum / counts                                 # float
		sum_sq = np.cumsum(ovr_vals.astype(np.float64) ** 2, axis=0)
		# population variance: E[x^2] - (E[x])^2
		cum_var = (sum_sq / counts) - (cum_mean ** 2)
		# numerical safety: clamp small negatives to zero
		cum_var = np.maximum(cum_var, 0.0)
		cum_std = np.sqrt(cum_var)

		cum_min = np.minimum.accumulate(ovr_vals, axis=0)
		cum_max = np.maximum.accumulate(ovr_vals, axis=0)

		# flatten arrays in run-major order (run 0 players 0..P-1, then run1, ...)
		runs_idx = np.repeat(np.arange(runs, dtype=int), n_players)      # length runs*n_players
		player_ids = np.tile(np.array(players, dtype=object), runs)      # length runs*n_players

		ovr_flat = ovr_vals.reshape(-1)                                  # int
		baseline_rep = np.tile(baseline[:, 0].astype(int), runs)         # repeat baseline for each run
		runseeds_rep = np.repeat(np.array(self._run_seeds, dtype=object), n_players)

		delta_flat = ovr_flat - baseline_rep
		# pct change safely (baseline may be zero)
		pct_flat = np.where(baseline_rep != 0, delta_flat / baseline_rep.astype(float), 0.0)
		above_flat = ovr_flat > baseline_rep

		cum_mean_flat = cum_mean.reshape(-1)
		cum_std_flat = cum_std.reshape(-1)
		cum_min_flat = cum_min.reshape(-1)
		cum_max_flat = cum_max.reshape(-1)

		# meta columns repeated for each run
		name_rep = np.tile(self._meta["Name"].astype(object).values, runs)
		pos_rep = np.tile(self._meta["Pos"].astype(object).values, runs)
		team_rep = np.tile(self._meta["Team"].astype(object).values, runs)
		age_rep = np.tile(self._meta["Age"].astype(object).values, runs)

		# assemble the simulation DataFrame
		sim_data = {
			"Run": runs_idx,
			"RunSeed": runseeds_rep,
			"PlayerID": player_ids,
			"Baseline": baseline_rep,
			"Value": ovr_flat,
			"Delta": delta_flat,
			"PctChange": pct_flat,
			"AboveBaseline": above_flat,
			"CumulativeMean": cum_mean_flat,
			"CumulativeStd": cum_std_flat,
			"CumulativeMin": cum_min_flat,
			"CumulativeMax": cum_max_flat,
			"Name": name_rep,
			"Pos": pos_rep,
			"Team": team_rep,
			"Age": age_rep,
		}

		# add all attributes (Ovr + other attrs) flattened
		for j, a in enumerate(attrs):
			sim_data[a] = sim[:, :, j].reshape(-1).astype(int)

		sim_df_export = pd.DataFrame(sim_data)
		print(f'Total Sims: {self.totalsimcounts}')
		# final column ordering to match original
		cols = ["Run", "RunSeed", "Name", "Pos", "Team", "Age", "PlayerID",
				"Baseline", "Value", "Delta", "PctChange", "AboveBaseline",
				"CumulativeMean", "CumulativeStd", "CumulativeMin", "CumulativeMax"] + attrs

		# concat baseline and sim blocks, baseline first
		self.export_ = pd.concat([baseline_df_export[cols], sim_df_export[cols]], ignore_index=True)

		print("Beginning write to excel.")
		return self.export_
