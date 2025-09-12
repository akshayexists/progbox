from progutils2 import *
import numpy as np
import pandas as pd
import random
import json

class runsim(progsandbox):
	"""Monte Carlo wrapper for NOEYETEST simulation."""
	def __init__(self, seed):
		if seed is None:
			raise ValueError("runsim requires a seed")
		self.SEED = seed
		super().__init__(seed=self.SEED)
		
		# Single RNG instances
		self.master_rng = random.Random(self.SEED)
		self.np_rng = np.random.default_rng(self.SEED)
		
		self.totalsimulationcounts = 0
		
		# Core attributes for OVR calculation
		self.CORE_ATTRS = ['Hgt', 'Str', 'Spd', 'Jmp', 'End', 'Ins', 'Dnk', 
						  'FT', '3Pt', 'oIQ', 'dIQ', 'Drb', 'Pss', '2Pt', 'Reb']

	def _prepare_baseline_data(self, df):
		"""Prepare baseline data and metadata in one pass."""
		players = list(df.index)
		
		# Extract metadata columns
		meta_cols = ['Name', 'Team', 'Age']
		meta = df[meta_cols].fillna('').copy()
		
		# Prepare baseline values for all attributes at once
		attrs = ['Ovr', 'PER'] + self.ATTRS
		baseline_data = np.zeros((len(players), len(attrs)), dtype=float)
		
		for i, player_idx in enumerate(players):
			row = df.loc[player_idx]
			
			# Calculate OVR from core attributes
			attr_dict = {k: float(row.get(k, 0) or 0) for k in self.CORE_ATTRS}
			baseline_data[i, 0] = float(self.calcovr(attr_dict))  # OVR
			
			# Add other attributes
			for j, attr in enumerate(attrs[1:], 1):
				baseline_data[i, j] = float(row.get(attr, 0) or 0)
		
		return players, attrs, baseline_data, meta

	def _run_single_simulation(self, df, baseline_data, players, attrs, run_seed):
		"""Execute a single simulation run."""
		run_rng = random.Random(run_seed)
		
		# Run progression
		after_df = self.runoneprog(df, rng=run_rng, seed = run_seed)
		
		# Align and fill missing data
		after_aligned = after_df.reindex(index=players, columns=attrs).fillna(
			pd.DataFrame(baseline_data, index=players, columns=attrs)
		)
		
		# Recalculate OVR for progressed players
		ovr_values = []
		for player_idx in players:
			self.totalsimulationcounts +=1
			row = after_aligned.loc[player_idx]
			attr_dict = {k: float(row.get(k, 0) or 0) for k in self.ATTRS}
			ovr_values.append(float(self.calcovr(attr_dict)))
		
		after_aligned['Ovr'] = ovr_values
		return after_aligned.to_numpy(dtype=float)

	def _build_export_dataframe(self, baseline_data, sim_results, players, attrs, meta, run_seeds):
		"""Build the final export DataFrame efficiently."""
		n_players, n_attrs = baseline_data.shape
		n_runs = len(sim_results)
		
		# Create baseline rows
		baseline_rows = self._create_baseline_rows(baseline_data, players, attrs, meta)
		
		# Create simulation rows
		sim_rows = self._create_simulation_rows(sim_results, baseline_data, players, attrs, meta, run_seeds)
		
		# Combine and return
		all_data = {**baseline_rows}
		for key in sim_rows:
			all_data[key] = np.concatenate([baseline_rows[key], sim_rows[key]])
		
		return pd.DataFrame(all_data)

	def _create_baseline_rows(self, baseline_data, players, attrs, meta):
		"""Create baseline data rows."""
		n_players = len(players)
		
		data = {
			'Run': np.full(n_players, -1),
			'RunSeed': [None] * n_players,
			'PlayerID': np.array(players, dtype=object),
			'Baseline': baseline_data[:, 0],  # OVR baseline
			'Value': baseline_data[:, 0],     # Same as baseline
			'Delta': np.zeros(n_players),
			'PctChange': np.zeros(n_players, dtype=float),
			'AboveBaseline': np.zeros(n_players, dtype=bool),
		}
		
		# Add metadata
		for col in ['Name', 'Team', 'Age']:
			data[col] = meta[col].values
		
		# Add attribute columns
		for j, attr in enumerate(attrs):
			data[attr] = baseline_data[:, j].astype(float)
		
		return data

	def _create_simulation_rows(self, sim_results, baseline_data, players, attrs, meta, run_seeds):
		"""Create simulation data rows."""
		n_runs = len(sim_results)
		n_players = len(players)
		
		# Flatten simulation results
		sim_flat = np.array(sim_results)  # (runs, players, attrs)
		ovr_flat = sim_flat[:, :, 0].flatten()  # OVR values
		baseline_ovr = np.tile(baseline_data[:, 0], n_runs)
		
		# Calculate deltas and percentages
		delta_flat = ovr_flat - baseline_ovr
		pct_flat = np.divide(delta_flat, baseline_ovr.astype(float), 
						   out=np.zeros_like(delta_flat, dtype=float), 
						   where=baseline_ovr != 0)
		
		data = {
			'Run': np.repeat(range(n_runs), n_players),
			'RunSeed': np.repeat(run_seeds, n_players),
			'PlayerID': np.tile(players, n_runs),
			'Baseline': baseline_ovr,
			'Value': ovr_flat,
			'Delta': delta_flat,
			'PctChange': pct_flat,
			'AboveBaseline': ovr_flat > baseline_ovr,
		}
		
		# Add metadata (tiled across runs)
		for col in ['Name', 'Team', 'Age']:
			data[col] = np.tile(meta[col].values, n_runs)
		
		# Add attribute columns
		for j, attr in enumerate(attrs):
			data[attr] = sim_flat[:, :, j].flatten().astype(float)
		
		return data

	def PROGEMUP(self, initial_df, runs=100, seed=None):
		"""
		Simplified vectorized PROGEMUP with PER included in output.
		"""
		if seed is not None:
			self.SEED = seed
			self.master_rng = random.Random(self.SEED)
			self.np_rng = np.random.default_rng(self.SEED)

		# Prepare all baseline data
		players, attrs, baseline_data, meta = self._prepare_baseline_data(initial_df)
		
		# Generate run seeds
		run_seeds = [self.master_rng.randint(0, 2**63 - 1) for _ in range(runs)]
		
		# Execute simulations
		sim_results = []
		for r, run_seed in enumerate(run_seeds):
			result = self._run_single_simulation(initial_df, baseline_data, players, attrs, run_seed)
			sim_results.append(result)
			
			if (r + 1) % 1000 == 0:
				print(f"Run {r + 1} complete (seed={run_seed})")
		
		# Build final export DataFrame
		export_df = self._build_export_dataframe(baseline_data, sim_results, players, attrs, meta, run_seeds)
		
		# Ensure proper column ordering
		base_cols = ['Run', 'RunSeed', 'Name', 'Team', 'Age', 'PlayerID',
					'Baseline', 'Value', 'Delta', 'PctChange', 'AboveBaseline']
		final_cols = base_cols + attrs
		
		self.export_ = export_df[final_cols]
		
		# Log results
		with open('outputs/godprogs.json', 'w', encoding='utf-8') as f:
			json.dump(sorted(GodProgSystem.playersgodprogged, key = lambda x: x['Name']), f, ensure_ascii=False, indent=4)
		with open('outputs/superlucky.json', 'w', encoding='utf-8') as f:
			json.dump(GodProgSystem.superlucky, f, ensure_ascii=False, indent=4)
		print(f'God Progs: {GodProgSystem.godProgCount}, Max Age God Progged: {GodProgSystem.maxagegp}. Exported god prog logs.')
		print(f"Total Sims: {self.totalsimulationcounts}")
		
		return self.export_
