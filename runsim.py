from progutils import *
import numpy as np
import pandas as pd
import random
import json

class runsim(progsandbox):
    """Monte Carlo wrapper for NOEYETEST simulation."""
    
    CORE_ATTRS = ['Hgt', 'Str', 'Spd', 'Jmp', 'End', 'Ins', 'Dnk', 
                  'FT', '3Pt', 'oIQ', 'dIQ', 'Drb', 'Pss', '2Pt', 'Reb']
    
    def __init__(self, seed):
        if seed is None:
            raise ValueError("runsim requires a seed")
        
        super().__init__(seed=seed)
        self.SEED = seed
        self.master_rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.totalsimulationcounts = 0

    def _extract_player_data(self, df):
        """Extract and prepare player data for simulations."""
        players = list(df.index)
        
        # Extract metadata
        meta_cols = ['Name', 'Team', 'Age']
        metadata = df[meta_cols].fillna('').copy()
        
        # Prepare baseline attribute data
        all_attrs = ['Ovr', 'PER'] + self.ATTRS
        baseline_data = np.zeros((len(players), len(all_attrs)))
        
        for i, player_idx in enumerate(players):
            row = df.loc[player_idx]
            
            # Calculate OVR from core attributes - vectorized extraction
            core_attrs = {attr: float(row.get(attr, 0) or 0) for attr in self.CORE_ATTRS}
            baseline_data[i, 0] = self.calcovr(core_attrs)
            
            # Add remaining attributes - direct array assignment
            for j, attr in enumerate(all_attrs[1:], 1):
                baseline_data[i, j] = float(row.get(attr, 0) or 0)
        
        return players, all_attrs, baseline_data, metadata

    def _simulate_single_run(self, df, baseline_data, players, attrs, run_seed):
        """Execute one simulation run and return results."""
        # Run progression
        progressed_df = self.runoneprog(df, rng=random.Random(run_seed), seed=run_seed)
        
        # Align with original player list - fillna with baseline avoids missing data
        baseline_df = pd.DataFrame(baseline_data, index=players, columns=attrs)
        aligned_df = progressed_df.reindex(index=players, columns=attrs).fillna(baseline_df)
        
        # Recalculate OVR - vectorized where possible
        ovr_values = []
        for player_idx in players:
            self.totalsimulationcounts += 1
            attr_dict = {attr: float(aligned_df.at[player_idx, attr]) for attr in self.ATTRS}
            ovr_values.append(self.calcovr(attr_dict))
        
        aligned_df['Ovr'] = ovr_values
        return aligned_df.to_numpy()

    def _build_results_dataframe(self, sim_results, baseline_data, players, attrs, metadata, run_seeds):
        """Convert simulation results into structured DataFrame."""
        n_runs, n_players = len(sim_results), len(players)
        
        # Stack all simulation results
        stacked_results = np.stack(sim_results)  # shape: (runs, players, attrs)
        
        # Extract OVR and calculate deltas - fully vectorized
        sim_ovr = stacked_results[:, :, 0].flatten()
        baseline_ovr = np.tile(baseline_data[:, 0], n_runs)
        delta_ovr = sim_ovr - baseline_ovr
        
        # Safe percentage calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_change = np.where(baseline_ovr != 0, delta_ovr / baseline_ovr, 0)
        
        # Build core result data - using np.tile and np.repeat for efficiency
        results_data = {
            'Run': np.repeat(range(n_runs), n_players),
            'RunSeed': np.repeat(run_seeds, n_players),
            'PlayerID': np.tile(players, n_runs),
            'Baseline': baseline_ovr,
            'Value': sim_ovr,
            'Delta': delta_ovr,
            'PctChange': pct_change,
            'AboveBaseline': sim_ovr > baseline_ovr,
        }
        
        # Add metadata - single tile operation per column
        for col in metadata.columns:
            results_data[col] = np.tile(metadata[col].values, n_runs)
        
        # Add attributes - direct array operations
        for i, attr in enumerate(attrs):
            results_data[attr] = stacked_results[:, :, i].flatten()
        
        return pd.DataFrame(results_data)

    def _export_logs(self):
        """Export god progression logs."""
        try:
            # Check if GodProgSystem exists and has data
            if not hasattr(GodProgSystem, 'playersgodprogged'):
                return
            
            with open('outputs/raw/godprogs.json', 'w', encoding='utf-8') as f:
                god_progs_sorted = sorted(GodProgSystem.playersgodprogged, key=lambda x: x['Name'])
                json.dump(god_progs_sorted, f, ensure_ascii=False, indent=4)
            
            with open('outputs/raw/superlucky.json', 'w', encoding='utf-8') as f:
                json.dump(GodProgSystem.superlucky, f, ensure_ascii=False, indent=4)
            
            print(f'God Progs: {GodProgSystem.godProgCount}, '
                  f'Max Age God Progged: {GodProgSystem.maxagegp}. Exported god prog logs.')
        except Exception as e:
            print(f"Warning: Could not export logs - {e}")

    def PROGEMUP(self, initial_df, runs=100, seed=None):
        """
        Execute Monte Carlo simulation with simplified, vectorized processing.
        
        Args:
            initial_df: DataFrame with player data
            runs: Number of simulation runs
            seed: Optional seed override
            
        Returns:
            DataFrame with simulation results
        """
        # Update seed if provided
        if seed is not None:
            self.SEED = seed
            self.master_rng = random.Random(seed)
            self.np_rng = np.random.default_rng(seed)

        # Prepare data
        players, attrs, baseline_data, metadata = self._extract_player_data(initial_df)
        
        # Pre-generate all seeds at once
        run_seeds = [self.master_rng.randint(0, 2**63 - 1) for _ in range(runs)]
        
        # Execute simulations with cleaner progress tracking
        sim_results = []
        for run_idx, run_seed in enumerate(run_seeds, 1):
            result = self._simulate_single_run(initial_df, baseline_data, players, attrs, run_seed)
            sim_results.append(result)
            
            if run_idx % 100 == 0:
                print(f"Run {run_idx}/{runs} complete")
        
        # Build final results DataFrame
        results_df = self._build_results_dataframe(sim_results, baseline_data, players, attrs, metadata, run_seeds)
        
        # Ensure proper column ordering
        base_columns = ['Run', 'RunSeed', 'Name', 'Team', 'Age', 'PlayerID',
                       'Baseline', 'Value', 'Delta', 'PctChange', 'AboveBaseline']
        self.export_ = results_df[base_columns + attrs]
        
        # Export logs and summary
        self._export_logs()
        print(f"Total Simulations: {self.totalsimulationcounts}")
        
        return self.export_
