from progutils import *
import numpy as np
import pandas as pd
import random
import json
import os
from pathlib import Path

class runsim(progsandbox):
    """Monte Carlo wrapper for NoEyeTest simulation."""
    
    def __init__(self, seed):
        if seed is None:
            raise ValueError("runsim requires a seed")
        
        super().__init__(seed=seed)
        self.seed = seed
        self.master_rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.total_simulation_count = 0
        self.results_df = None

    def _extract_player_data(self, df):
        """Extract and prepare player data for simulations."""
        players = list(df.index)
        
        # Extract metadata
        meta_cols = ['Name', 'Team', 'Age']
        metadata = df[meta_cols].fillna('').copy()
        
        # Prepare baseline attribute data
        all_attrs = ['Ovr', 'PER', 'DWS', 'BLK', 'STL'] + Config.ALL_ATTRS
        baseline_data = np.zeros((len(players), len(all_attrs)), dtype=np.float64)
        
        # Vectorized attribute extraction
        for i, player_idx in enumerate(players):
            row = df.loc[player_idx]
            
            # Calculate OVR from core attributes
            core_attrs = {attr: float(row.get(attr, 0) or 0) for attr in Config.ALL_ATTRS}
            baseline_data[i, 0] = calcovr(core_attrs)
            
            # Extract remaining attributes
            for j, attr in enumerate(all_attrs[1:], 1):
                baseline_data[i, j] = float(row.get(attr, 0) or 0)
        
        return players, all_attrs, baseline_data, metadata

    def _simulate_single_run(self, df, baseline_data, players, attrs, run_seed):
        """Execute one simulation run and return results."""
        # Run progression with dedicated RNG
        progressed_df = self.runoneprog(df, rng=random.Random(run_seed), seed=run_seed)
        
        # Align with original player list
        baseline_df = pd.DataFrame(baseline_data, index=players, columns=attrs)
        aligned_df = progressed_df.reindex(index=players, columns=attrs).fillna(baseline_df)
        
        # Recalculate OVR for all players
        ovr_values = np.zeros(len(players), dtype=int)
        for i, player_idx in enumerate(players):
            self.total_simulation_count += 1
            attr_dict = {attr: float(aligned_df.at[player_idx, attr]) 
                        for attr in Config.ALL_ATTRS}
            ovr_values[i] = calcovr(attr_dict)
        
        aligned_df['Ovr'] = ovr_values
        return aligned_df.to_numpy()

    def _build_results_dataframe(self, sim_results, baseline_data, players, attrs, metadata, run_seeds):
        """Convert simulation results into structured DataFrame."""
        n_runs = len(sim_results)
        n_players = len(players)
        
        # Stack all simulation results (runs × players × attrs)
        stacked_results = np.stack(sim_results, axis=0)
        
        # Extract OVR and calculate deltas (vectorized)
        sim_ovr = stacked_results[:, :, 0].flatten()
        baseline_ovr = np.tile(baseline_data[:, 0], n_runs)
        delta_ovr = sim_ovr - baseline_ovr
        
        # Calculate percentage change (handle division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_change = np.where(baseline_ovr != 0, delta_ovr / baseline_ovr, 0)
        
        # Build core result data using vectorized operations
        results_data = {
            'Run': np.repeat(range(n_runs), n_players),
            'RunSeed': np.repeat(run_seeds, n_players),
            'PlayerID': np.tile(players, n_runs),
            'Baseline': baseline_ovr,
            'Value': sim_ovr,
            'Delta': delta_ovr,
            'PctChange': pct_change,
            'AboveBaseline': (sim_ovr > baseline_ovr),
        }
        
        # Add metadata columns
        for col in metadata.columns:
            results_data[col] = np.tile(metadata[col].values, n_runs)
        
        # Add attribute columns
        for i, attr in enumerate(attrs):
            results_data[attr] = stacked_results[:, :, i].flatten()
        
        return pd.DataFrame(results_data)

    def _export_logs(self, output_dir='outputs/raw'):
        """Export god progression logs to JSON files."""
        try:
            # Resolve output directory safely
            workspace_base = Path(__file__).parent.resolve()

            if os.path.isabs(output_dir):
                output_path = Path(output_dir).resolve()
            else:
                output_path = (workspace_base / output_dir).resolve()

            # Validate path is within workspace
            try:
                output_path.relative_to(workspace_base)
            except ValueError:
                safe_default = workspace_base / 'outputs' / 'raw'
                print(f"Warning: Invalid output directory outside workspace. Using: {safe_default}")
                output_path = safe_default

            # Create directory if needed
            output_path.mkdir(parents=True, exist_ok=True)

            # Export detailed god progression records (array of objects)
            godprogs_file = output_path / 'godprogs.json'
            god_progs_records = self.tracking.godprog_records if hasattr(self.tracking, 'godprog_records') else []

            with open(godprogs_file, 'w', encoding='utf-8') as f:
                json.dump(god_progs_records, f, ensure_ascii=False, indent=2)

            # Export superlucky counts (dict of name -> count)
            superlucky_file = output_path / 'superlucky.json'
            superlucky_data = self.tracking.playersgodprogged if hasattr(self.tracking, 'playersgodprogged') else {}

            with open(superlucky_file, 'w', encoding='utf-8') as f:
                json.dump(superlucky_data, f, ensure_ascii=False, indent=2)

            # Print summary
            if self.tracking.godprogcount > 0:
                print(f'\nGod Progressions: {self.tracking.godprogcount}')
                print(f'Logs exported to: {output_path}')
            else:
                print(f'\nNo god progressions occurred.')

        except (OSError, ValueError) as e:
            print(f"Error: Could not export logs - {e}")
        except Exception as e:
            print(f"Warning: Unexpected error during log export - {e}")

    def PROGEMUP(self, initial_df, runs=100, seed=None, output_dir='outputs/raw'):
        """
        Execute Monte Carlo simulation with vectorized processing.
        
        Args:
            initial_df: DataFrame with player data (must include Age, PER, DWS, BLK, STL, and attributes)
            runs: Number of simulation runs (default: 100)
            seed: Optional seed override
            output_dir: Directory for log exports (default: 'outputs/raw')
            
        Returns:
            DataFrame with simulation results including baseline, value, delta, and pct change
        """
        
        # Update seed if provided
        if seed is not None:
            self.seed = seed
            self.master_rng = random.Random(seed)
            self.np_rng = np.random.default_rng(seed)
        
        print(f"Starting Monte Carlo simulation: {runs} runs, seed={self.seed}")
        
        # Extract and prepare data
        players, attrs, baseline_data, metadata = self._extract_player_data(initial_df)
        print(f"Processing {len(players)} players with {len(attrs)} attributes")
        
        # Pre-generate all run seeds
        run_seeds = [self.master_rng.randint(0, 2**63 - 1) for _ in range(runs)]
        
        # Execute simulations
        sim_results = []
        for run_idx, run_seed in enumerate(run_seeds, 1):
            result = self._simulate_single_run(initial_df, baseline_data, players, attrs, run_seed)
            sim_results.append(result)
            
            # Progress indicator
            if run_idx % 100 == 0 or run_idx == runs:
                print(f"Completed: {run_idx}/{runs} runs")
        
        # Build final results DataFrame
        print("Building results DataFrame...")
        results_df = self._build_results_dataframe(
            sim_results, baseline_data, players, attrs, metadata, run_seeds
        )
        
        # Ensure proper column ordering
        base_columns = ['Run', 'RunSeed', 'Name', 'Team', 'Age', 'PlayerID',
                       'Baseline', 'Value', 'Delta', 'PctChange', 'AboveBaseline']
        self.results_df = results_df[base_columns + attrs]
        
        # Export logs and print summary
        self._export_logs(output_dir)
        print(f"\nTotal simulations executed: {self.total_simulation_count}")
        print(f"Results shape: {self.results_df.shape}")
        
        return self.results_df
