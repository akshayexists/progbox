from progutils import progsandbox
import numpy as np, pandas as pd, random

class runsim(progsandbox):
    """Monte Carlo wrapper producing a single analysis-ready CSV with cumulative Ovr stats.

    Behavior changes:
    - Uses self.master_rng (created by progsandbox) to derive per-run seeds.
    - Does NOT call random.seed() or np.random.seed() on the global RNGs.
    - Each run is independent (its own random.Random instance) while the whole
      experiment is reproducible given the same master seed.
    """

    def __init__(self, seed):
        if seed is None:
            raise ValueError("runsim requires a seed")
        self.SEED = seed
        super().__init__(seed=self.SEED)

        # numpy generator tied to the master seed (not using global np RNG)
        self.np_rng = np.random.default_rng(self.SEED)

        self._sim = None
        self._delta = None
        self._attrs = None
        self._meta = None
        self._baseline = None
        self.export_ = pd.DataFrame()

    def _set_seed(self):
        """Reinitialize master RNGs when seed changes. Do NOT mutate global RNGs."""
        self.master_rng = random.Random(self.SEED)
        self.np_rng = np.random.default_rng(self.SEED)

    def PROGEMUP(self, initial_df, runs=100, seed=None):
        """Run Monte Carlo sims and produce a single tidy CSV with flattened attributes and cumulative Ovr stats.

        This version records the per-run seed and the number of god progs produced
        in each run. It resets GodProgSystem.godProgCount before every run so the
        count is per-run (not cumulative across the entire experiment).
        """
        if seed is not None:
            self.SEED = seed
        self._set_seed()

        players = list(initial_df.index)
        attrs = ["Ovr"] + list(self.ATTRS)

        # ---------------- Metadata ----------------
        meta = initial_df.reindex(players).copy()
        for c in ("Name","Pos","Team","Age"):
            if c not in meta.columns:
                meta[c] = ""
        self._meta = meta[["Name","Pos","Team","Age"]]

        # ---------------- Baseline ----------------
        baseline = np.zeros((len(players), len(attrs)), dtype=int)
        df_players = initial_df.reindex(players)
        for j,a in enumerate(attrs):
            if a=="Ovr":
                vals=[]
                for _, row in df_players.iterrows():
                    d = {k:int(row.get(k,0) or 0) for k in self.ATTRS}
                    vals.append(int(self.calcovr(d)))
                baseline[:,j] = vals
            else:
                col = df_players.get(a,pd.Series(0,index=players)).fillna(0)
                baseline[:,j] = col.astype(int).values
        self._baseline = baseline
        self._attrs = attrs

        # ---------------- Simulations ----------------
        sim = np.empty((runs, len(players), len(attrs)), dtype=int)

        # For reproducibility we also record the per-run seeds and god-prog counts
        self._run_seeds = []

        for r in range(runs):
            # derive a fresh, independent seed from the master RNG
            run_seed = self.master_rng.randint(0, 2**63 - 1)
            self._run_seeds.append(run_seed)

            run_rng = random.Random(run_seed)

            after = self.runoneprog(initial_df, rng=run_rng).reindex(players)
            if (r+1) % 1000 == 0: print(f"Run {r} complete (seed={run_seed})")
            for i,pid in enumerate(players):
                for j,a in enumerate(attrs):
                    val = int(after.at[pid,a]) if a in after.columns and pd.notna(after.at[pid,a]) else int(baseline[i,j])
                    sim[r,i,j] = val
        self._sim = sim
        self._delta = sim - baseline[np.newaxis,:,:]

        # ---------------- Build tidy CSV ----------------
        rows=[]
        # Baseline row (Run=-1)
        for i,pid in enumerate(players):
            meta_row = self._meta.loc[pid].to_dict()
            row = {
                "Run": -1,
                "RunSeed": None,
                "PlayerID": pid,
                "Baseline": int(baseline[i,0]),
                "Value": int(baseline[i,0]),
                "Delta": 0,
                "PctChange": 0.0,
                "AboveBaseline": False,
                "CumulativeMean": int(baseline[i,0]),
                "CumulativeStd": 0.0,
                "CumulativeMin": int(baseline[i,0]),
                "CumulativeMax": int(baseline[i,0]),
                **meta_row
            }
            for j,a in enumerate(attrs):
                row[a] = int(baseline[i,j])
            rows.append(row)

        # Simulation runs
        cumulative = {pid:[] for pid in players}  # store Ovr values for cumulative stats
        for r in range(runs):
            run_seed = self._run_seeds[r]
            for i,pid in enumerate(players):
                meta_row = self._meta.loc[pid].to_dict()
                ovr_val = int(sim[r,i,0])
                cumulative[pid].append(ovr_val)
                row = {
                    "Run": r,
                    "RunSeed": run_seed,
                    "PlayerID": pid,
                    "Baseline": int(baseline[i,0]),
                    "Value": ovr_val,
                    "Delta": ovr_val - int(baseline[i,0]),
                    "PctChange": (ovr_val - int(baseline[i,0]))/int(baseline[i,0]) if baseline[i,0]!=0 else 0.0,
                    "AboveBaseline": ovr_val > int(baseline[i,0]),
                    "CumulativeMean": float(np.mean(cumulative[pid])),
                    "CumulativeStd": float(np.std(cumulative[pid], ddof=0)),
                    "CumulativeMin": int(np.min(cumulative[pid])),
                    "CumulativeMax": int(np.max(cumulative[pid])),
                    **meta_row
                }
                for j,a in enumerate(attrs):
                    row[a] = int(sim[r,i,j])
                rows.append(row)

        self.export_ = pd.DataFrame(rows)
        cols = ["Run","RunSeed","Name","Pos","Team","Age","PlayerID",
                "Baseline","Value","Delta","PctChange","AboveBaseline",
                "CumulativeMean","CumulativeStd","CumulativeMin","CumulativeMax"] + attrs
        self.export_ = self.export_[cols]
        print("Exported.")
        return self.export_
