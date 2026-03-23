from runsim import runsim
from exportcleaner import exportcleaner
from analysis import generate_analysis
from datetime import datetime
from pathlib import Path
import json
import os

if __name__=="__main__":
    # ── Run configuration ─────────────────────────────────────────────────────────
    # Store these in a dict so we can easily export them later
    config = {
        "teams": [], # [] = all teams; e.g. ['GSW', 'BOS'] for specific teams
        "seed": 69,
        "runs": 500,
        "n_workers": max(os.cpu_count()-1, 1),
        "export_file": 'data/export.json',
        "teaminfo_file": 'data/teaminfo.json'
    }

    RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_path = f'outputs/{RUN_TS}'
    path = f'outputs/{RUN_TS}/raw/'
    Path(path).mkdir(parents=True, exist_ok=True)

    # ── Export Metadata ───────────────────────────────────────────────────────────
    # We do this early so even if the sim crashes, we know what we tried to run
    with open(f'{analysis_path}/metadata.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("Master seed:", config["seed"])

    data, metadata = exportcleaner(
        export_file=config["export_file"],
        teaminfo_file=config["teaminfo_file"],
        teams=config["teams"],
    )

    # ── Simulation ────────────────────────────────────────────────────────────────
    sim = runsim(seed=config["seed"])
    print("beginning the simulation, give it a while to start up!...")
    
    df = sim.PROGEMUP(
        data, 
        runs=config["runs"], 
        output_dir=path, 
        n_workers=config["n_workers"]
    )
    
    df.to_csv(path + 'outputs.csv')
    print(f'Written to {path}')

    print(f'Generating Sim Analytics Report!')
    sim.analytics.print_report()

    # ── Perform analysis ──────────────────────────────────────────────────────────
    generate_analysis(analysis_path)