import json
import pandas as pd

YEAR = 2019
ATTRS = ["dIQ", "Dnk", "Drb", "End", "2Pt", "FT",
         "Ins", "Jmp", "oIQ", "Pss", "Reb", "Spd", "Str", "3Pt", "Hgt", "Ovr"]
FAILSAFE = {'end': 'endu', '2pt': 'fg', '3pt': 'tp', 'str': 'stre'}

def exportcleaner(export_file, teams:list, teaminfo_file) -> pd.DataFrame:
    """
    Parse the export file and return a flat pandas.DataFrame with one row per player.
    Supports multiple teams.
    
    Columns: Team, Name, Age, PER + all ratings in ATTRS.
    """
    # Load input
    with open(export_file, "rb") as f: data = json.load(f)

    with open(teaminfo_file, "rb") as f: team_lookup = json.load(f)

    players = data.get("players", [])
    records = []

    for p in players:
        # Only include players who have played
        if not p['stats']: continue

        if p["tid"] < -1: continue #skip if retired or UDFA

        # Safe PER extraction
        stat_entry = p["stats"][-2] if p["stats"][-1].get("playoffs") else p["stats"][-1]
        per = stat_entry.get("per", 0)
        if per == 0: continue   #don't need players who are not in this game

        player_team = team_lookup.get(str(p["tid"]))
        # Team filter (skip if not in desired teams)
        if len(teams) > 0 and player_team not in teams:
            continue

        name = f"{p['firstName']} {p['lastName']}"
        age = int(YEAR - p["born"]["year"])

        # Automatically exclude those under 25
        if age < 25:
            continue

        # Ratings lookup (lowercased keys)
        value_lookup = {k.lower(): v for k, v in p["ratings"][-1].items()}

        # Core info
        row = {
            "Team": player_team,
            "Name": name,
            "Age": age,
            "PER": per
        }

        # Attributes with FAILSAFE mapping
        row.update({
            a: value_lookup.get(FAILSAFE.get(a.lower(), a.lower()), 0)
            for a in ATTRS
        })

        records.append(row)

    return pd.DataFrame(records)
