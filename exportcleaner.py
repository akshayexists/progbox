import json
import pandas as pd

YEAR = 2019
ATTRS = ["dIQ", "Dnk", "Drb", "End", "2Pt", "FT",
         "Ins", "Jmp", "oIQ", "Pss", "Reb", "Spd", "Str", "3Pt", "Hgt", "Ovr"]
FAILSAFE = {'end': 'endu', '2pt': 'fg', '3pt': 'tp', 'str': 'stre'}

def exportcleaner(export_file, teams, teaminfo_file, isdict=False) -> pd.DataFrame:
    """
    Parse the export file and return a flat pandas.DataFrame with one row per player.
    Supports multiple teams.
    
    Columns: Team, Name, Age, PER + all ratings in ATTRS.
    """
    # Load input
    if isdict:
        data = export_file
    else:
        with open(export_file, "rb") as f:
            data = json.load(f)

    with open(teaminfo_file, "rb") as f:
        team_lookup = json.load(f)

    # Normalize teams argument to a list
    if isinstance(teams, str):
        teams = [teams]

    players = data.get("players", [])
    records = []

    for p in players:
        if p["ratings"][-1]["season"] != YEAR:
            continue

        player_team = team_lookup.get(str(p["tid"]))
        if teams != None:
            if player_team not in teams:
                continue


        """
        if int(YEAR - p["born"]["year"]) < 25:
            print(f"{p["firstName"]} {p['lastName']} skipped")
            continue
        """

        name = f"{p['firstName']} {p['lastName']}"
        value_lookup = {k.lower(): v for k, v in p["ratings"][-1].items()}

        # Core info
        row = {
            "Team": player_team,
            "Name": name,
            "Age": YEAR - p["born"]["year"],
            "PER": (
                p["stats"][-2 if p["stats"] and p["stats"][-1].get("playoffs") else -1]["per"]
                if p["stats"] else 0
            ),
        }

        # Attributes
        row.update({
            a: value_lookup.get(FAILSAFE.get(a.lower(), a.lower()), 0)
            for a in ATTRS
        })

        records.append(row)

    return pd.DataFrame(records)
