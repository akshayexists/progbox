from runsim import runsim
from exportcleaner import exportcleaner

#look at all available teams
teams = None    #change to list of teams if specific teams are to be tested eg. ['GSW', 'BOS', 'NOL']

seed = 69420
print("Master seed:", seed)
runs = 200

data = exportcleaner(export_file='data/export.json', teaminfo_file='data/teaminfo.json', teams = teams)
print(data)

sim = runsim(seed=seed)
df = sim.PROGEMUP(data, runs=runs)
df.to_csv("montecarlo.csv", index=False)
