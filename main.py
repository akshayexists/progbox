from runsim import runsim
from exportcleaner import exportcleaner
import secrets

teams = ['DAL']

seed = 69420
print("Master seed:", seed)
runs = 10000

data = exportcleaner(export_file='data/export.json', teaminfo_file='data/teaminfo.json', teams = teams)
print(data)

sim = runsim(seed=seed)
df = sim.PROGEMUP(data, runs=runs)
df.to_csv("montecarlo.csv", index=False)
