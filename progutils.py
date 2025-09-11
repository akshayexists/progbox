import math
import random
import copy

#  port of NoEyeTest / BBGM progression logic for players 25+
# RNG clarity: this version uses an object-local RNG (random.Random) so
# - re-running the whole simulation with the same master seed is reproducible
# - each Monte Carlo run can be made independent by deriving a per-run seed
#   from the master RNG and passing a dedicated RNG into runoneprog
# LLM was used for code cleaning and slight refactoring.

class Config:
	AGE_RANGES = {
		"YOUNG": '25-30',
		"MID": '31-34',
		"OLD": '35+',
	}

	PROGRESSION_LIMITS = {
		"MAX_OVR": 80,
		"MIN_RATING": 30,
		"MAX_RATING": 61,
		"MAX_GOD_PROG_CHANCE": 0.09,
		"MIN_GOD_PROG": 7,
		"MAX_GOD_PROG": 13,
	}

	SKILL_KEYS = {
		"ALL": [
			'dIQ', 'Dnk', 'Drb', 'End', '2Pt', 'FT', 'Ins', 'Jmp', 'oIQ', 'Pss',
			'Reb', 'Spd', 'Str', '3Pt',
		],
		"PHYSICAL_OLD": ['Spd', 'Str', 'Jmp', 'End'],
		"PHYSICAL_MID": ['Spd', 'Str', 'Jmp'],
	}

	PROG_RANGES = {
		'25-30': {
			'min1': 5,
			'min2': 7,
			'max1': 4,
			'max2': 2,
			'hardMax': 4,
		},
		'31-34': {
			'min1': 6,
			'min2': 7,
			'max1': 4,
			'max2': 3,
			'hardMax': 2,
		},
		'35+': {
			'min1': 6,
			'min2': 9,
			'max1': None,
			'max2': None,
			'hardMax': 0,
		},
	}


class Utils:
	@staticmethod
	def randomInt(minv, maxv, rng=None):
		"""Return a deterministic integer in [minv,maxv] using the provided
		rng (random.Random) or the global random as fallback.
		"""
		if rng is None:
			return random.randint(minv, maxv)
		return rng.randint(minv, maxv)


class ProgressionCalculator:
	@staticmethod
	def getProgRange(per, progOptions, rng=None):
		# Accept an rng (random.Random) to avoid global RNG usage. If rng is None
		# the code falls back to the global random module to remain backward compatible.
		min1 = progOptions.get('min1')
		min2 = progOptions.get('min2')
		max1 = progOptions.get('max1')
		max2 = progOptions.get('max2')
		hardMin = progOptions.get('hardMin')
		hardMax = progOptions.get('hardMax')
		ovr = progOptions.get('ovr')
		age = progOptions.get('age')

		if min1 is None or min2 is None:
			raise ValueError('progOptions must include min1 and min2')

		if per <= 20 and age < 31:
			mn = math.ceil(per / 5) - 6
			mx = math.ceil(per / 4) - 1
		else:
			mn = math.ceil(per / min1) - min2
			if max1 is not None and max2 is not None:
				if max1 == 0:
					mx = 2
				else:
					mx = math.ceil(per / max1) - max2
			else:
				mx = 2

		if hardMin is not None:
			mn = hardMin
		if hardMax is not None and mx > hardMax:
			mx = hardMax

		if mx + ovr >= Config.PROGRESSION_LIMITS['MAX_OVR']:
			if ovr >= Config.PROGRESSION_LIMITS['MAX_OVR']:
				mx = 0
				if age > 30 and age < 35:
					mn = -10
				if age >= 35:
					mn = -14
				if age <= 30:
					# original JS: use Utils.randomInt(-2,0) < 0.02 seemed to be quirky
					# suggestion for JS: use if (Math.random() < 0.02) {adjustedMin = -2;}
					if (rng.random() if rng else random.random()) < 0.02:
						mn = -2
				if mn > mx:
					mn = 0
			else:
				mx = Config.PROGRESSION_LIMITS['MAX_OVR'] - ovr
				if mn + ovr >= Config.PROGRESSION_LIMITS['MAX_OVR']:
					mn = 0

		return int(mn), int(mx)


class GodProgSystem:
	godProgCount = 0
	godprogs = []
	maxagegp = 0

	@staticmethod
	def calculateGodProgChance(ovr):
		MIN_RATING = Config.PROGRESSION_LIMITS['MIN_RATING']
		MAX_RATING = Config.PROGRESSION_LIMITS['MAX_RATING']
		MAX_C = Config.PROGRESSION_LIMITS['MAX_GOD_PROG_CHANCE']

		if ovr < MIN_RATING:
			scale = 1.0
		elif ovr > MAX_RATING:
			scale = 0.01
		else:
			scale = 1.0 - (ovr - MIN_RATING) / (MAX_RATING - MIN_RATING)

		return scale * MAX_C

	@staticmethod
	def godProg(age, ovr, rng=None):
		# Only < 30 players considered, same as JS
		if age >= 30:
			return None

		chance = GodProgSystem.calculateGodProgChance(ovr)
		_rnd = rng.random() if rng is not None else random.random()
		if _rnd >= chance:
			return None

		rp = Utils.randomInt(Config.PROGRESSION_LIMITS['MIN_GOD_PROG'],
					 Config.PROGRESSION_LIMITS['MAX_GOD_PROG'], rng)
		
		GodProgSystem.godProgCount += 1
		GodProgSystem.godprogs.append(rp)
		if age > GodProgSystem.maxagegp: GodProgSystem.maxagegp = age
		#print("GODPROG!")
		return (rp, rp)


class progsandbox:
	def __init__(self, seed=0):
		self.ATTRS = [
			'dIQ', 'Dnk', 'Drb', 'End', '2Pt', 'FT', 'Ins', 'Jmp',
			'oIQ', 'Pss', 'Reb', 'Spd', 'Str', '3Pt', 'Hgt',
		]

		# master RNG controls reproducibility for the whole object. Do NOT
		# call random.seed() on the global RNG here.
		self.master_seed = seed
		self.master_rng = random.Random(seed)
		self.SEED = seed
		self.oldKeys = set(Config.SKILL_KEYS['PHYSICAL_OLD'])
		self.midKeys = set(Config.SKILL_KEYS['PHYSICAL_MID'])

	def _bound(self, v, lo, hi):
		return min(max(v, lo), hi)

	def limit_rating(self, v):
		return max(0, min(100, int(v)))

	def calcovr(self, r):
		s = (
			0.159 * (r['Hgt'] - 47.5) + 0.0777 * (r['Str'] - 50.2) + 0.123 * (r['Spd'] - 50.8)
			+ 0.051 * (r['Jmp'] - 48.7) + 0.0632 * (r['End'] - 39.9) + 0.0126 * (r['Ins'] - 42.4)
			+ 0.0286 * (r['Dnk'] - 49.5) + 0.0202 * (r['FT'] - 47.0) + 0.0726 * (r['3Pt'] - 47.1)
			+ 0.133 * (r['oIQ'] - 46.8) + 0.159 * (r['dIQ'] - 46.7) + 0.059 * (r['Drb'] - 54.8)
			+ 0.062 * (r['Pss'] - 51.3) + 0.01 * (r['2Pt'] - 47.0) + 0.01 * (r['Reb'] - 51.4) + 48.5
		)

		if s >= 68:
			fudge = 8
		elif s >= 50:
			fudge = 4 + (s - 50) * (4 / 18)
		elif s >= 42:
			fudge = -5 + (s - 42) * (9 / 8)
		elif s >= 31:
			fudge = -5 - (42 - s) * (5 / 11)
		else:
			fudge = -10

		return self.limit_rating(int(round(s + fudge)))

	def _params_for_age(self, age):
		if 25 <= age <= 30:
			return copy.deepcopy(Config.PROG_RANGES['25-30'])
		elif 31 <= age <= 34:
			return copy.deepcopy(Config.PROG_RANGES['31-34'])
		else:
			return copy.deepcopy(Config.PROG_RANGES['35+'])

	def _apply_attribute_progress(self, k, age, mn, mx, ratings, rng=None):
		"""Encapsulate the per-attribute progression logic and use the
		provided rng for deterministic draws when present.

		Returns the (possibly updated) rating for attribute k.
		"""
		# choose progression value for this key; if mn>mx this yields 0
		prog = Utils.randomInt(mn, mx, rng) if mn <= mx else 0

		# old player physical caps
		if age >= 30 and k in self.oldKeys and mx > 0:
			# oldProgPhys = Math.random() * 0.05 + 0.01 in JS
			r1 = rng.random() if rng is not None else random.random()
			oldProgPhys = r1 * 0.05 + 0.01
			# JS: if (Math.random() >= oldProgPhys) return true (skip)
			if (rng.random() if rng is not None else random.random()) >= oldProgPhys:
				# skip progression entirely for this attribute
				return ratings[k]
			# otherwise allow progression but cap it if necessary
			if mx > 3:
				prog = min(prog, 3)

		# mid-age (25..29) slowdown for physical attributes
		if 25 <= age < 30 and k in self.midKeys and prog > 0:
			ageFactor = 0.7 - (age - 25) * 0.1
			probProgression = max(ageFactor, 0)
			if (rng.random() if rng is not None else random.random()) > probProgression:
				prog = 0

		return self.limit_rating(ratings[k] + prog)

	def runoneprog(self, roster_df, rng=None):
		"""
		Apply PER-based progression to each player row in roster_df.
		slow iterrows/at overhead REPLACED!!
		"""
		if rng is None:
			rng = self.master_rng

		# Work on a copy of the roster
		out = roster_df.copy(deep=True)

		# Extract arrays for speed
		ages = out["Age"].to_numpy(dtype=int, copy=False)
		pers = out["PER"].to_numpy(dtype=float, copy=False)

		# We’ll update all attributes in place
		attr_arrays = {k: out[k].to_numpy(copy=True) if k in out.columns else np.zeros(len(out), dtype=int)
					for k in self.ATTRS}
		# Ensure Ovr column exists (will be recalculated)
		if "Ovr" not in out.columns:
			out["Ovr"] = 0
		ovr_array = out["Ovr"].to_numpy(copy=True)

		keys = [k for k in self.ATTRS if k != 'Hgt']

		for i in range(len(out)):
			age = ages[i]
			if age <= 25:
				continue
			per = pers[i]
			if per == 0:
				continue

			# Build ratings dict once for this player from arrays
			ratings = {k: int(round(attr_arrays[k][i])) for k in self.ATTRS if k != 'Hgt'}
			ratings['Hgt'] = int(attr_arrays['Hgt'][i])

			ovr = self.calcovr(ratings)

			# Params for age and PER
			params = self._params_for_age(age)
			params['hardMin'] = None
			params['ovr'] = int(ovr)
			params['age'] = int(age)

			mn, mx = ProgressionCalculator.getProgRange(per, params, rng)

			# God progression
			gp = GodProgSystem.godProg(age, int(ovr), rng)
			if gp is not None:
				mn, mx = gp

			# Apply per-key progression
			for k in keys:
				new_val = self._apply_attribute_progress(k, age, mn, mx, ratings, rng)
				ratings[k] = new_val
				attr_arrays[k][i] = new_val

			# Recalculate Ovr with updated ratings
			ovr_array[i] = self.calcovr(ratings)

		# Write updated arrays back into DataFrame
		for k, arr in attr_arrays.items():
			out[k] = arr
		out["Ovr"] = ovr_array

		return out
