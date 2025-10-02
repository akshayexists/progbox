import math
import random
import copy
import numpy as np
import pandas as pd

"""
references: https://github.com/fearandesire/NoEyeTest/blob/dev/tiers.md
			https://github.com/zengm-games/zengm/blob/master/src/worker/core/player/ovr.basketball.ts
			noeyetest.js (https://github.com/fearandesire/NoEyeTest/blob/dev)
"""


class Config:
	"""Centralized configuration for all progression constants."""
	
	# Age brackets
	YOUNG_MAX = 30
	MID_MAX = 34
	OLD_MIN = 35
	
	# Progression limits
	MAX_OVR = 80
	MIN_RATING = 30
	MAX_RATING = 61
	MAX_GOD_PROG_CHANCE = 0.09
	MIN_GOD_PROG = 7
	MAX_GOD_PROG = 13
	
	# Physical attributes by age group
	PHYSICAL_OLD = {'Spd', 'Str', 'Jmp', 'End'}
	PHYSICAL_MID = {'Spd', 'Str', 'Jmp'}
	
	# All progression attributes (excluding Hgt)
	ALL_ATTRS = ['dIQ', 'Dnk', 'Drb', 'End', '2Pt', 'FT', 'Ins', 'Jmp', 
				 'oIQ', 'Pss', 'Reb', 'Spd', 'Str', '3Pt', 'Hgt']
	
	# OVR calculation order (CRITICAL - do not change)
	OVR_CALC_ORDER = ['Hgt', 'Str', 'Spd', 'Jmp', 'End', 'Ins',
					  'Dnk', 'FT', '3Pt', 'oIQ', 'dIQ', 'Drb',
					  'Pss', '2Pt', 'Reb']
	
	# OVR coefficients and centers for vectorized ovr calc (CRITICAL - do not change)
	OVR_COEFFS = np.array([0.159, 0.0777, 0.123, 0.051, 0.0632, 0.0126,
						   0.0286, 0.0202, 0.0726, 0.133, 0.159, 0.059,
						   0.062, 0.01, 0.01], dtype=float)
	
	OVR_CENTERS = np.array([47.5, 50.2, 50.8, 48.7, 39.9, 42.4,
							49.5, 47.0, 47.1, 46.8, 46.7, 54.8,
							51.3, 47.0, 51.4], dtype=float)
	
	# Progression parameters by age range
	PROG_PARAMS = {
		(25, 30): {'min1': 5, 'min2': 7, 'max1': 4, 'max2': 2, 'hardMax': 4, 'hardMin': None},
		(31, 34): {'min1': 6, 'min2': 7, 'max1': 4, 'max2': 3, 'hardMax': 2, 'hardMin': None},
		(35, 99): {'min1': 6, 'min2': 9, 'max1': None, 'max2': None, 'hardMax': 0, 'hardMin': None},
	}
	
	# Special progression rules
	EARLY_PROG_PER_THRESHOLD = 20
	EARLY_PROG_AGE_THRESHOLD = 31
	EARLY_PROG_PER_DIVISOR_MN = 5
	EARLY_PROG_PER_OFFSET_MN = -6
	EARLY_PROG_PER_DIVISOR_MX = 4
	EARLY_PROG_PER_OFFSET_MX = -1
	
	DEFAULT_MAX_PROG = 2
	

class GodProgSystem:
	"""Tracks and manages 'god progression' events for young players."""
	
	godProgCount = 0
	godprogs = []
	maxagegp = 0
	playersgodprogged = []
	superlucky = {}
	
	@staticmethod
	def calculate_god_prog_chance(ovr):
		"""Calculate god progression chance based on OVR."""
		if ovr < Config.MIN_RATING:
			scale = 1.0
		elif ovr > Config.MAX_RATING:
			scale = 0.01
		else:
			scale = 1.0 - (ovr - Config.MIN_RATING) / (Config.MAX_RATING - Config.MIN_RATING)
		return scale * Config.MAX_GOD_PROG_CHANCE
	
	@classmethod
	def attempt_god_prog(cls, age, ovr, rng, name, seed):
		"""Attempt god progression for a player. Returns (min, max) tuple or None."""
		if age >= Config.YOUNG_MAX or rng.random() >= cls.calculate_god_prog_chance(ovr):
			return None
		
		# God progression activated
		prog_amount = rng.randint(Config.MIN_GOD_PROG, Config.MAX_GOD_PROG)
		
		# Track statistics
		cls.godProgCount += 1
		cls.maxagegp = max(cls.maxagegp, age)
		cls.superlucky[str(name)] = cls.superlucky.get(str(name), 0) + 1
		cls.playersgodprogged.append({
			'RunSeed': str(seed), 'Name': name, 'Age': str(age), 
			'Jump': str(prog_amount), 'OVR': str(ovr), 
			'Chance': str(cls.calculate_god_prog_chance(ovr))
		})
		
		return (prog_amount, prog_amount)


class ProgressionCalculator:
	"""Handles the core progression range calculations."""
	
	@staticmethod
	def get_age_params(age):
		"""Get progression parameters for a given age."""
		for (min_age, max_age), params in Config.PROG_PARAMS.items():
			if min_age <= age <= max_age:
				result = copy.deepcopy(params)
				result['age'] = age
				return result
		# Fallback to oldest bracket
		result = copy.deepcopy(Config.PROG_PARAMS[(35, 99)])
		result['age'] = age
		return result
	
	@staticmethod
	def calculate_base_range(per, params):
		"""Calculate base min/max progression range from PER and age params."""
		age = params['age']
		
		# Early progression rule for young, low-PER players
		if per <= Config.EARLY_PROG_PER_THRESHOLD and age < Config.EARLY_PROG_AGE_THRESHOLD:
			mn = math.ceil(per / Config.EARLY_PROG_PER_DIVISOR_MN) + Config.EARLY_PROG_PER_OFFSET_MN
			mx = math.ceil(per / Config.EARLY_PROG_PER_DIVISOR_MX) + Config.EARLY_PROG_PER_OFFSET_MX
		else:
			# Standard progression calculation
			mn = math.ceil(per / params['min1']) - params['min2']

			if params.get('max1') is not None and params.get('max2') is not None:
				mx = math.ceil(per / params['max1']) - params['max2']
			else:
				mx = Config.DEFAULT_MAX_PROG
		
		return mn, mx
	
	@staticmethod
	def apply_hard_limits(mn, mx, params):
		"""Apply hard min/max limits from config."""
		hardMin = params.get('hardMin')
		hardMax = params.get('hardMax')
		
		# Direct assignment if hardMin is set (only for specific cases)
		if hardMin is not None:
			mn = hardMin
		
		# JS: if ((hardMax && max > hardMax) || max > hardMax) max = hardMax;
		# The OR condition is redundant, simplifies to: if (hardMax && max > hardMax)
		if hardMax is not None and mx > hardMax:
			mx = hardMax
		
		return mn, mx
	
	@staticmethod
	def apply_ovr_cap_logic(mn, mx, ovr, age, rng):
		"""Apply OVR cap logic and age-based adjustments."""
		ovrProgression = mx + ovr
		flagLower = mn + ovr
		
		# Check if progression would exceed cap
		if ovrProgression >= Config.MAX_OVR:
			if ovr >= Config.MAX_OVR:
				# Already at cap - enforce Hard Max: 0
				mx = 0
				if age > 30 and age < 35:
					mn = -10
				if age >= 35:
					mn = -14
				if age <= 30:
					# JS: const randomMin = Utils.randomInt(-2, 0);
					# JS: if (randomMin < 0.02) adjustedMin = -2;
					# This generates a value in [-2, 0], checks if it's < 0.02
					# Since randInt(-2, 0) gives {-2, -1, 0}, only -2 satisfies < 0.02, suspect this is a bug in JS
					randomMin = rng.randint(-2, 0)
					if randomMin < 0.02:
						mn = -2
				# Prevent inverted range
				if mn > mx:
					mn = 0
			else:
				# Approaching cap
				mx = Config.MAX_OVR - ovr
				if flagLower >= Config.MAX_OVR:
					mn = 0
		
		return int(mn), int(mx)
	
	@classmethod
	def get_progression_range(cls, per, age, ovr, rng):
		"""Main entry point for calculating progression range."""
		params = cls.get_age_params(age)
		mn, mx = cls.calculate_base_range(per, params)
		mn, mx = cls.apply_hard_limits(mn, mx, params)
		return cls.apply_ovr_cap_logic(mn, mx, ovr, age, rng)


class AttributeProgression:
	"""Handles individual attribute progression logic."""
	
	@staticmethod
	def apply_physical_caps(attr, age, mn, mx, rng):
		"""Apply physical attribute caps for older players. Returns (mn, mx, skip_prog)."""
		# CRITICAL FIX: Check age >= 30, not age < 30
		if age < 30 or attr not in Config.PHYSICAL_OLD or mx <= 0:
			return mn, mx, False
		
		# Old player physical progression chance
		oldProgPhys = rng.random() * 0.05 + 0.01
		if rng.random() >= oldProgPhys:
			return 0, 0, True  # Skip progression entirely
		
		# Cap progression for old physical attributes
		if mx > 3:
			mx = 3
		
		return mn, mx, False
	
	@staticmethod
	def apply_mid_age_slowdown(attr, age, prog, rng):
		"""Apply mid-age slowdown for physical attributes."""
		if not (25 <= age < 30 and attr in Config.PHYSICAL_MID and prog > 0):
			return prog
		
		age_factor = 0.7 - (age - 25) * 0.1
		prob_progression = max(age_factor, 0)
		
		# JS: return Math.random() > probProgression;
		# Returns true to SKIP, so we invert: keep prog if random() <= prob
		return prog if rng.random() <= prob_progression else 0
	
	@classmethod
	def progress_attribute(cls, attr, age, mn, mx, current_rating, rng):
		"""Progress a single attribute with all applicable rules."""
		# Apply physical caps first
		mn, mx, skip = cls.apply_physical_caps(attr, age, mn, mx, rng)
		if skip:
			return current_rating
		
		# Calculate base progression
		prog = rng.randint(mn, mx) if mn <= mx else 0
		
		# Apply mid-age slowdown and return bounded result
		prog = cls.apply_mid_age_slowdown(attr, age, prog, rng)
		return max(0, min(Config.MAX_OVR, current_rating + prog))


class progsandbox:
	"""Main progression sandbox for basketball player development."""
	
	def __init__(self, seed=0):
		self.ATTRS = Config.ALL_ATTRS
		self.master_seed = seed
		self.master_rng = random.Random(seed)
		self.SEED = seed
	
	def calcovr(self, attrs_dict):
		"""
		Calculate overall rating from attributes dictionary.
		CRITICAL: Maintains exact same logic and coefficients as bbgm ovr calculation script: https://github.com/zengm-games/zengm/blob/master/src/worker/core/player/ovr.basketball.ts.
		"""
		# Extract values in the exact order required
		vals = np.array([float(attrs_dict.get(col, 0)) for col in Config.OVR_CALC_ORDER], dtype=float)
		
		# Weighted sum
		s = float((vals - Config.OVR_CENTERS) @ Config.OVR_COEFFS + 48.5)
		
		# Piecewise fudge factor (unchanged from original)
		if s >= 68:
			fudge = 8.0
		elif s >= 50:
			fudge = 4.0 + (s - 50.0) * (4.0 / 18.0)
		elif s >= 42:
			fudge = -5.0 + (s - 42.0) * (9.0 / 8.0)
		elif s >= 31:
			fudge = -5.0 - (42.0 - s) * (5.0 / 11.0)
		else:
			fudge = -10.0
		
		return max(0, min(100, int(round(s + fudge))))
	
	def runoneprog(self, roster_df, rng, seed):
		"""
		Apply PER-based progression to each player in the roster.
		"""
		rng = rng or self.master_rng
		out = roster_df.copy(deep=True)
		
		# Extract data arrays for vectorized operations
		ages = out["Age"].to_numpy(dtype=int)
		pers = np.where(np.isfinite(out["PER"].to_numpy(dtype=float)), 
						out["PER"].to_numpy(dtype=float), 0.0)
		names = out.get("Name", pd.Series(range(len(out)))).to_numpy(dtype=str)
		
		# Prepare attribute arrays
		attr_arrays = {attr: out.get(attr, 0).to_numpy(copy=True, dtype=float) 
					   for attr in self.ATTRS}
		
		# Initialize OVR if not present
		ovr_array = out.get("Ovr", 0).to_numpy(copy=True, dtype=float)
		
		# Progressive attributes (excluding height)
		prog_attrs = [k for k in self.ATTRS if k != 'Hgt']
		
		# Process each player
		for i in range(len(out)):
			age, per, name = ages[i], pers[i], names[i]
			
			# Skip conditions
			if age < 25 or per <= 0:
				continue
			
			# Build current ratings dict
			ratings = {attr: int(round(attr_arrays[attr][i])) for attr in self.ATTRS}
			ovr = self.calcovr(ratings)
			
			# Get progression range
			mn, mx = ProgressionCalculator.get_progression_range(per, age, ovr, rng)
			
			# Check for god progression
			if (god_prog := GodProgSystem.attempt_god_prog(age, ovr, rng, name, seed)):
				mn, mx = god_prog
			
			# Apply progression to each attribute
			for attr in prog_attrs:
				new_rating = AttributeProgression.progress_attribute(
					attr, age, mn, mx, ratings[attr], rng
				)
				attr_arrays[attr][i] = new_rating
				ratings[attr] = new_rating
			
			# Recalculate OVR with updated ratings
			ovr_array[i] = self.calcovr(ratings)
		
		# Write updated arrays back to DataFrame
		for attr, arr in attr_arrays.items():
			out[attr] = arr
		out["Ovr"] = ovr_array
		
		return out
