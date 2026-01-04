import math
import random
import numpy as np

"""
Progression utilities (reworked for NoEyeTest v4.0, removed complex class structure since it really is just a single function call to fix them all)

references: https://github.com/fearandesire/NoEyeTest/blob/dev/tiers.md
			https://github.com/zengm-games/zengm/blob/master/src/worker/core/player/ovr.basketball.ts
			noeyetest.js (https://github.com/fearandesire/NoEyeTest/blob/dev)
            plan to update noeyetest.js https://github.com/akshayexists/progbox/blob/updatealgo/NETv4.md
"""

# Configuration
class Config:
    # Attribute bounds
    ATTR_MIN = 0
    ATTR_MAX = 100
    
    # OVR calculation (kept identical to original)
    OVR_CALC_ORDER = ['Hgt', 'Str', 'Spd', 'Jmp', 'End', 'Ins',
                      'Dnk', 'FT', '3Pt', 'oIQ', 'dIQ', 'Drb',
                      'Pss', '2Pt', 'Reb']
    OVR_COEFFS = np.array([0.159, 0.0777, 0.123, 0.051, 0.0632, 0.0126,
                           0.0286, 0.0202, 0.0726, 0.133, 0.159, 0.059,
                           0.062, 0.01, 0.01], dtype=float)
    OVR_CENTERS = np.array([47.5, 50.2, 50.8, 48.7, 39.9, 42.4,
                            49.5, 47.0, 47.1, 46.8, 46.7, 54.8,
                            51.3, 47.0, 51.4], dtype=float)
    
    # Attribute groups
    PHYSICAL_ATTRS = {'Spd', 'Str', 'Jmp', 'End'}
    SKILL_ATTRS = {'Drb', '2Pt', '3Pt', 'FT', 'Pss', 'Ins', 'Dnk'}
    MENTAL_ATTRS = {'oIQ', 'dIQ', 'Reb'}
    ALL_ATTRS = ['dIQ', 'Dnk', 'Drb', 'End', '2Pt', 'FT', 'Ins', 'Jmp',
                 'oIQ', 'Pss', 'Reb', 'Spd', 'Str', '3Pt', 'Hgt']
    NUMCOLS = ["Age", "PER", "DWS", "BLK", "STL"] + ALL_ATTRS
    
    # Defense Adjustment
    DEFENSIVE_WEIGHTS = {'dws': 0.4, 'stl': 0.3, 'blk': 0.3}
    DEFENSIVE_BONUS_CAP = 3.0
    
    # Budget Calculation
    BASE_DIVISOR = 3.0
    JITTER_RANGE = 0.5
    AGE_MULTIPLIERS = {
        25: 1.0, 26: 1.0, 27: 1.0,
        28: 0.5, 29: 0.5, 30: 0.5,
        31: -0.3, 32: -0.3, 33: -0.3, 34: -0.3,
        35: -0.8  # 36+ same
    }
    LOW_PER_BONUS = {'ageThreshold': 28, 'perThreshold': 12, 'bonus': 2}
    GOD_PROGRESSION = {
        'baseRate': 0.03,
        'ovrPenalty': 0.025,
        'oldAgeMultiplier': 0.5,
        'ageThreshold': 28,
        'bonusRange': [5, 10]
    }
    
    # Distribution
    BASE_WEIGHTS = {'physical': 0.25, 'shooting': 0.30, 'playmaking': 0.20, 'defense': 0.25}
    AGED_WEIGHTS = {'physical': 0.10, 'shooting': 0.35, 'playmaking': 0.25, 'defense': 0.30}
    CATEGORY_SKILLS = {
        'physical': ['Spd', 'Str', 'Jmp', 'End'],
        'shooting': ['2Pt', 'FT', '3Pt', 'Dnk'],
        'playmaking': ['Pss', 'Drb', 'oIQ'],
        'defense': ['dIQ', 'Ins', 'Reb']
    }
    
    # Caps & Limits
    SKILL_CAPS = {
        '25-30': {'min': -2, 'max': 4},
        '31-34': {'min': -3, 'max': 2},
        '35+': {'min': -5, 'max': 0}
    }
    PHYSICAL_DECLINE_AGE = 30
    PHYSICAL_SKILLS = ['Spd', 'Jmp', 'End']
    OVR_HARD_CAP = 80
    
    # Validation
    MIN_AGE = 25
    
    # Debug
    DEBUG = False
    NOTIFY_GOD_PROGS = True
    NOTIFY_ALL_PROGS = False


# OVR calculation
def calcovr(attrs_dict):
    """Calculate OVR from attribute dictionary"""
    vals = np.array([float(attrs_dict.get(col, 0)) for col in Config.OVR_CALC_ORDER], dtype=float)
    s = float((vals - Config.OVR_CENTERS) @ Config.OVR_COEFFS + 48.5)
    
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

# Helper functions
def get_age_multiplier(age):
    """Get age multiplier for budget calculation"""
    if age >= 35:
        return Config.AGE_MULTIPLIERS[35]
    return Config.AGE_MULTIPLIERS.get(age, 1.0)

def get_skill_caps(age):
    """Get min/max caps for skill changes based on age"""
    if 25 <= age <= 30:
        return Config.SKILL_CAPS['25-30']
    elif 31 <= age <= 34:
        return Config.SKILL_CAPS['31-34']
    else:  # 35+
        return Config.SKILL_CAPS['35+']

def calculate_defensive_bonus(dws, stl, blk):
    """Calculate defense-adjusted bonus (Phase 2)"""
    bonus = (dws * Config.DEFENSIVE_WEIGHTS['dws'] + 
             stl * Config.DEFENSIVE_WEIGHTS['stl'] + 
             blk * Config.DEFENSIVE_WEIGHTS['blk'])
    return min(bonus, Config.DEFENSIVE_BONUS_CAP)

def calculate_base_budget(adjusted_per, age, rng):
    """Calculate base budget with age multiplier and jitter (Phase 3)"""
    age_mult = get_age_multiplier(age)
    raw_budget = (adjusted_per / Config.BASE_DIVISOR) * age_mult
    jitter = rng.uniform(-Config.JITTER_RANGE, Config.JITTER_RANGE)
    return math.floor(raw_budget + jitter)

def apply_budget_adjustments(budget, adjusted_per, age, ovr, rng):
    """Apply low-PER bonus and god progression (Phase 4)"""
    is_god_prog = False
    
    # Low-PER bonus
    if age <= Config.LOW_PER_BONUS['ageThreshold'] and adjusted_per < Config.LOW_PER_BONUS['perThreshold']:
        budget += Config.LOW_PER_BONUS['bonus']
    
    # God progression
    god_prob = (Config.GOD_PROGRESSION['baseRate'] - 
                (ovr / 100) * Config.GOD_PROGRESSION['ovrPenalty'])
    if age > Config.GOD_PROGRESSION['ageThreshold']:
        god_prob *= Config.GOD_PROGRESSION['oldAgeMultiplier']
    
    if rng.random() < god_prob:
        god_bonus = rng.randint(*Config.GOD_PROGRESSION['bonusRange'])
        budget += god_bonus
        is_god_prog = True
    
    return budget, is_god_prog

def distribute_to_categories(budget, age):
    """Distribute budget to skill categories (Phase 5)"""
    weights = Config.AGED_WEIGHTS if age >= 30 else Config.BASE_WEIGHTS
    
    category_budgets = {}
    allocated = 0
    
    for category, weight in weights.items():
        cat_budget = round(budget * weight)
        category_budgets[category] = cat_budget
        allocated += cat_budget
    
    # Redistribute rounding error to priority category
    error = budget - allocated
    if error != 0:
        priority = 'defense' if age >= 30 else 'physical'
        category_budgets[priority] += error
    
    return category_budgets

def distribute_to_skills(category_budgets, rng):
    """Distribute category budgets to individual skills (Phase 5)"""
    skill_changes = {}
    
    for category, cat_budget in category_budgets.items():
        skills = Config.CATEGORY_SKILLS[category]
        skill_count = len(skills)
        
        if skill_count == 0:
            continue
        
        base_per_skill = cat_budget / skill_count
        
        for skill in skills:
            jitter = rng.uniform(-0.5, 0.5)
            change = round(base_per_skill + jitter)
            skill_changes[skill] = change
    
    return skill_changes

def enforce_skill_caps(skill_changes, age):
    """Apply per-skill caps (Phase 6)"""
    caps = get_skill_caps(age)
    
    for skill in skill_changes:
        skill_changes[skill] = max(caps['min'], min(caps['max'], skill_changes[skill]))

def enforce_physical_decline(skill_changes, age):
    """Prevent physical attributes from increasing for age 30+ (Phase 6)"""
    if age >= Config.PHYSICAL_DECLINE_AGE:
        for skill in Config.PHYSICAL_SKILLS:
            if skill in skill_changes:
                skill_changes[skill] = min(skill_changes[skill], 0)

def enforce_ovr_cap(skill_changes, current_attrs_array):
    """Scale positive changes to respect OVR cap (Phase 6)"""
    # Work with array directly, avoiding dict copies
    projected = current_attrs_array.copy()
    
    # Apply changes
    for i, skill in enumerate(Config.ALL_ATTRS):
        if skill in skill_changes:
            projected[i] = max(Config.ATTR_MIN, min(Config.ATTR_MAX, 
                                                     projected[i] + skill_changes[skill]))
    
    # Fast OVR calc using array slicing
    projected_ovr = calcovr_from_array(projected)
    
    if projected_ovr > Config.OVR_HARD_CAP:
        scale_factor = 1 - ((projected_ovr - Config.OVR_HARD_CAP) / projected_ovr)
        
        for skill in skill_changes:
            if skill_changes[skill] > 0:
                skill_changes[skill] = int(skill_changes[skill] * scale_factor)

def calcovr_from_array(attrs_array):
    """Calculate OVR directly from numpy array (optimized)"""
    # Map array indices to OVR calc order
    indices = [Config.ALL_ATTRS.index(attr) for attr in Config.OVR_CALC_ORDER]
    vals = attrs_array[indices]
    s = float((vals - Config.OVR_CENTERS) @ Config.OVR_COEFFS + 48.5)
    
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


# Core progression function
def progplayer(player_row, rng):
    """
    Progress a single player based on NoEyeTest v4.0 spec.
    player_row: numpy array with [Age, PER, DWS, BLK, STL, ...attributes]
    Returns: modified numpy array (in-place for efficiency)
    """
    # Extract values
    age = int(player_row[0])
    
    # Phase 1: Validation
    if age < Config.MIN_AGE:
        return player_row
    
    per = player_row[1]
    dws = player_row[2]
    blk = player_row[3]
    stl = player_row[4]
    
    # Get current attributes as array slice (zero-copy view)
    current_attrs = player_row[5:5+len(Config.ALL_ATTRS)]
    current_ovr = calcovr_from_array(current_attrs)
    
    # Phase 2: Defense-Adjusted PER
    defensive_bonus = calculate_defensive_bonus(dws, stl, blk)
    adjusted_per = per + defensive_bonus
    
    # Phase 3: Budget Calculation
    budget = calculate_base_budget(adjusted_per, age, rng)
    
    # Phase 4: Budget Adjustments
    budget, is_god_prog = apply_budget_adjustments(budget, adjusted_per, age, current_ovr, rng)
    
    # Phase 5: Distribution
    category_budgets = distribute_to_categories(budget, age)
    skill_changes = distribute_to_skills(category_budgets, rng)
    
    # Phase 6: Enforcement
    enforce_skill_caps(skill_changes, age)
    enforce_physical_decline(skill_changes, age)
    enforce_ovr_cap(skill_changes, current_attrs)
    
    # Phase 7: Apply changes (in-place)
    for i, attr in enumerate(Config.ALL_ATTRS):
        if attr in skill_changes:
            new_val = current_attrs[i] + skill_changes[attr]
            player_row[5 + i] = max(Config.ATTR_MIN, min(Config.ATTR_MAX, new_val))
    
    return player_row

# Tracking
class tracking:
    def __init__(self):
        self.godprogcount = 0
        self.playersgodprogged = {}
    
    def updategodprog(self, player_name):
        self.godprogcount += 1
        if player_name in self.playersgodprogged:
            self.playersgodprogged[player_name] += 1
        else:
            self.playersgodprogged[player_name] = 1

#--------------
# Main sandbox
class progsandbox:
    def __init__(self, seed=0):
        self.master_seed = seed
        self.master_rng = random.Random(seed)
        self.tracking = tracking()
    
    def runoneprog(self, roster_df, rng=None, seed=None):
        """
        Apply progression to each player in roster_df.
        - roster_df: pandas DataFrame with player rows (must contain Age, PER, DWS, BLK, STL and attribute columns)
        - rng: random.Random-like instance (if None, uses internal RNG)
        - seed: opaque seed stored for god prog logs
        Returns updated DataFrame (copy).
        """
        rng = rng or self.master_rng
        out = roster_df.copy(deep=True)
        
        # Ensure required columns exist
        required_cols = Config.NUMCOLS + ['Team', 'Name']
        for col in required_cols:
            if col not in out.columns:
                if col in ['Team', 'Name']:
                    out[col] = ''
                else:
                    out[col] = 0
        
        # Ensure Ovr column exists
        if 'Ovr' not in out.columns:
            out['Ovr'] = 0
        
        # Extract numeric data (zero-copy view where possible)
        num_arr = out[Config.NUMCOLS].to_numpy(dtype=np.float64)
        
        # Pre-compute OVR index mapping for vectorized calculation
        ovr_indices = [Config.ALL_ATTRS.index(attr) for attr in Config.OVR_CALC_ORDER]
        
        # Progress each player (vectorized operations where possible)
        god_prog_indices = []
        for i in range(len(num_arr)):
            age = int(num_arr[i, 0])
            if age < Config.MIN_AGE:
                continue
            
            # Track changes for god prog detection
            original_attrs = num_arr[i, 5:].copy()
            
            # In-place progression
            progplayer(num_arr[i], rng)
            
            # Fast god prog detection using numpy
            changes = num_arr[i, 5:] - original_attrs
            if np.sum(changes > 0) >= 5 and np.sum(changes) >= 8:
                god_prog_indices.append(i)
        
        # Write back numeric data
        out[Config.NUMCOLS] = num_arr
        
        # Vectorized OVR calculation
        attr_arrays = num_arr[:, 5:5+len(Config.ALL_ATTRS)]
        ovr_vals = np.zeros(len(out), dtype=int)
        
        for i in range(len(out)):
            ovr_vals[i] = calcovr_from_array(attr_arrays[i])
        
        out['Ovr'] = ovr_vals
        
        # Update god prog tracking
        if god_prog_indices:
            for idx in god_prog_indices:
                player_name = out.loc[idx, 'Name'] if 'Name' in out.columns else f"Player_{idx}"
                self.tracking.updategodprog(player_name)
        
        return out
