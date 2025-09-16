import numpy as np
from collections import defaultdict

class Config:
    """Centralized configuration matching original structure."""
    
    # OVR Calculation Constants (EXACT - DO NOT CHANGE FOR COMPATIBILITY)
    OVR_CALC_ORDER = ['Hgt', 'Str', 'Spd', 'Jmp', 'End', 'Ins', 'Dnk', 'FT', '3Pt',
                      'oIQ', 'dIQ', 'Drb', 'Pss', '2Pt', 'Reb']
    OVR_COEFFS = np.array([0.159, 0.0777, 0.123, 0.051, 0.0632, 0.0126, 0.0286,
                           0.0202, 0.0726, 0.133, 0.159, 0.059, 0.062, 0.01, 0.01])
    OVR_CENTERS = np.array([47.5, 50.2, 50.8, 48.7, 39.9, 42.4, 49.5, 47.0, 47.1,
                            46.8, 46.7, 54.8, 51.3, 47.0, 51.4])

    # Attribute Categories
    ATTRIBUTE_CATEGORIES = {
        'Physical': ['Str', 'Spd', 'Jmp', 'End', 'Reb'],
        'Technical': ['Ins', 'Dnk', 'FT', '3Pt', 'Drb', 'Pss', '2Pt'],
        'Mental': ['oIQ', 'dIQ']
    }
    ATTR_TO_CAT = {attr: cat for cat, attrs in ATTRIBUTE_CATEGORIES.items() for attr in attrs}
    
    # Performance scaling
    PER_SCALING_FACTOR = 40.0  # Increased to reduce PER impact and limit power creep
    
    # God Progression - refined parameters
    GOD_PROG_AGE_LIMIT = 30
    GOD_PROG_JUMP_MIN = 6
    GOD_PROG_JUMP_MAX = 12
    GOD_PROG_CHANCE_SCALE = {
        'MIN_RATING': 0.0, 'MAX_CHANCE': 0.09,  # Slightly higher base chance
        'MAX_RATING': 61.0, 'MIN_CHANCE': 0.01  # Steeper decay
    }
    
    # Beta distribution parameters for progression
    # Format: alpha, beta, scale (multiplier for final result), shift (offset)
    BETA_PROGRESSION_PARAMS = {
        '25-29': {
            'Physical':  {'alpha': 2.0, 'beta': 4.0, 'scale': 4.0, 'shift': -1.5},  # Slight decline bias
            'Technical': {'alpha': 3.0, 'beta': 2.5, 'scale': 4.5, 'shift': -1.0},  # Growth bias
            'Mental':    {'alpha': 4.0, 'beta': 2.0, 'scale': 3.0, 'shift': -0.5},  # Strong growth bias
        },
        '30-33': {
            'Physical':  {'alpha': 1.5, 'beta': 3.5, 'scale': 3.5, 'shift': -2.0},  # Decline starts
            'Technical': {'alpha': 2.5, 'beta': 3.0, 'scale': 3.5, 'shift': -1.2},  # Neutral to slight decline
            'Mental':    {'alpha': 3.0, 'beta': 2.5, 'scale': 2.5, 'shift': -0.5},  # Still growing but slower
        },
        '34-36': {
            'Physical':  {'alpha': 1.0, 'beta': 4.0, 'scale': 3.0, 'shift': -2.5},  # Clear decline
            'Technical': {'alpha': 2.0, 'beta': 3.5, 'scale': 2.8, 'shift': -1.5},  # Decline sets in
            'Mental':    {'alpha': 2.5, 'beta': 3.0, 'scale': 2.0, 'shift': -0.8},  # Slower growth
        },
        '37+': {
            'Physical':  {'alpha': 0.8, 'beta': 5.0, 'scale': 2.5, 'shift': -3.0},  # Strong decline
            'Technical': {'alpha': 1.5, 'beta': 4.0, 'scale': 2.2, 'shift': -2.0},  # Noticeable decline
            'Mental':    {'alpha': 2.0, 'beta': 3.5, 'scale': 1.5, 'shift': -1.0},  # Minimal change
        }
    }
    
    # Refined hard caps - slightly tighter to limit power creep
    HARD_CAPS = {
        '25-29': {'Physical': {'min': -2, 'max': 2}, 'Technical': {'min': -1, 'max': 3}, 'Mental': {'min': 0, 'max': 2}},
        '30-33': {'Physical': {'min': -3, 'max': 1}, 'Technical': {'min': -2, 'max': 2}, 'Mental': {'min': -1, 'max': 2}},
        '34-36': {'Physical': {'min': -4, 'max': 0}, 'Technical': {'min': -3, 'max': 1}, 'Mental': {'min': -1, 'max': 1}},
        '37+':   {'Physical': {'min': -5, 'max': 0}, 'Technical': {'min': -4, 'max': 0}, 'Mental': {'min': -2, 'max': 0}},
    }

class GodProgSystem:
    """God progression system with verbose logging."""
    godProgCount = 0
    maxagegp = 0
    playersgodprogged = []
    superlucky = defaultdict(int)

    @classmethod
    def reset(cls):
        cls.godProgCount = 0
        cls.maxagegp = 0
        cls.playersgodprogged = []
        cls.superlucky.clear()

    @classmethod
    def _calculate_chance(cls, rating):
        scale = Config.GOD_PROG_CHANCE_SCALE
        min_r, max_c = scale['MIN_RATING'], scale['MAX_CHANCE']
        max_r, min_c = scale['MAX_RATING'], scale['MIN_CHANCE']

        if rating <= min_r: return max_c
        if rating >= max_r: return min_c * max_c

        # Smoother exponential decay
        fraction = (max_r - rating) / (max_r - min_r)
        chance = max_c * (fraction ** 3.5)  # Slightly less steep
        return max(chance, min_c)

    @classmethod
    def process(cls, player, player_id, ovr, rng, seed):
        age = int(player.get('Age', 0))
        if age >= Config.GOD_PROG_AGE_LIMIT:
            return None
        chance = cls._calculate_chance(ovr)

        if rng.random() < chance:
            jump = rng.randint(Config.GOD_PROG_JUMP_MIN, Config.GOD_PROG_JUMP_MAX)

            # Verbose logging
            cls.godProgCount += 1
            cls.maxagegp = max(cls.maxagegp, age)
            cls.superlucky[player.get('Name', 'N/A')] += 1
            cls.playersgodprogged.append({
                'RunSeed': seed, 'PlayerID': player_id, 'Name': player.get('Name', 'N/A'),
                'Age': age, 'Jump': jump, 'OVR': ovr, 'Chance': chance
            })

            # Return jump for all relevant attributes
            all_attrs = [attr for attr in Config.OVR_CALC_ORDER if attr in Config.ATTR_TO_CAT]
            return {attr: jump for attr in all_attrs}

        return None

class NormalProgressionEngine:
    """Beta distribution-based progression engine."""
    
    def _get_age_bracket(self, age):
        if 25 <= age <= 29: return '25-29'
        if 30 <= age <= 33: return '30-33'
        if 34 <= age <= 36: return '34-36'
        if age >= 37: return '37+'
        return None

    def process(self, player, attr, np_rng):
        age_bracket = self._get_age_bracket(int(player.get('Age', 0)))
        category = Config.ATTR_TO_CAT.get(attr)

        if not age_bracket or not category:
            return 0
            
        params = Config.BETA_PROGRESSION_PARAMS[age_bracket][category]
        caps = Config.HARD_CAPS[age_bracket][category]
        
        # Get PER adjustment (reduced impact to limit power creep)
        per = float(player.get('PER', 0.0))
        per_adj = per / Config.PER_SCALING_FACTOR
        
        # Generate beta distribution sample
        beta_sample = np_rng.beta(params['alpha'], params['beta'])
        
        # Transform to progression value
        roll = (beta_sample * params['scale']) + params['shift'] + per_adj
        
        # Apply regression to mean for extreme ratings (anti-power creep)
        current_val = float(player.get(attr, 40))
        if current_val > 70:
            roll -= 0.4  # Stronger regression for high ratings
        elif current_val > 60:
            roll -= 0.2  # Moderate regression for above-average ratings
        elif current_val < 35:
            roll += 0.15  # Small boost for very low ratings
            
        # Apply hard caps
        clamped_roll = max(caps['min'], min(roll, caps['max']))
        return round(clamped_roll)

class progsandbox:
    """Main progression sandbox - compatible with existing harness."""
    
    def __init__(self, seed):
        self.ATTRS = Config.OVR_CALC_ORDER
        self._normal_engine = NormalProgressionEngine()

    def calcovr(self, attrs_dict):
        """Calculate OVR using exact original formula."""
        vals = np.array([float(attrs_dict.get(col, 0) or 0) for col in Config.OVR_CALC_ORDER])
        s = float((vals - Config.OVR_CENTERS) @ Config.OVR_COEFFS + 48.5)
        
        # Original fudge factor logic
        if s >= 68: fudge = 8.0
        elif s >= 50: fudge = 4.0 + (s - 50.0) * (4.0 / 18.0)
        elif s >= 42: fudge = -5.0 + (s - 42.0) * (9.0 / 8.0)
        elif s >= 31: fudge = -5.0 - (42.0 - s) * (5.0 / 11.0)
        else: fudge = -10.0
            
        return int(max(0, min(100, round(s + fudge))))

    def runoneprog(self, df, rng, seed):
        """Run one season of progression - enhanced but compatible."""
        np_rng = np.random.default_rng(seed)
        prog_df = df.copy()
        
        for player_id, player in prog_df.iterrows():
            if int(player.get('Age', 0)) < 25:
                continue

            # Check for god progression
            god_jump = GodProgSystem.process(player, player_id, player.get('Ovr'), rng, seed)

            for attr in self.ATTRS:
                if attr == 'Hgt' or attr not in Config.ATTR_TO_CAT:
                    continue

                if god_jump:
                    final_change = god_jump[attr]
                else:
                    final_change = self._normal_engine.process(player, attr, np_rng)

                if final_change != 0:
                    current_rating = float(player.get(attr, 0.0))
                    new_rating = current_rating + final_change
                    prog_df.loc[player_id, attr] = max(0, min(80, new_rating))

        return prog_df
