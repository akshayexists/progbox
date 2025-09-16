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
    PER_SCALING_FACTOR = 80.0  # Reduced impact for more balanced progression
    
    # God Progression - refined parameters
    GOD_PROG_AGE_LIMIT = 30
    GOD_PROG_JUMP_MIN = 6
    GOD_PROG_JUMP_MAX = 12
    GOD_PROG_CHANCE_SCALE = {
        'MIN_RATING': 0.0, 'MAX_CHANCE': 0.004,  # Slightly higher base chance
        'MAX_RATING': 65.0, 'MIN_CHANCE': 0.0000001  # Steeper decay
    }
    
    # Refined progression parameters with smoother curves
    PROGRESSION_PARAMS = {
        '25-29': {
            'Physical':  {'dist': 'normal', 'mean': 0.1, 'sd': 0.9},
            'Technical': {'dist': 'normal', 'mean': 0.6, 'sd': 1.1},
            'Mental':    {'dist': 'normal', 'mean': 0.8, 'sd': 0.6},
        },
        '30-33': {
            'Physical':  {'dist': 'triangular', 'left': -1.5, 'mode': -0.2, 'right': 1.2},
            'Technical': {'dist': 'triangular', 'left': -0.8, 'mode': 0.2, 'right': 1.5},
            'Mental':    {'dist': 'triangular', 'left': -0.3, 'mode': 0.3, 'right': 1.0},
        },
        '34-36': {
            'Physical':  {'dist': 'triangular', 'left': -2.5, 'mode': -0.8, 'right': 0.5},
            'Technical': {'dist': 'triangular', 'left': -1.2, 'mode': -0.1, 'right': 0.8},
            'Mental':    {'dist': 'triangular', 'left': -0.4, 'mode': 0.1, 'right': 0.6},
        },
        '37+': {
            'Physical':  {'dist': 'triangular', 'left': -3.5, 'mode': -1.2, 'right': 0.0},
            'Technical': {'dist': 'triangular', 'left': -1.8, 'mode': -0.4, 'right': 0.3},
            'Mental':    {'dist': 'triangular', 'left': -0.8, 'mode': -0.1, 'right': 0.4},
        }
    }
    
    # Refined hard caps
    HARD_CAPS = {
        '25-29': {'Physical': {'min': -2, 'max': 3}, 'Technical': {'min': -1, 'max': 4}, 'Mental': {'min': 0, 'max': 3}},
        '30-33': {'Physical': {'min': -3, 'max': 2}, 'Technical': {'min': -2, 'max': 3}, 'Mental': {'min': -1, 'max': 2}},
        '34-36': {'Physical': {'min': -4, 'max': 1}, 'Technical': {'min': -3, 'max': 1}, 'Mental': {'min': -1, 'max': 1}},
        '37+':   {'Physical': {'min': -6, 'max': 0}, 'Technical': {'min': -4, 'max': 0}, 'Mental': {'min': -2, 'max': 0}},
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
        if rating >= max_r: return min_c

        # Smoother exponential decay
        fraction = (max_r - rating) / (max_r - min_r)
        chance = max_c * (fraction ** 1.8)  # Slightly less steep
        return max(chance, min_c)

    @classmethod
    def process(cls, player, player_id, rng, seed):
        age = int(player.get('Age', 0))
        if age >= Config.GOD_PROG_AGE_LIMIT:
            return None

        current_OVR = float(player.get('OVR', 0.0))
        chance = cls._calculate_chance(current_OVR)

        if rng.random() < chance:
            jump = rng.randint(Config.GOD_PROG_JUMP_MIN, Config.GOD_PROG_JUMP_MAX)

            # Verbose logging
            cls.godProgCount += 1
            cls.maxagegp = max(cls.maxagegp, age)
            cls.superlucky[player.get('Name', 'N/A')] += 1
            cls.playersgodprogged.append({
                'RunSeed': seed, 'PlayerID': player_id, 'Name': player.get('Name', 'N/A'),
                'Age': age, 'Jump': jump, 'OVR': current_OVR, 'Chance': chance
            })

            # Return jump for all relevant attributes
            all_attrs = [attr for attr in Config.OVR_CALC_ORDER if attr in Config.ATTR_TO_CAT]
            return {attr: jump for attr in all_attrs}

        return None

class NormalProgressionEngine:
    """Enhanced normal progression with synergy effects."""
    
    def _get_age_bracket(self, age):
        if 25 <= age <= 29: return '25-29'
        if 30 <= age <= 33: return '30-33'
        if 34 <= age <= 36: return '34-36'
        if age >= 37: return '37+'
        return None

    def _get_synergy_modifier(self, player, attr):
        """Simple synergy bonus based on related attributes."""
        category = Config.ATTR_TO_CAT.get(attr)
        if not category:
            return 0
            
        related_attrs = Config.ATTRIBUTE_CATEGORIES[category]
        avg_rating = np.mean([float(player.get(a, 50)) for a in related_attrs])
        
        # Small bonus/penalty based on related skill levels
        if avg_rating > 65:
            return 0.2
        elif avg_rating < 40:
            return -0.1
        return 0

    def process(self, player, attr, np_rng):
        age_bracket = self._get_age_bracket(int(player.get('Age', 0)))
        category = Config.ATTR_TO_CAT.get(attr)

        if not age_bracket or not category:
            return 0
            
        params = Config.PROGRESSION_PARAMS[age_bracket][category]
        caps = Config.HARD_CAPS[age_bracket][category]
        per = float(player.get('PER', 15.0))  # Default PER baseline
        per_adj = (per - 15.0) / Config.PER_SCALING_FACTOR  # Normalized PER impact
        
        # Base roll with PER adjustment
        if params['dist'] == 'normal':
            roll = np_rng.normal(loc=params['mean'] + per_adj, scale=params['sd'])
        elif params['dist'] == 'triangular':
            roll = np_rng.triangular(left=params['left'], mode=params['mode'] + per_adj, right=params['right'])
        else:
            roll = 0
        
        # Add synergy modifier
        synergy_mod = self._get_synergy_modifier(player, attr)
        roll += synergy_mod
        
        # Apply regression to mean for extreme values
        current_val = float(player.get(attr, 50))
        if current_val > 70:
            roll -= 0.3  # High ratings regress slightly
        elif current_val < 35:
            roll += 0.2  # Low ratings get small boost
            
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
            god_jump = GodProgSystem.process(player, player_id, rng, seed)

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
