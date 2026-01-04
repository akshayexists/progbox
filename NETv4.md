# Plan for NoEyeTest v4.0

made by [fenix](https://github.com/fearandesire)

---


### v4.0 Must Achieve
- ✅ Zero cap violations (mathematically provable)
- ✅ Defensive contribution valued (explicit DWS/STL/BLK weight)
- ✅ Single-pass execution (no mutation loops)
- ✅ Traceable logic (linear input → output)
- ✅ Tunable system (centralized config)

---

## Design Principles

1. **Immutability:** Calculate budget once, no mutation of intermediate values
2. **Single Source of Truth:** One budget pool, one age multiplier lookup, one cap pass
3. **Transparency:** Explicit formulas, named intermediates, commented rationale
4. **Fail-Safe:** Enforce caps before applying, validate inputs, log violations

---

## Mathematical Model

### Input Requirements
```javascript
{ per, dws, stl, blk, age, ovr, ratings... }
```

### Step 1: Defense-Adjusted PER
```javascript
defensiveBonus = (dws × 0.4) + (stl × 0.3) + (blk × 0.3)
cappedBonus = min(defensiveBonus, 3.0)
adjustedPER = per + cappedBonus
```

### Step 2: Age Multiplier
| Age | Multiplier |
|-----|------------|
| 25-27 | 1.0 |
| 28-30 | 0.5 |
| 31-34 | -0.3 |
| 35+ | -0.8 |

### Step 3: Base Budget
```javascript
rawBudget = (adjustedPER / 3.0) × ageMultiplier
jitter = uniform(-0.5, 0.5)
budget = floor(rawBudget + jitter)
```

### Step 4: Budget Adjustments

**Low-PER Bonus:**
```javascript
if (age <= 28 && adjustedPER < 12) budget += 2;
```

**God Progression:**
```javascript
godProbability = (0.03 - (ovr / 100) × 0.025) × (age <= 28 ? 1.0 : 0.5);
if (random() < godProbability) budget += randomInt(5, 10);
```

| OVR | Age ≤28 | Age >28 |
|-----|---------|---------|
| 60 | 1.5% | 0.75% |
| 70 | 1.25% | 0.625% |
| 80 | 0.5% | 0.25% |

### Step 5: Budget Distribution

**Category Weights:**
| Category | Base (25-29) | Aged (30+) | Skills |
|----------|--------------|------------|--------|
| Physical | 25% | 10% | spd, stre, jmp, endu |
| Shooting | 30% | 35% | fg, ft, tp, dnk |
| Playmaking | 20% | 25% | pss, drb, oiq |
| Defense | 25% | 30% | diq, ins, reb |

**Algorithm:**
```javascript
// 1. Allocate to categories
categoryBudgets[cat] = round(budget × weight)

// 2. Redistribute rounding error to priority category (defense if 30+, physical if <30)

// 3. Per-skill allocation
skillChange = round((categoryBudget / skillCount) + uniform(-0.5, 0.5))
```

### Step 6: Enforcement & Caps

**Per-Skill Caps:**
| Age Range | Min | Max |
|-----------|-----|-----|
| 25-30 | -2 | +4 |
| 31-34 | -3 | +2 |
| 35+ | -5 | 0 |

**Physical Decline (Age 30+):**
```javascript
// spd, jmp, endu cannot increase
skillChanges[physicalSkill] = min(skillChanges[physicalSkill], 0)
```

**OVR Cap (80 Max):**
```javascript
projectedOVR = calculateOVR(currentRatings + skillChanges)
if (projectedOVR > 80) {
  scaleFactor = 1 - ((projectedOVR - 80) / projectedOVR)
  // Scale positive changes only, preserve negatives
  positiveChanges *= scaleFactor
}
```

---

## Configuration Constants

```javascript
const CONFIG = {
  // Defense Adjustment
  DEFENSIVE_WEIGHTS: { dws: 0.4, stl: 0.3, blk: 0.3 },
  DEFENSIVE_BONUS_CAP: 3.0,

  // Budget Calculation
  BASE_DIVISOR: 3.0,
  JITTER_RANGE: 0.5,
  AGE_MULTIPLIERS: {
    25: 1.0, 26: 1.0, 27: 1.0,
    28: 0.5, 29: 0.5, 30: 0.5,
    31: -0.3, 32: -0.3, 33: -0.3, 34: -0.3,
    35: -0.8 // 36+ same
  },
  LOW_PER_BONUS: { ageThreshold: 28, perThreshold: 12, bonus: 2 },
  GOD_PROGRESSION: {
    baseRate: 0.03,
    ovrPenalty: 0.025,
    oldAgeMultiplier: 0.5,
    ageThreshold: 28,
    bonusRange: [5, 10]
  },

  // Distribution
  BASE_WEIGHTS: { physical: 0.25, shooting: 0.30, playmaking: 0.20, defense: 0.25 },
  AGED_WEIGHTS: { physical: 0.10, shooting: 0.35, playmaking: 0.25, defense: 0.30 },
  CATEGORY_SKILLS: {
    physical: ['spd', 'stre', 'jmp', 'endu'],
    shooting: ['fg', 'ft', 'tp', 'dnk'],
    playmaking: ['pss', 'drb', 'oiq'],
    defense: ['diq', 'ins', 'reb']
  },

  // Caps & Limits
  SKILL_CAPS: {
    '25-30': { min: -2, max: 4 },
    '31-34': { min: -3, max: 2 },
    '35+': { min: -5, max: 0 }
  },
  PHYSICAL_DECLINE_AGE: 30,
  PHYSICAL_SKILLS: ['spd', 'jmp', 'endu'],
  OVR_HARD_CAP: 80,

  // Validation
  MIN_AGE: 25,
  MIN_GAMES: 20,

  // Debug
  DEBUG: false,
  NOTIFY_GOD_PROGS: true,
  NOTIFY_ALL_PROGS: false
};
```

---

## Logic Flow (Pseudo-Code)

```javascript
async function progressPlayer(player) {
  // 1. Validate
  if (!player.watch || player.age < 25) return null;
  const stats = getPlayerStats(player, previousSeason);
  if (!stats || stats.gp < MIN_GAMES) return null;

  // 2. Defense-Adjusted PER
  const defensiveBonus = min(
    (stats.dws * 0.4) + (stats.stl * 0.3) + (stats.blk * 0.3),
    3.0
  );
  const adjustedPER = stats.per + defensiveBonus;

  // 3. Budget Calculation
  const ageMultiplier = getAgeMultiplier(player.age);
  let budget = floor((adjustedPER / 3.0) * ageMultiplier + uniform(-0.5, 0.5));

  // 4. Adjustments
  if (player.age <= 28 && adjustedPER < 12) budget += 2;
  if (random() < godThreshold(player.ovr, player.age)) budget += randomInt(5, 10);

  // 5. Distribution
  const weights = player.age >= 30 ? AGED_WEIGHTS : BASE_WEIGHTS;
  const categoryBudgets = distributeToCatgories(budget, weights);
  const skillChanges = distributeToSkills(categoryBudgets, player.age);

  // 6. Enforcement
  enforceSkillCaps(skillChanges, player.age);
  enforcePhysicalDecline(skillChanges, player.age);
  const projectedOVR = projectOVR(player.ratings, skillChanges);
  if (projectedOVR > 80) scaleChangesToOVRCap(skillChanges, projectedOVR);

  // 7. Apply
  applyChanges(player.ratings, skillChanges);
  return { player, budget, adjustedPER, skillChanges, finalOVR: calculateOVR(player.ratings) };
}
```

---

## Test Cases

| Case | Input | Expected |
|------|-------|----------|
| **Elite Scorer** | age:26, per:28, dws:0.5, stl:0.3, blk:0.1, ovr:75 | adjPER≈28.2, budget≈9, OVR≤80 |
| **Defensive Anchor** | age:28, per:12, dws:5, stl:2, blk:2, ovr:68 | adjPER≈15 (capped), budget≈4-5 (+low-PER bonus) |
| **Aging Veteran** | age:33, per:18, dws:2, stl:1, blk:0.5, ovr:76 | budget≈-2, physicals decline, OVR decreases |
| **OVR Cap** | age:27, per:30, dws:3, stl:1.5, blk:1, ovr:78 | budget≈10-12, scaled to OVR=80 exactly |
| **God Prog** | age:25, per:15, dws:1, stl:0.8, blk:0.5, ovr:65 | budget +5-10 if triggered, respects per-skill caps |

### Validation Checklist
- [ ] All test cases pass
- [ ] Zero OVR cap violations in 1000 player sim
- [ ] God progs at ~1-3% rate
- [ ] Defensive specialists progress fairly
- [ ] Physical decline visible in 30+ players

---

## Implementation Phases

| Phase | Goal | Deliverable | Success Criteria |
|-------|------|-------------|------------------|
| **1** | Foundation | CONFIG, utilities, validation, logging | Script runs, logs flagged players |
| **2** | Defense-Adjusted PER | `calculateDefensiveBonus()` | Elite defender: +3.0, poor: +0.5 |
| **3** | Budget Calculation | Budget values per player | 26yo/PER20: 6-7, 33yo/PER20: -2, god ~2% |
| **4** | Distribution | Skill-level changes | Budget 10 → ~0-1 per skill, sum conserved |
| **5** | Enforcement | Capped changes | No skill/OVR violations, regression preserved |
| **6** | Integration | Full script with notifications | Players update, no runtime errors |
| **7** | Testing & Tuning | Production v4.0 | All validation passes |

---

## Tuning Guide

| Goal | Adjustment |
|------|------------|
| ↑ Progression | Lower BASE_DIVISOR (3.0→2.5), raise age multipliers |
| ↑ Defense reward | Raise DWS weight (0.4→0.5), raise DEFENSIVE_BONUS_CAP |
| ↓ Physical decline | Raise 30+ multiplier (-0.3→-0.1) |
| ↓ God prog frequency | Lower baseRate (0.03→0.01), raise ovrPenalty |

---

## Decisions Made

| Question | Decision | Rationale |
|----------|----------|-----------|
| Position-specific weights? | Skip v4.0 | BBGM handles in OVR calc |
| God progs bypass caps? | No | Budget increase, but respect per-skill caps |
| Handle rookies (<25)? | Out of scope | Focus on fixing 26+ issues |
| Log all progs? | Toggle (default: god only) | Avoid notification spam |

---

## Progression Logic Diagrams

### Phase 1: Input Validation

```mermaid
flowchart TD
    A[Player + Stats] --> B{Watch flag?}
    B -->|No| Z[Skip - Return null]
    B -->|Yes| C{Age >= 25?}
    C -->|No| Z
    C -->|Yes| D{Games >= 20?}
    D -->|No| Z
    D -->|Yes| E[✅ Valid Player]
    E --> F[Proceed to Phase 2]

    style Z fill:#ff6b6b,color:#fff
    style E fill:#90EE90,color:#000
    style F fill:#74b9ff,color:#000
```

### Phase 2: Defense-Adjusted PER

```mermaid
flowchart TD
    A[Valid Player] --> B[Calculate Defensive Bonus]
    B --> C["bonus = DWS×0.4 + STL×0.3 + BLK×0.3"]
    C --> D{bonus > 3.0?}
    D -->|Yes| E[Cap at 3.0]
    D -->|No| F[Use calculated bonus]
    E --> G[adjPER = PER + cappedBonus]
    F --> G
    G --> H[Proceed to Phase 3]

    style G fill:#90EE90,color:#000
    style H fill:#74b9ff,color:#000
```

### Phase 3: Budget Calculation

```mermaid
flowchart TD
    A[adjPER] --> B[Get Age Multiplier]
    B --> C{Age Range?}
    C -->|25-27| D[mult = 1.0]
    C -->|28-30| E[mult = 0.5]
    C -->|31-34| F[mult = -0.3]
    C -->|35+| G[mult = -0.8]
    D --> H["rawBudget = adjPER/3.0 × mult"]
    E --> H
    F --> H
    G --> H
    H --> I["budget = floor + jitter ±0.5"]
    I --> J[Proceed to Phase 4]

    style H fill:#90EE90,color:#000
    style J fill:#74b9ff,color:#000
```

### Phase 4: Budget Adjustments

```mermaid
flowchart TD
    A[Base Budget] --> B{Age <= 28 AND<br/>adjPER < 12?}
    B -->|Yes| C[budget += 2<br/>🟢 Low-PER Bonus]
    B -->|No| D[No bonus]
    C --> E[Roll God Progression]
    D --> E
    E --> F["prob = 0.03 - OVR×0.025 × ageMult"]
    F --> G{random < prob?}
    G -->|Yes| H["budget += 5-10<br/>🌟 GOD PROG"]
    G -->|No| I[Normal progression]
    H --> J[Final Budget]
    I --> J
    J --> K[Proceed to Phase 5]

    style C fill:#90EE90,color:#000
    style H fill:#ffd700,color:#000
    style J fill:#90EE90,color:#000
    style K fill:#74b9ff,color:#000
```

### Phase 5: Budget Distribution

```mermaid
flowchart TD
    A[Final Budget] --> B{Age >= 30?}
    B -->|Yes| C[Use AGED_WEIGHTS<br/>Phys:10% Shoot:35%<br/>Play:25% Def:30%]
    B -->|No| D[Use BASE_WEIGHTS<br/>Phys:25% Shoot:30%<br/>Play:20% Def:25%]
    C --> E[Allocate to Categories]
    D --> E
    E --> F[Distribute to Skills<br/>+ jitter per skill]
    F --> G[14 skill changes calculated]
    G --> H[Proceed to Phase 6]

    style E fill:#90EE90,color:#000
    style G fill:#90EE90,color:#000
    style H fill:#74b9ff,color:#000
```

### Phase 6: Enforcement & Caps

```mermaid
flowchart TD
    A[Skill Changes] --> B[Apply Per-Skill Caps]
    B --> C["25-30: -2 to +4<br/>31-34: -3 to +2<br/>35+: -5 to 0"]
    C --> D{Age >= 30?}
    D -->|Yes| E[Physical Decline<br/>spd,jmp,endu cannot increase]
    D -->|No| F[Skip physical cap]
    E --> G[Project OVR]
    F --> G
    G --> H{projOVR > 80?}
    H -->|Yes| I[Scale positive changes<br/>🟠 Apply OVR cap scaling]
    H -->|No| J[No scaling needed]
    I --> K[Final Skill Changes]
    J --> K
    K --> L[Proceed to Phase 7]

    style E fill:#74b9ff,color:#000
    style I fill:#ff9f43,color:#000
    style K fill:#90EE90,color:#000
    style L fill:#74b9ff,color:#000
```

### Phase 7: Output & Application

```mermaid
flowchart TD
    A[Final Skill Changes] --> B[Apply Changes to Ratings]
    B --> C[Calculate Final OVR]
    C --> D[Return Result Object]
    D --> E["{ player, budget,<br/>adjPER, skillChanges,<br/>finalOVR }"]
    E --> F[✅ Complete]

    style D fill:#a29bfe,color:#000
    style E fill:#a29bfe,color:#000
    style F fill:#90EE90,color:#000
```

---

## Key Formulas Reference

```javascript
// Defense-Adjusted PER
adjPER = per + min((dws × 0.4) + (stl × 0.3) + (blk × 0.3), 3.0)

// Base Budget
budget = floor((adjPER / 3.0) × ageMultiplier + uniform(-0.5, 0.5))

// God Probability
godProb = (0.03 - (ovr / 100) × 0.025) × (age ≤ 28 ? 1.0 : 0.5)

// OVR Scaling
scaleFactor = 1 - ((projectedOVR - 80) / projectedOVR)
```


