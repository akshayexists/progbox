# Chart Reference

A technical reference for all 28 charts in the analysis dashboard, detailing the underlying mathematics and actionable tuning guidance.

---

## 1. Age Tier Outcome Profiles

### Mathematics

The kernel density estimate (KDE) for each age tier is computed using Gaussian kernels with Scott's rule for bandwidth selection:

$$\hat{f}(x) = \frac{1}{nh}\sum_{i=1}^{n} K\left(\frac{x - X_i}{h}\right), \quad h = n^{-1/5} \hat{\sigma}$$

where $K$ is the standard Gaussian kernel. The three tiers are formed by tertile splits on the `Age` column (`pd.qcut`). For each pair of tiers, a two-sample Kolmogorov-Smirnov test evaluates the maximum absolute difference in empirical CDFs:

$$D_{n_1, n_2} = \sup_x |\hat{F}_1(x) - \hat{F}_2(x)|$$

Cohen's d with Hedges' correction for small-sample bias is computed for pairwise comparisons:

$$d = \frac{\bar{X}_1 - \bar{X}_2}{s_{pooled}} \cdot \left(1 - \frac{3}{4(n_1+n_2)-9}\right), \quad s_{pooled} = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}$$

When a tier has zero variance (all deltas identical), a vertical reference line replaces the degenerate KDE. Dashed vertical lines mark each tier's mean.

### Tuning Application

This is the primary diagnostic for the age-progression model. It validates whether the simulation's aging coefficients produce distributionally distinct outcomes for different age groups.

- **Directional Integrity**: Verify that the Youngest tier's mean is positive and the Oldest tier's mean is negative. If the ordering is inverted, the aging curve coefficients in the engine have the wrong sign.
- **Tier Separation**: If the KS test yields $p > 0.05$ and Cohen's $d < 0.2$, age has no practical effect on outcomes. You must increase the magnitude of the age-based progression modifiers.
- **Distributional Shape**: Fat tails in the Youngest tier indicate high breakout potential; fat left tails in the Oldest tier indicate severe career-decline risk. Narrow, symmetric distributions imply the system is overly deterministic and lacks realistic variance.
- **Overlapping Distributions**: If the Middle tier completely overlaps with both Youngest and Oldest, consider whether a three-tier aging model is justified, or if a continuous age coefficient would be more efficient.

---

## 2. Multi-Driver Calibration

### Mathematics

The top 3 drivers are selected by $|\rho_{Spearman}|$ with `MeanDelta`. For each driver, a scatter plot is rendered with OLS trend lines. The annotated $r^2$ is the **partial** $r^2$, controlling for Age and other top drivers via residualization (Frisch-Waugh-Lovell theorem):

1. Regress driver $x_j$ on controls $Z = [\mathbf{1}, \text{Age}, x_{k \neq j}]$ to get residuals $e_{x_j}$
2. Regress $y$ (MeanDelta) on $Z$ to get residuals $e_y$
3. $r_{partial} = \text{Corr}(e_{x_j}, e_y)$, $\quad r^2_{partial} = r^2_{partial}$

This isolates the unique variance explained by each driver, providing clarity through the confounding Age and collinear inputs.

### Tuning Application

This chart isolates which input stats genuinely drive progression, independent of age. It is used to validate the design intent of the progression system.

- **Slope Direction**: A negative slope for a stat that should have a positive relationship with outcomes is a clear inversion requiring a coefficient sign flip.
- **Partial $r^2$ Magnitude**: If all partial $r^2$ values are $< 0.05$, the named drivers have no unique explanatory power. The system is either purely age-driven or dominated by unmeasured factors. Increase the weight of these inputs in the progression formula.
- **Tier-Specific Slopes**: If the Youngest tier shows a steep positive slope for a driver but the Oldest tier shows a flat slope, the driver acts only on development.
- **Action**: If a driver designed to be impactful shows near-zero partial $r^2$, increase its coefficient in the progression calculation or verify it is properly wired into the simulation loop.

---

## 3. Top Attribute Divergence

### Mathematics

The 6 attributes with the largest mean absolute delta (vs. baseline) are selected. For each attribute × age tier, 95% confidence intervals are computed via bootstrap resampling (2,000 iterations):

$$CI = \left[Q_{0.025}(\{\bar{\Delta}^*_b\}_{b=1}^{2000}), \quad Q_{0.975}(\{\bar{\Delta}^*_b\}_{b=1}^{2000})\right]$$

where $\bar{\Delta}^*_b$ is the mean of the $b$-th resample. Asymmetric error bars reflect the actual sampling distribution, avoiding Gaussian assumptions for potentially skewed attribute deltas.

### Tuning Application

This provides granular decomposition of the OVR Delta, showing exactly which attributes are moving and by how much in each age tier.

- **Magnitude Balance**: If athletic attributes (`Spd`, `Jmp`) move 5× more than skill attributes (`FT`, `oIQ`), the system heavily prioritizes athletic development. Adjust attribute-specific age coefficients to balance this ratio if skills are undervalued.
- **Direction in Oldest Tier**: Declining attributes in the Oldest tier must be physically plausible. `Spd` and `End` should decline; `oIQ` should remain stable or rise. If `oIQ` drops sharply, the aging model applies cognitive decline incorrectly.
- **CI Overlap**: Extensive CI overlap between tiers for a given attribute means age does not differentially affect that attribute. Adjust the attribute's age-sensitivity parameter.
- **Action**: If an attribute delta is universally near zero with tight CIs, its progression coefficient is effectively dead and needs to be increased to matter in the simulation.

---

## 4. Player Outcome Reliability

### Mathematics

Each player is plotted in $(\bar{\Delta}, \sigma_\Delta)$ space, colored continuously by Baseline OVR using a diverging RdYlBu_r colorscale. Quadrant boundaries are placed at the median $\bar{\Delta}$ and median $\sigma_\Delta$:

$$\bar{\Delta}_i = \frac{1}{R}\sum_{r=1}^{R} \Delta_{i,r}, \quad \sigma_{\Delta_i} = \sqrt{\frac{1}{R-1}\sum_{r=1}^{R} (\Delta_{i,r} - \bar{\Delta}_i)^2}$$

Quadrants are labeled: STAR (high $\bar{\Delta}$, low $\sigma$), VOLATILE (high $\bar{\Delta}$, high $\sigma$), STABLE (low $\bar{\Delta}$, low $\sigma$), DANGER (low $\bar{\Delta}$, high $\sigma$).

### Tuning Application

This maps the risk-return space of the progression system, identifying which player archetypes are reliable versus chaotic.

- **Baseline-Dependent Clustering**: If high-Baseline players (red) cluster in DANGER, the ceiling is too aggressive, making elite players both decline steeply and unpredictably. Soften the cap or reduce variance for high-Baseline players.
- **Vertical Spread**: Large vertical spread (varying $\sigma$) at similar horizontal positions ($\bar{\Delta}$) indicates unmodeled heterogeneity. Some players with the same mean outcome are far less predictable. Introduce a variance modifier tied to an input stat (e.g., "consistency" or "work ethic") to control this.
- **Quadrant Population**: A well-tuned system should have most players in STAR or STABLE. A DANGER-heavy system feels punishing and random. To fix this, either increase deterministic progression components or reduce global RNG variance.
- **Action**: If low-Baseline players (blue) are stuck in STABLE with $\bar{\Delta} \approx 0$, the floor is too rigid. Low-rated players should have higher mean deltas (room to grow).

---

## 5. Attribute Movement Heatmap

### Mathematics

A matrix $Z \in \mathbb{R}^{N \times M}$ of mean attribute deltas is rendered as a diverging heatmap:

$$z_{ij} = \bar{\Delta}_{attr_j}^{player_i} = \overline{Sim_{attr_j}^{(i)}} - Baseline_{attr_j}^{(i)}$$

Players are sorted by Baseline OVR (descending). Columns are sorted by mean $|z_{\cdot j}|$ (descending). Colorscale is symmetric, bounded at the 97th percentile of $|z_{ij}|$ to prevent outlier distortion. Rendering is capped at 80 rows for browser performance.

### Tuning Application

The comprehensive audit view, exposing every player-attribute combination simultaneously to spot global patterns.

- **Row Patterns**: A player with uniformly red cells is declining across the board. A player with mixed red/blue is undergoing skill conversion. If skill conversion is unintended, remove negative correlations in the attribute progression covariance matrix.
- **Column Uniformity**: An entirely monochromatic column indicates the attribute is driven purely by a global modifier (like a blanket aging penalty) rather than player-specific inputs. Add input-dependent progression to that attribute.
- **Diagonal Structure**: Red cells concentrated in the upper-left (high Baseline, high-movement attributes) and blue in the lower-right confirm the classic ceiling-compression × floor-support pattern.
- **Action**: Isolated deep-red or deep-blue cells (single player-attribute outliers) suggest a specific interaction bug in the engine for that archetype.

---

## 6. Tier Separation by Top Driver

### Mathematics

Players are divided into quartiles of the top driver **within each age tier** using rank-first quantile cutting to guarantee approximately equal group sizes:

$$Q_k = \text{qcut}(\text{rank\_first}(x_j | \text{AgeTier}), 4)$$

Box plots show the full distribution of OVR Delta for each quartile, with whiskers extending to $1.5 \times IQR$ and outliers beyond.

### Tuning Application

Tests whether the top driver creates practical tier separation after controlling for age.

- **Monotonicity**: Mean $\Delta$ should increase monotonically from Q1 to Q4. Non-monotonic patterns (e.g., Q2 > Q4) indicate a nonlinear or threshold-based relationship requiring a nonlinear transformation of the driver in the engine.
- **IQR Overlap**: If the interquartile ranges of Q1 and Q4 overlap substantially, the driver has weak practical effect despite possibly being statistically significant. Increase the driver's coefficient.
- **Outlier Density**: Heavy outliers in Q4 indicate that the driver enables extreme outcomes but doesn't guarantee them. If Q4 should reliably produce stars, reduce variance for high-driver-value players.
- **Action**: If quartile separation is only visible in the Youngest tier, the driver's effect is being nullified by aging mechanics. Ensure the driver interacts with the aging curve appropriately.

---

## 7. Simulation Force Field (Mean Pull vs. RNG Wobble)

### Mathematics

Baseline OVR is discretized into $q$ quantile bins ($q = \min(15, \max(5, N/8))$). For each bin $\times$ age tier, the "Mean Pull" (deterministic force) and "RNG Wobble" (stochastic uncertainty) are:

$$\mu_\Delta = \bar{\Delta}_{bin,tier}, \quad \sigma_\Delta = \text{Std}(\Delta)_{bin,tier}$$

The wobble band is $\mu \pm \sigma$. A system-level signal-to-noise ratio (SNR) is computed as the root-mean-square of weighted signals divided by root-mean-square of weighted noises:

$$SNR = \frac{\sqrt{\sum_k w_k \mu_k^2}}{\sqrt{\sum_k w_k \sigma_k^2}}, \quad w_k = n_k$$

SNR > 1 indicates the deterministic trend dominates; SNR < 1 indicates noise dominates.

### Tuning Application

Reframes the Monte Carlo as a physical system, showing exactly where the tuner's settings have force versus where RNG overwhelms intent.

- **Zero-Force Crossover**: The Baseline OVR where $\mu_\Delta = 0$ is the system's equilibrium. Below this, the system pulls players up; above it, the system pulls them down. Verify this crossover occurs at the intended OVR target.
- **Wobble Dominance**: Bins where the $\pm \sigma$ band crosses zero (or spans both positive and negative territory) are regions where the outcome is indistinguishable from a coin flip. To impose control, reduce the per-step variance parameter for players in that Baseline range.
- **Tier Divergence**: The Oldest tier's force line should be below the Youngest's. If they converge, age effects are too weak.
- **Action**: If SNR < 1.0, the system is globally dominated by RNG. Scale up deterministic progression coefficients by a factor of $1/SNR$ to achieve parity between signal and noise.

---

## 8. Convergence Diagnostics

### Mathematics

For the 6 most volatile players, running statistics are computed incrementally:

$$\bar{\Delta}_k = \frac{1}{k}\sum_{i=1}^{k} \Delta_i, \quad s_k^2 = \frac{k}{k-1}\left(\frac{1}{k}\sum_{i=1}^{k} \Delta_i^2 - \bar{\Delta}_k^2\right), \quad MCSE_k = \frac{s_k}{\sqrt{k}}$$

Global convergence is defined as the fraction of players where $MCSE < \theta_{threshold}$ (default 0.5 OVR pts):

$$\text{Pct Converged} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\left[\frac{s_i}{\sqrt{n_i}} < \theta\right]$$

### Tuning Application

This chart validates the statistical reliability of the entire dashboard. It must be checked before interpreting any other chart.

- **Band Narrowing**: The $\pm MCSE$ ribbon must collapse to a narrow band (relative to the mean) by the right edge. If it remains wide, the player's true mean is unknown, and more simulation runs are required.
- **Drift**: A running mean that oscillates or trends persistently indicates the sampling distribution is not yet stationary. Increase the run count.
- **Global Convergence Target**: < 80% convergence means the dashboard statistics are unreliable. ≥ 95% is the minimum target for actionable tuning.
- **Action**: If specific archetypes fail to converge while others do, those archetypes have higher variance. Targeted variance reduction for those archetypes (via input-specific noise parameters) will improve convergence without needing exponentially more runs.

---

## 9. Variance Decomposition (ICC)

### Mathematics

For each varying attribute, the moment-based Intraclass Correlation Coefficient is:

$$ICC = \frac{\sigma^2_{between}}{\sigma^2_{between} + \sigma^2_{within}}, \quad \sigma^2_{between} = \text{Var}(\bar{X}_{player}), \quad \sigma^2_{within} = \overline{\text{Var}(X_{player})}$$

ICC near 1.0 indicates outcomes determined by player identity (deterministic); ICC near 0.0 indicates outcomes determined by simulation noise (RNG). Bars are color-coded: green (ICC ≥ 0.7), orange (0.4 ≤ ICC < 0.7), red (ICC < 0.4).

### Tuning Application

The RNG audit. Identifies which attributes the tuner can reliably control versus which are at the mercy of randomness.

- **High-ICC Attributes**: Tuning these is efficient, changes to their progression coefficients will reliably produce the intended effect.
- **Low-ICC Attributes**: These are dominated by noise. Adjusting their deterministic coefficients is pointless until their variance parameter is reduced. Focus tuning efforts on reducing the per-step variance for these attributes.
- **Systemic Pattern**: If all attributes are red (ICC < 0.4), the entire system is 60%+ random. Reduce the global RNG multiplier. If physical attributes have high ICC but skill attributes have low ICC, the skill variance parameters are disproportionately large.
- **Action**: For any attribute with ICC < 0.3 that should be deterministic (e.g., `oIQ`), decrease the variance parameter by a factor of at least 2.

---

## 10. Multivariate True Leverage (OLS β ± 95% CI)

### Mathematics

For each age tier, a standardized OLS regression is fit:

$$z(y) = \beta_0 + \sum_{j=1}^{p} \beta_j z(x_j) + \epsilon$$

Coefficients are estimated via QR decomposition: $Q, R = \text{QR}(X)$, $\hat{\beta} = R^{-1}Q^Ty$. Standard errors derive from $\text{Var}(\hat{\beta}) = \hat{\sigma}^2 (R^{-1})(R^{-1})^T$. 95% CIs are $\hat{\beta}_j \pm 1.96 \cdot SE_j$. Variance Inflation Factors detect collinearity:

$$VIF_j = \frac{1}{1 - R^2_{j|(-j)}}$$

where $R^2_{j|(-j)}$ is from regressing $x_j$ on all other predictors. VIF > 10 is flagged.

### Tuning Application

Reveals the independent, simultaneous impact of each input stat on MeanDelta, controlling for confounding.

- **CI Inclusion of Zero**: If a 95% CI includes zero, that input has no statistically significant independent effect. It is either irrelevant or completely confounded by other inputs.
- **VIF Flags**: VIF > 10 means the coefficient is numerically unstable due to collinearity. Do not tune based on unstable coefficients. Remove one of the collinear inputs from the model or redesign the inputs to be independent.
- **Tier Comparison**: A $\beta$ that is large and positive for Youngest but near-zero for Oldest means the input only drives development, not decline. Use this to validate that development and aging are correctly decoupled in the engine.
- **Action**: If an input designed to be a primary driver shows $\beta \approx 0$ with a tight CI, it is functionally dead in the simulation. Increase its weight in the progression formula or verify its data pipeline.

---

## 11. Volatility Landscape

### Mathematics

For each age tier and candidate input stat, the Spearman rank correlation with `StdDelta` is computed. Spearman's $\rho$ is defined as the Pearson correlation applied to rank-transformed data, which naturally handles monotonic nonlinearities and outliers:

$$\rho_S(X, Y) = \rho_P(\text{rank}(X), \text{rank}(Y))$$

### Tuning Application

Identifies which input stats make a player's outcome unpredictable (volatility drivers).

- **Baseline vs. StdDelta**: A negative $\rho_S$ means high-Baseline players have lower variance (ceiling compression). A positive $\rho_S$ means high-Baseline players have higher variance (elite chaos). Most systems should show negative $\rho_S$.
- **Age vs. StdDelta**: A positive $\rho_S$ indicates older players have more uncertain outcomes (unpredictable decline). This is realistic but should be controlled. If too high, aging becomes a lottery.
- **Actionable Levers**: Any input with $|\rho_S| > 0.3$ regarding `StdDelta` is a volatility lever. To make outcomes more predictable for players with high values of that input, add an input-dependent variance reducer to the engine (e.g., `variance_modifier = 1 / (1 + \alpha \cdot \text{input})`).

---

## 12. Attribute Delta Co-movement

### Mathematics

A Spearman correlation matrix of all varying attribute deltas:

$$C_{ij} = \rho_S(\Delta_{attr_i}, \Delta_{attr_j})$$

Rendered as a diverging heatmap bounded $[-1, +1]$. Positive values indicate co-movement; negative values indicate tradeoffs; near-zero indicates independence.

### Tuning Application

Exposes hidden structure in the progression model. Unwanted correlations imply the system forces archetypes; wanted correlations imply realistic physical coupling.

- **Positive Clusters**: A block of high positive correlations (e.g., `Spd`, `Jmp`, `End` all at $\rho > 0.6$) indicates an "athleticism factor", qhere these attributes rise and fall together. If intentional, this is a latent factor. If not, decouple their RNG seeds or age curves.
- **Negative Correlations (Tradeoffs)**: If `3Pt` $\uparrow$ implies `Reb` $\downarrow$, the system forces a shooter/rebounder specialization. Decide if this is desired game design. If not, remove the negative covariance from the attribute progression matrix.
- **Independence**: A mostly white heatmap means attributes evolve independently. This provides maximum tuning flexibility.
- **Action**: If the heatmap shows strong off-diagonal structure that contradicts design intent, adjust the covariance matrix in the attribute progression engine to zero out unwanted couplings.

---

## 13. Per-Attribute Progression by Age Tier

### Mathematics

All varying attributes are displayed in a faceted grid. For each attribute $\times$ age tier, the mean delta and 95% CI are:

$$\bar{\Delta}_{attr,tier} \pm 1.96 \cdot SE_{attr,tier}, \quad SE = \frac{s}{\sqrt{n}}$$

relying on the Central Limit Theorem for the normality of the sample mean.

### Tuning Application

The complete attribute-level blueprint of the progression system, showing every varying attribute's age-gradient.

- **Age-Gradient Magnitude**: The difference between Youngest and Oldest means dictates how steep the age curve is. If gradients are too shallow, attribute-specific age coefficients need amplification.
- **CI Non-Overlap**: Non-overlapping CIs between Youngest and Oldest confirm a statistically significant age effect. Overlapping CIs mean age doesn't affect that attribute.
- **Physical Plausibility**: `Spd` and `End` should decline with age; `oIQ` should not. If the bars show the reverse, the age-coefficient signs are wrong.
- **Action**: Adjust attribute-specific age multipliers. If `FT` (free throw) is declining with age but should be age-stable, set its age coefficient to zero or positive.

---

## 14. Cap Ceiling & Probability of Improvement

### Mathematics

**Left Axis (Compression Curve)**: Players binned by Baseline OVR into 15 quantiles. Mean simulated OVR is plotted against baseline midpoint. The $y = x$ line represents "no change."

**Right Axis (Logistic Survival Curve)**: Empirical $P(\Delta > 0)$ per bin, fit to a logistic function per age tier via nonlinear least squares (`scipy.optimize.curve_fit`):

$$P(\Delta > 0 | \text{Baseline}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 \cdot \text{Baseline})}}$$

### Tuning Application

Directly measures the OVR cap's existence and severity, and how it affects the probability of improvement.

- **Compression Distance**: The vertical gap between $y = x$ and the mean OVR curve at high baselines quantifies cap severity. A 5+ OVR gap means the cap is heavily compressing elite players.
- **Cap Estimate**: The red dashed line at $\max(\overline{SimOVR}) + 1$. If this is 71 but the design target is 80, the cap parameter must be raised.
- **Logistic Crossover**: The Baseline where $P(\Delta > 0) = 0.5$ is the tipping point. This should be lower for Youngest and higher for Oldest. If they are identical, age doesn't affect improvement probability.
- **Action**: If the logistic curve drops too steeply (cliff edge), the cap is a hard cutoff. Replace it with a soft asymptotic cap (e.g., $\Delta \propto 1 - \frac{OVR}{Cap}$) for smoother degradation.

---

## 15. Age-Adjusted Over/Under Performers

### Mathematics

Each player's age-adjusted Z-score normalizes their mean delta against their age tier:

$$Z_i = \frac{\bar{\Delta}_i - \mu_{tier}}{\sigma_{tier}}$$

The histogram is normalized to probability density, overlaid with a standard normal $N(0,1)$ PDF. The comparison reveals whether the residual distribution matches the expected normal or exhibits excess kurtosis/skewness.

### Tuning Application

Identifies players who perform unexpectedly well or poorly after removing the expected age effect.

- **Distribution Match**: If the Z-score histogram fits $N(0,1)$, the system has no consistent over/underperformers and we can conclude that the deviation is random. Fat tails indicate some players systematically beat or miss expectations.
- **Asymmetry**: Right-skew means more overperformers; left-skew means more underperformers. Either indicates a missing variable that systematically helps or hurts a subset of players.
- **Extreme Z-scores ($|Z| > 2$)**: These players are outliers. Investigate their attribute profiles to find the hidden driver. Often, a specific stat combination exploits an edge case in the progression formula.
- **Action**: If the distribution is consistently skewed across multiple simulation runs, add the missing explanatory variable to the model. If it's random across runs, the skew is sampling noise.

---

## 16. Tail Risk Analysis

### Mathematics

Empirical tail probabilities for thresholds $\theta \in \{2, 5, 10\}$:

$$P(\Delta > \theta | tier) = \frac{1}{n_{tier}} \sum_{i=1}^{n_{tier}} \mathbb{1}[\Delta_i > \theta], \quad P(\Delta < -\theta | tier) = \frac{1}{n_{tier}} \sum_{i=1}^{n_{tier}} \mathbb{1}[\Delta_i < -\theta]$$

Gain probabilities are plotted as solid bars (+Y), loss probabilities as faded bars (−Y).

### Tuning Application

Quantifies extreme outcome risk, which are the events users notice most. This is the system's "feeling" check.

- **Gain/Loss Asymmetry**: If $P(\Delta > +5) \gg P(\Delta < -5)$ for Youngest, young players have huge upside with limited downside (a lottery ticket). Adjust if the design requires young players to also carry bust risk.
- **Threshold Decay**: $P(\Delta > +10)$ should be < 5%. If it's 15%+, extreme outcomes are too common. Reduce the per-step variance or the number of simulation steps.
- **Catastrophic Decline**: If $P(\Delta < -10) > 15\%$ for Oldest, career-ending declines are too frequent. Cap the maximum per-step decline for older players.
- **Action**: Tune the variance parameter $\sigma^2$ proportional to the excess tail probability. If $P(>10)$ is 3× the expected Gaussian tail, reduce $\sigma$ by $\sqrt{3}$.

---

## 17. Interaction Surface

### Mathematics

For each age tier, a bivariate OLS plane is fit:

$$\hat{\Delta} = \beta_0 + \beta_1 x_1 + \beta_2 x_2$$

Predicted on a 25×25 mesh grid. Actual data points are overlaid as a 3D scatter. Note: This is an additive plane; if an interaction term $x_1 \cdot x_2$ exists in the true model, the plane will average across it, showing a tilted but flat surface.

### Tuning Application

Visualizes how two inputs jointly determine outcomes, making their additive relationship intuitive.

- **Plane Tilt Direction**: If the plane slopes upward along $x_1$ but is flat along $x_2$, $x_1$ is the active driver and $x_2$ is irrelevant. Remove $x_2$ from the progression formula to simplify.
- **Tier Inversion**: If the Youngest plane tilts upward but the Oldest tilts downward for the same driver, that driver promotes development but accelerates decline, a realistic "high-risk, high-reward" archetype.
- **Data-Cloud Alignment**: If points scatter $\pm 5$ OVR from the plane, the two drivers explain little variance. If they hug the plane, the model is a good fit.
- **Action**: If the plane is flat for all tiers, the top 2 drivers are not actually drivers. Re-evaluate the driver selection or increase their coefficients.

---

## 18. Effect Size Forest Plot

### Mathematics

Pairwise Cohen's d (with Hedges' correction) between age tiers:

$$d = \frac{\bar{X}_1 - \bar{X}_2}{s_{pooled}} \cdot \left(1 - \frac{3}{4(n_1+n_2)-9}\right)$$

Supplemented by Mann-Whitney U (non-parametric location) and Kolmogorov-Smirnov (distributional shape) tests. 95% CIs derived from bootstrap resampling of the mean difference. Reference lines at $d = 0.2$ (small), $0.5$ (medium), $0.8$ (large).

### Tuning Application

The definitive answer on whether age tiers produce meaningfully different outcomes, accounting for both statistical significance and practical magnitude.

- **Magnitude Assessment**: $|d| > 0.8$ is a large, important effect. $|d| < 0.5$ is practically negligible even if statistically significant. Tune the system to achieve $d > 0.8$ between Youngest and Oldest.
- **Asymmetric Gaps**: If Youngest vs. Oldest $d = 1.2$ but Middle vs. Oldest $d = 0.3$, the Middle tier is functionally similar to Oldest. Consider a two-tier (Young/Old) model instead of three.
- **CI Width**: Wide CIs indicate insufficient sample size or high within-tier variance. Narrow CIs with small $d$ definitively prove the tiers are practically equivalent.
- **Action**: If pairwise $d < 0.5$ across the board, the age-based progression modifiers are too weak. Scale them up proportionally to the target $d$.

---

## 19. Risk-Adjusted Efficiency

### Mathematics

The Certainty Equivalent (CARA utility, $A=1$) penalizes variance quadratically:

$$CE_i = \mu_i - \frac{1}{2}\sigma_i^2$$

The Sharpe ratio measures risk-adjusted return:

$$S_i = \frac{\mu_i}{\sigma_i}$$

A rolling-window Pareto frontier tracks the maximum CE achieved at each Baseline level. Players above the frontier are optimal; below are suboptimal.

### Tuning Application

Evaluates the progression system as an investment portfolio, allowing you to identify which players offer the best risk-adjusted returns.

- **Frontier Shape**: A rising frontier (higher Baseline $\rightarrow$ higher CE) means elite players have the best risk-adjusted returns. A declining frontier means the ceiling crushes elite-player efficiency.
- **Distance from Frontier**: Players far below the frontier are inefficient, meaning their variance is too high for their mean. Identify these players' common traits and reduce their variance modifiers if you want to balance it out.
- **Sharpe Threshold**: Sharpe > 1.0 means the signal exceeds the noise (reliable improvement). Sharpe < 0 implies reliable decline. A system with most players near Sharpe = 0 is dominated by RNG.
- **Action**: To improve system efficiency, reduce $\sigma$ for players with high $\mu$ but low CE (dragged down by variance).

---

## 20. Rank Stability (Kendall's W)

### Mathematics

The OVR rank matrix $R \in \mathbb{R}^{n \times m}$ (n players $\times$ m runs) is used to compute Kendall's W coefficient of concordance:

$$W = \frac{12S}{m^2(n^3 - n)}, \quad S = \sum_{i=1}^{n} (R_{i\cdot} - \bar{R})^2$$

where $R_{i\cdot} = \sum_j R_{ij}$. $W \in [0, 1]$. Per-player rank coefficient of variation: $RankCV_i = \sigma_{rank_i} / \bar{rank}_i$.

### Tuning Application

Measures whether the simulation's leaderboard is stable or if RNG reshuffles rankings every run.

- **W > 0.7**: Stable hierarchy. The same players consistently rank highest. The deterministic backbone is strong.
- **W < 0.4**: Near-random rankings. The "best" player in one run might be 20th in the next. The system lacks deterministic differentiation. Increase the spread of progression rates between players.
- **Rank CV Distribution**: A right-skewed CV distribution means most players are stable but a few are volatile. Investigate the volatile ones as the likeliest explanation is that their ranks are being determined by RNG, not attributes.
- **Action**: If $W$ is low, increase the variance of deterministic progression rates (e.g., widen the distribution of ratings) or reduce the per-run RNG multiplier.

---

## 21. Outlier Detection (Mahalanobis)

### Mathematics

Mahalanobis distance in $(\bar{\Delta}, \sigma_\Delta)$ space:

$$D_i = \sqrt{(\mathbf{x}_i - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}_i - \boldsymbol{\mu})}$$

Under multivariate normality, $D_i^2 \sim \chi^2(df=2)$. P-values: $p_i = 1 - F_{\chi^2(2)}(D_i^2)$. Players with $p < 0.05$ are flagged. The 95% and 99% $\chi^2$ contours are drawn as ellipses scaled by axis standard deviations.

### Tuning Application

Identifies structurally anomalous players whose risk-return profile violates the system's normal behavior.

- **Outlier Direction**: A high-$\bar{\Delta}$, low-$\sigma$ outlier is a "guaranteed star" (the system always boosts them), while a low-$\bar{\Delta}$, high-$\sigma$ outlier is a "coin flip" (the system can't decide what to do with them).
- **Clustered Outliers**: Multiple outliers sharing a trait implies a systematic edge case, not random noise. That trait is exploiting a gap in the progression logic.
- **Zero Outliers**: The variance structure is uniform. No archetype produces anomalous risk-return profiles. This is ideal.
- **Action**: For guaranteed-star outliers, verify their attribute combination isn't bypassing the cap. For coin-flip outliers, add a deterministic anchor (e.g., a minimum progression floor) to reduce their variance.

---

## 22. Funnel Plot (Heteroscedasticity)

### Mathematics

Scatter plot of $(Baseline, \sigma_\Delta)$ with a rolling-window average of $\sigma_\Delta$. A Spearman correlation test assesses heteroscedasticity:

$$\rho_S(Baseline, \sigma_\Delta)$$

$\rho < -0.3, p < 0.05$: Ceiling compression. $\rho > 0.3, p < 0.05$: Expanding variance. Otherwise: homoscedastic.

### Tuning Application

Directly visualizes ceiling compression.

- **Funnel Shape**: A narrowing funnel (high baseline $\rightarrow$ low $\sigma$) confirms ceiling compression. Elite players are locked in place. If the design intends for elite players to have some upside, the cap must be softened.
- **Inverse Funnel**: Expanding variance at high baselines means elite players have the most chaotic outcomes. This creates a frustrating user experience. Cap the variance parameter for high-Baseline players.
- **Color Gradient**: If high-Baseline, low-$\sigma$ points are also red (negative $\bar{\Delta}$), these players are in "certain decline".
- **Action**: If ceiling compression is detected ($\rho < -0.3$), change the cap formula from a hard linear subtractor to a soft logistic asymptote, which preserves some variance near the ceiling.

---

## 23. Conditional Probability Map

### Mathematics

The dataset is binned by Baseline OVR and Age. Within each cell, the empirical probability of improvement is:

$$P(\Delta > 0 | Baseline \in B_j, Age \in A_k) = \frac{\sum_{i \in B_j \cap A_k} \mathbb{1}[\Delta_i > 0]}{n_{jk}}$$

Rendered as a 2D heatmap with a RdYlGn colorscale [0, 1].

### Tuning Application

The most granular view of who improves and who doesn't, jointly conditioned on age and baseline.

- **Gradient Direction**: The gradient should run from upper-left (young, low baseline $\rightarrow$ green) to lower-right (old, high baseline $\rightarrow$ red). Any reversal is a bug in the conditional probability structure.
- **Steepness**: A sharp boundary (adjacent cells jumping from 80% green to 20% red) means the system has a cliff edge. Smooth the transition by making progression probabilities a continuous function of inputs.
- **Green Zone Size**: The "improvement zone" ($P > 0.7$) should be large enough to include most young players. If it's tiny, the system is too pessimistic.
- **Action**: If the probability map is nearly uniform at 50%, the deterministic components are too weak to shift the odds. Increase the magnitude of age and baseline progression modifiers.

---

## 24. Full Input Driver Ranking

### Mathematics

For every input stat, partial correlation with `MeanDelta` controls for Age and Baseline via residualization (Frisch-Waugh-Lovell):

$$r_{partial}(x_j, y | Z) = \text{Corr}(e_{x_j}, e_y)$$

The Benjamini-Hochberg procedure controls false discovery rate at $\alpha = 0.05$:

$$\text{Reject } H_0 \text{ for } p_{(k)} \text{ if } p_{(k)} \leq \frac{\alpha \cdot k}{m}$$

Significant drivers are colored green; non-significant are gray.

### Tuning Application

Comprehensive audit of every input stat's unique contribution, ranking them by effect strength.

- **Significance Filtering**: Gray bars (BH-adjusted non-significant) have no reliable unique effect. They are either irrelevant or completely confounded by Age/Baseline. Do not waste tuning effort on them.
- **Confounding Detection**: A stat with high simple correlation but low partial correlation is confounded, its apparent effect is actually due to Age or Baseline. Remove it from the progression formula to avoid multicollinearity.
- **Negative Partial Correlations**: These indicate that, all else equal, higher values predict worse progression. This might be a suppression effect or a genuine bug (e.g., a "potential" stat that hurts development).
- **Action**: Prune non-significant inputs from the progression formula to reduce noise and improve numerical stability. Focus tuning on the top 3–5 significant drivers.

---

## 25. Partial Dependence (Nonlinearity)

### Mathematics

Each top-8 input stat is binned into quantile groups. The partial dependence approximation is:

$$PD(x_j \in B_k) \approx \frac{1}{n_k}\sum_{i: x_{ij} \in B_k} \bar{\Delta}_i \pm 1.96 \cdot SE$$

Tier-specific curves use fewer bins due to smaller samples. This approximates $E_{x_{-j}}[f(x_j, x_{-j})]$, revealing nonlinearities invisible to linear correlation.

### Tuning Application

Reveals thresholds, saturation, and nonlinear effects that correlation coefficients cannot capture.

- **Linearity Check**: If all curves are straight lines, the system is linear, and Chart 24 tells the full story. Bends require nonlinear tuning.
- **Inflection Points**: A sharp bend marks a threshold. Example: if PER has no effect below 60 but a strong positive effect above 60, the system has a "PER threshold." 
- **Saturation**: A flattening curve at high values indicates diminishing returns. If the design calls for continued returns, remove the saturation cap.
- **Action**: If tier-specific curves diverge (e.g., positive slope for Youngest, flat for Oldest), implement an age-dependent coefficient for that input. If they are parallel, a single global coefficient suffices.

---

## 26. Input→Output Sensitivity Matrix

### Mathematics

For each input stat $\times$ output metric pair, Spearman correlation:

$$\rho_S(x_j, y_k), \quad y_k \in \{\text{MeanDelta}, \text{StdDelta}, \text{PctPositive}, P_{95}, P_{05}, \text{IQR}\}$$

Rendered as a heatmap with text annotations, sorted by $|\rho_S(x_j, \text{MeanDelta})|$.

### Tuning Application

Reveals that different inputs drive different aspects of the outcome distribution, a single input can affect the mean, the variance, or the tails independently.

- **Mean vs. StdDelta Divergence**: An input with high $|\rho|$ for MeanDelta but low $|\rho|$ for StdDelta is a clean signal lever, it changes the outcome level without affecting uncertainty. An input with high $|\rho|$ for StdDelta only is a pure volatility lever.
- **PctPositive vs. P95**: An input correlating with PctPositive but not P95 affects the probability of modest improvement without amplifying extreme upside. An input correlating with P95 but not PctPositive is a "boom" driver.
- **Action**: To reduce system chaos without affecting average outcomes, target inputs with high StdDelta/IQR correlations and add variance-reduction modifiers tied to those inputs.

---

## 27. Incremental R² Waterfall

### Mathematics

Forward stepwise OLS selection adds the feature maximizing $R^2$ gain at each step:

$$\Delta R^2_k = R^2_k - R^2_{k-1}$$

Process terminates when $\Delta R^2 < 0.001$. All features are z-scored. Cumulative $R^2$ is plotted on the right axis.

### Tuning Application

Answers the fundamental question: how much does each input actually improve prediction, and how much of the system is pure RNG?

- **First Entry**: The feature entered first has the highest simple predictive power (usually Age or Baseline).
- **Diminishing Returns**: If the 2nd feature adds < 0.02 to $R^2$, the system is effectively one-dimensional. To create a multi-dimensional system, increase the coefficients of secondary inputs.
- **Total $R^2$**: The cumulative $R^2$ at the last step is the explainable variance. $1 - R^2_{total}$ is pure RNG. If total $R^2 = 0.30$, 70% of the outcome is noise.
- **Action**: If total $R^2$ is unacceptably low, the tuner must either add missing features to the model or reduce the global variance parameter to increase the signal-to-noise ratio.

---

## 28. Pairwise Interaction Strength

### Mathematics

For each pair of top-8 input stats, a regression with an interaction term is fit on standardized variables:

$$z(y) = \beta_0 + \beta_1 z(x_1) + \beta_2 z(x_2) + \beta_3 z(x_1) \cdot z(x_2) + \epsilon$$

The heatmap displays the standardized interaction coefficient $\beta_3$. $\beta_3 = 0$ implies additivity; $\beta_3 > 0$ implies synergy; $\beta_3 < 0$ implies antagonism.

### Tuning Application

Exposes synergies and antagonisms between input stats. Inputs with zero interaction are independent levers; inputs with strong interaction must be tuned together.

- **Positive Interaction (Synergy)**: $\beta_3 > 0$ means the combined effect exceeds the sum of individual effects. 
- **Negative Interaction (Antagonism)**: $\beta_3 < 0$ means the combined effect is less than the sum. 
- **Action**: If the heatmap is entirely near-zero, the system is purely additive. This is easy to tune but lacks strategic depth. To add depth, introduce explicit interaction terms ($x_1 \cdot x_2$) in the progression formula for complementary stats.