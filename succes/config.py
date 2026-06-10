"""
succes/config.py
----------------
Global constants and tunable hyperparameters.

All scenario, GA, and engine behaviour can be controlled here.
Edit freely — this is the first place to look when calibrating.

TO-DO (future work):
  - CO2 shadow price as policy sweep variable
  - Seasonal calibration against ENTSO-E Transparency data
  - Sensitivity analysis runner (systematic parameter sweeps)
  - Per-run scenario versioning and checksums
  - Validation module comparing against reference market metrics
  - Storage charge/discharge as explicit GA optimisation variables
    (currently handled in post-GA dispatch only)
  - Open-source prep: README, pyproject.toml, pytest smoke tests
  - RES curtailment below price threshold (post-dispatch price-feedback loop):
      Option A (cheap): per-scenario demand floor in _compute_res_net_demand
        — 5 lines in julia_bridge.py, implicit curtailment in low-demand scenarios.
      Option B (correct): iterative post-dispatch adjustment in engine.jl
        — after preliminary MCP, reduce RES output for units with
          curtailment_threshold > mcp, re-run MCP step once.
        Requires passing res_avail + thresholds as arrays into Julia payload.
"""

# ── Time ──────────────────────────────────────────────────────────────────────
WINDOW_HOURS: int = 24          # hours per rolling optimisation window
HOURS_PER_YEAR: int = 8760

# ── Penalty prices (€/MWh) ────────────────────────────────────────────────────
MUST_RUN_PENALTY:    float = 200.0    # excess generation from must-run units
SCARCITY_PENALTY:    float = 3_000.0  # unserved demand (value of lost load)
CONGESTION_PENALTY:  float = 1_000.0  # overloaded transmission link
CURTAILMENT_PENALTY: float = 200.0    # negative residual demand, cannot absorb

# ── Risk objective ────────────────────────────────────────────────────────────
# Objective = E[cost] + LAMBDA_RISK × CVaR_alpha
# Lower lambda → more risk-neutral → fewer over-commitments but more tail risk.
# Higher lambda → more risk-averse → safer but more always-on behaviour.
# Recommended range: 0.10–0.30 for balanced UC with realistic commitment fractions.
LAMBDA_RISK: float  = 0.15     # default; overridden in scenario files
CVAR_ALPHA:  float  = 0.05     # tail fraction for CVaR (5% worst scenarios)

# ── Solver defaults ───────────────────────────────────────────────────────────
N_SCENARIOS:   int   = 30       # scenarios per window evaluation
GA_EPOCHS:     int   = 400
GA_POP_SIZE:   int   = 80
GA_PC:         float = 0.85     # crossover probability
GA_PM:         float = 0.005    # bit-flip mutation rate
RANDOM_SEED:   int   = 42

# ── GA convergence hyperparameters ────────────────────────────────────────────
# These can be turned on/off independently.

# Warm-start: seed window N's population with best solution from window N-1.
# Dramatically reduces epochs needed to converge (demand changes ~5-10% day-to-day).
# Effect: faster convergence, lower penalty on early windows.
WARM_START_ENABLED: bool  = True
WARM_START_FRACTION: float = 0.25   # fraction of population seeded from previous window

# Diversity restart: when GA stalls for STALL_PATIENCE epochs, replace the
# worst DIVERSITY_REPLACE_FRAC of the population with random mutations of best.
# Effect: escapes local optima, extends productive search.
DIVERSITY_RESTART_ENABLED: bool  = True
STALL_PATIENCE:            int   = 30    # epochs without improvement before restart
DIVERSITY_REPLACE_FRAC:    float = 0.20  # fraction of population replaced on restart

# Adaptive epoch budget: detect early convergence and stop the GA.
# Saved compute can be logged (future: redistributed to harder windows).
# Effect: 2-5x faster on easy windows (weekends, low-demand).
ADAPTIVE_EPOCHS_ENABLED:   bool  = True
ADAPTIVE_MIN_IMPROVEMENT:  float = 5e-4  # relative improvement threshold to continue
                                          # 5e-4 = 0.05% over ADAPTIVE_PATIENCE epochs
                                          # 1e-4 was too tight — GA still improves at ~4e-4/80ep
ADAPTIVE_PATIENCE:         int   = 80    # stop if no improvement for this many epochs

# ── Parallelism ───────────────────────────────────────────────────────────────
PARALLEL: bool = False
N_WORKERS: int = 4

# ── Market coupling ───────────────────────────────────────────────────────────
FLOW_PASSES: int = 5

# ── Carbon pricing ────────────────────────────────────────────────────────────
# EU ETS CO2 price in EUR/tCO2.
# TODO: expose as policy sweep variable for sensitivity analysis.
CO2_PRICE: float = 82.0

# ── Rolling horizon overlap ───────────────────────────────────────────────────
# The GA optimises over WINDOW_HOURS + OVERLAP_HOURS total.
# Only the first WINDOW_HOURS are committed ("binding horizon").
# The overlap gives the GA foresight beyond the current window boundary.
#
# Why 24h (was 12h):
#   DE lignite has min_run_hours=16–18 and startup costs 40–80k EUR (recalibrated).
#   In a 24h binding window with an overnight valley (demand ~30% of peak from 01:00–06:00),
#   committing lignite at hour 18 meant paying startup to run through 6 unprofitable hours.
#   With 24h overlap (48h total planning horizon), the GA sees the next-day morning ramp
#   and correctly values the lignite commitment: startup cost is amortised over 30+ hours.
#   This is expected to raise lignite CF from ~70% toward the real ~80%, closing the DE–FR gap.
OVERLAP_HOURS: int = 24

# ── Seed evaluation ───────────────────────────────────────────────────────────
MAX_PAIR_UNITS: int = 20

# ── Stochastic scenario options ───────────────────────────────────────────────
# DETERMINISTIC MODE: set True to run a single base scenario with no noise.
# All stochastic sources are disabled: N_SCENARIOS is forced to 1, all noise
# std values are zeroed, FOR and hydro stochasticity are disabled.
# Use this to validate price formation and check for structural scarcity
# before introducing stochastic uncertainty.
DETERMINISTIC: bool = False

# Demand autocorrelation: AR(1) coefficient for hour-to-hour demand errors.
# 0.0 = i.i.d. noise (previous behaviour), 0.7 = strongly correlated errors.
# Real demand forecast errors have hourly autocorrelation of 0.6-0.8.
DEMAND_AR1: float = 0.70

# Renewable stochasticity: add correlated wind/solar noise on top of residual demand.
# This represents intra-day forecast errors in renewable output.
# WIND_NOISE_STD: std of wind noise as fraction of demand peak (e.g. 0.03 = ±3% of peak).
# SOLAR_NOISE_STD: same for solar. Set to 0.0 to disable.
WIND_NOISE_STD:  float = 0.04   # ~4% of demand peak, correlated across regions
SOLAR_NOISE_STD: float = 0.03   # ~3% of demand peak, daytime only

# Forced outage stochasticity: apply per-scenario Bernoulli outage draws.
# True = each scenario independently samples which units are available each hour.
# False = only expected-value derate (previous behaviour, faster but less realistic).
FOR_STOCHASTIC: bool = True

# Hydro inflow uncertainty: resample inflow per scenario with AR(1) noise.
# Correlated with demand (cold/dark weather → low wind AND low hydro).
HYDRO_INFLOW_STOCHASTIC: bool = True
HYDRO_INFLOW_AR1:        float = 0.80   # strong day-to-day correlation
HYDRO_INFLOW_STD:        float = 0.15   # ±15% inflow noise (fraction of mean)
# Correlation between hydro inflow and demand error:
# Negative = cold dark calm weather → high demand AND low hydro (realistic for winter)
HYDRO_DEMAND_CORRELATION: float = -0.40
