# SUCCES — Stochastic Unit Commitment for Coupled European Scenarios

**Version:** v3 (bug-fix release — fixed-unit dispatch, unique OCGT names, NL capacity, pre-pass flows)
**Architecture:** Python orchestration + Julia GA/dispatch engine
**Regions:** AT, BE, CH, CZ, DE, DK, FR, GB, NL, NO, PL, SE (12-region CWE + Nordic + GB)

---

## Changelog

### v3 — Bug-fix release (2026-05-24)

Four bugs that corrupted results or silenced output were fixed. **All changes are
backward-compatible** — no payload schema version bump required; the new
`fixed_units_commit` key is additive and gracefully defaults in older builds.

| # | File(s) | Symptom | Root cause | Fix |
|---|---------|---------|------------|-----|
| 1 | `julia_bridge.py` | FR_Nuclear dispatching ~545 MW instead of ~45,000 MW (and all other `fixed_on=True` units mishandled) | `engine_batch.jl` reads `"fixed_units_commit"` to set `N_fixed`, but `julia_bridge.py` only sent `"commitment"` (fixed+free merged). So `N_fixed=0` for every region and the GA chromosome overwrote all fixed units. | Added `"fixed_units_commit": fixed_com.tolist()` as a separate key in `rdata[r]`. `"commitment"` is kept unchanged for `engine.jl` which reads it directly. |
| 2 | `examples/europe_nordic.py` | CZ, DK, PL had doubled dispatch arrays (672 values instead of 336) in the result JSON | `_add_ocgt_tier("CZ/DK/PL", 2, ...)` was called twice, creating duplicate unit names `CZ_OCGT_T3_01`…`_08`. `solver._auto_report` appended to `gen[u.name]` twice for duplicate names. | Added `OCGT_TIERS[3]` (label `"T4"`, MC ≈159 EUR/MWh) and changed the three extra-capacity calls to `tier_idx=3`. Units now named `_T4_01`…`_T4_08`. |
| 3 | `examples/europe_nordic.py` | NL mean price 155 EUR/MWh (target 80–100); only 42 MW mean import despite DE surplus | NL thermal capacity was 14,700 MW (1.47× peak). With stochastic renewable shortfalls the fleet hit scarcity frequently. The 25% pre-pass headroom cap additionally limited DE→NL flows. | Added one extra NL OCGT T4 tier (8 × 250 MW = 2,000 MW). NL cap/peak now 1.64× (16,400 MW vs 10,000 MW peak). |
| 4 | `succes/engine.jl` | `mean_flows` all zero in JSON output; reporter flow map blank | `flows_mean_TL` was built from `flows_3d` (post-dispatch net_pos), which is always ≈0 because dispatch consumed the demand adjustment baked in by the pre-pass. | Introduced `flows_pre_3d[S,T,L]` tensor. Per-link flows are recorded during the ATC inner loop and averaged into `flows_mean_TL`. `flows_3d` (from `cross_border_flows!`) is kept as a fallback. |

---

## Architecture Overview

```
europe_nordic.py          ← scenario definition (fleet, demand profiles, scenario generators)
succes/
  config.py               ← ALL tuneable hyperparameters (one place to change everything)
  scenarios.py            ← stochastic scenario generation (demand, hydro, FOR, renewables)
  assets.py               ← fleet asset classes (ThermalPlant, HydroPlant, StorageAsset)
  network.py              ← transmission network (ATC + PTDF)
  solver.py               ← rolling horizon solver
  julia_bridge.py         ← Python↔Julia data serialisation
  julia_ga.py             ← GA orchestration (calls engine_ga.jl via juliacall)
  engine_ga.jl            ← GA loop (warm-start, diversity restart, adaptive epochs)
  engine_batch.jl         ← fitness evaluator (4-phase dispatch × S scenarios × P individuals)
  engine.jl               ← final deterministic dispatch
  reporter.py             ← HTML report + flows visualisation
```

**Data flow:**
1. `europe_nordic.py` builds fleet + scenario functions
2. `solver.py` iterates 14 × 24h windows
3. Per window: Python builds scenario bank → serialises to Julia payload → `engine_ga.jl` runs GA
4. GA evaluates each individual via `engine_batch.jl` (4-phase dispatch, S scenarios)
5. Best solution → `engine.jl` final dispatch → results accumulated
6. `reporter.py` generates HTML report

---

## Scenario Generation (`scenarios.py`)

### Demand — AR(1) autocorrelated errors

Real demand forecast errors are strongly autocorrelated hour-to-hour (ρ ≈ 0.6–0.8).
The old model used i.i.d. noise, creating unrealistic spiky scenarios.

**Implementation:** Discrete Ornstein-Uhlenbeck process
```
e_t = ρ · e_{t-1} + √(1-ρ²) · N(0, σ)
demand_s,t = profile_t · (1 + e_t)
```

**Config:** `DEMAND_AR1 = 0.70` (realistic hourly autocorrelation)

### Wind and Solar — Spatially Correlated Forecast Errors

Wind and solar forecast errors are correlated across neighbouring regions
(a weather system affects DE, DK, NL simultaneously).

**Implementation:** A single `wind_block (S,)` and `solar_block (S,)` are drawn
once per scenario by `build_scenario_bank` and passed to all `DemandGenerator`
instances. Each region scales the block by its `wind_sensitivity` parameter.

Solar noise applies only during daytime hours (h07–h19), reflecting that solar
forecast errors are zero overnight.

**Config:** `WIND_NOISE_STD = 0.04`, `SOLAR_NOISE_STD = 0.03`
(Set to 0.0 to disable each independently)

**Region sensitivities** (in `europe_nordic.py`):
- High wind: DE 1.2×, DK 1.5×, GB 1.4×, NL 1.3×
- Moderate: FR 0.8×, BE 0.9×, PL 0.7×
- Low (hydro dominated): AT 0.3×, CH 0.2×, NO 0.4×

### Hydro Inflow — Dark Doldrum Correlation

The "dark doldrum" scenario: cold, still, cloudy winter weather simultaneously
produces **high demand** (heating load) AND **low hydro inflow** (frozen snowpack)
AND **low wind** output. This is the hardest UC scenario.

**Implementation:** `HydroInflowGenerator` produces (S, N_hydro) multipliers
per scenario. Multipliers are anti-correlated with the shared demand/wind shock
via a partial correlation model.

**Config:**
```python
HYDRO_INFLOW_STOCHASTIC = True    # False → deterministic baseline inflow
HYDRO_INFLOW_AR1        = 0.80    # strong day-to-day persistence
HYDRO_INFLOW_STD        = 0.15    # ±15% noise around mean inflow
HYDRO_DEMAND_CORRELATION = -0.40  # negative: high demand = low hydro
```

### Forced Outage Rates — Per-Scenario Bernoulli

Previously: FOR was only a deterministic expected-value derate
(`commitment × (1 - for_rate)`). This produced no stochastic tail scenarios.

**Now:** Per-scenario Bernoulli sampling. Each scenario independently draws
which units are unavailable at each hour. Creates realistic tail scenarios
where multiple units fail simultaneously (correlated stress).

**Config:** `FOR_STOCHASTIC = True` (False → reverts to deterministic derate)

**Julia engine:** `engine_batch.jl` Phase 1 checks `avail_matrix[s, i, t]`
before dispatching unit i. Unavailable units contribute 0 MW in that scenario.

### Seasonal Calibration — Winter

All demand profiles represent **European winter (January) conditions**:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Demand peaks | DE 45 GW, FR 52 GW, GB 30 GW | Gross demand minus wind/solar at winter output |
| Valley ratios | 0.30–0.36 | Deep overnight trough (industry down, heating load) |
| Wind sensitivity | DE 1.2×, DK 1.5× | Large offshore+onshore fleets |
| FOR rates | +20-30% vs annual | Cold-weather thermal stress |
| Hydro inflow | Low (min seasonal) | Snowpack frozen, rainfall low |
| CO2 price | 82 EUR/tCO2 | 2023-era ETS baseline |
| Gas price | 35 EUR/MWh | TTF winter baseline |

**Future:** Calibrate against ENTSO-E Transparency Platform hourly data
for Jan 2023/2024 to get quantitative validation anchors. (TODO item)

---

## GA Engine (`engine_ga.jl`)

### Hyperparameters (all in `config.py`)

All three features can be **independently toggled on/off** via config booleans.

#### Warm-Start (`WARM_START_ENABLED`)

Seeds a fraction of the population with the best solution from the previous window.
Demand changes ~5–10% day-to-day, so the previous optimal commitment is a good
starting point. Reduces epochs needed to converge and cuts the "cold start" penalty
on early windows.

```python
WARM_START_ENABLED   = True
WARM_START_FRACTION  = 0.25   # 25% of population seeded from prev window
```

**Implementation:** `_run_julia_batch_ga_state` stores `prev_best` in `julia_ga.py`.
Passed as `prev_best_solution` in payload. Julia seeds slots 1..n_warm with
mutated copies of the previous best.

#### Diversity Restart (`DIVERSITY_RESTART_ENABLED`)

When the GA stalls (no improvement for `STALL_PATIENCE` epochs), the worst
`DIVERSITY_REPLACE_FRAC` of the population is replaced with mutated copies of
the best solution. This escapes local optima and extends productive search.

```python
DIVERSITY_RESTART_ENABLED = True
STALL_PATIENCE            = 50    # epochs without improvement before restart
DIVERSITY_REPLACE_FRAC    = 0.20  # 20% of population replaced
```

The replacement uses a higher mutation rate (pm=0.05 vs standard 0.005) to
ensure diversity. Fitness of replaced individuals is set to Inf to force
immediate re-evaluation.

#### Adaptive Epoch Budget (`ADAPTIVE_EPOCHS_ENABLED`)

Stops the GA early if it has converged. Convergence is defined as < `ADAPTIVE_MIN_IMPROVEMENT`
relative improvement over `ADAPTIVE_PATIENCE` consecutive epochs.

```python
ADAPTIVE_EPOCHS_ENABLED   = True
ADAPTIVE_PATIENCE         = 80    # stop if flat for 80 epochs
ADAPTIVE_MIN_IMPROVEMENT  = 1e-4  # 0.01% relative improvement threshold
```

Effect: 2–5x faster on easy windows (weekends, low-demand days). The saved
compute is not currently redistributed; it simply returns early.

### GA Structure

- **Population:** 80 individuals (configurable via `GA_POP_SIZE`)
- **Representation:** binary vector of length `n_free_units × T` — one bit per unit per hour
- **Island model:** 2 islands of 40, migration every 20 epochs
- **Crossover:** block crossover (respects unit boundaries) at rate `GA_PC = 0.85`
- **Mutation:** overnight block flip + random bit flip at `GA_PM = 0.005`
- **Elitism:** global best always survives; island best preserved within island

---

## PTDF Network Coupling

The 15 CWE AC interconnectors use DC load-flow PTDF constraints.
The 10 HVDC cables (NordLink, Viking, IFA, BritNed, etc.) use bilateral ATC.

**PTDF matrix:** 15×12, computed once at startup from line reactances.
**RAM:** 75% of thermal rating (standard ENTSO-E FBMC reliability margin).
**Slack bus:** DE (largest region, most connections).

After bilateral ATC trades are computed in Phase 2, the net export vector is
projected onto physical AC lines via `PTDF × net_exports`. If any line exceeds
its RAM, all trades are scaled down proportionally (no LP — O(L×R) per hour).

---

## Fleet Design

### Technology tiers

Each fuel type has 4 cost/flexibility tiers. Tier diversity is essential for GA
difficulty: the GA must choose *which* T1 units to commit, not just *whether* to
run T1s. With 12 units per tier (350 MW each), the combinatorial space is C(12,k)
which is orders of magnitude larger than C(6,k) with 780 MW units.

| Tier | CCGT MC | Startup | Min-run | Purpose |
|------|---------|---------|---------|---------|
| T1 | ~89 EUR/MWh | 12k | 2h | Base, always committed first |
| T2 | ~96 | 18k | 3h | Mid-merit |
| T3 | ~107 | 22k | 4h | Peak cover, 4h min-run creates lead-time GA pressure |
| T4 | ~119 | 28k | 2h | True peak only |

### DE lignite (replaces nuclear)

Lignite at CO2=82: MC ≈ 97–99 EUR/MWh. Not price-competitive with CCGT T1/T2
but `min_run_hours = 16–18` means once committed it runs all day. This creates
the key UC temporal coupling: the GA must decide at h04 whether to commit
lignite knowing it locks in cost through h20.

### Capacity sizing

Target: base stack (CCGT + lignite + coal) = ~70% of peak demand.
At valley (32% of peak): ~32%/70% = 46% of base stack capacity is needed.
OCGTs and NewPeak fill the remaining 30% at peak.

---

## Storage

**Current:** StorageAsset (batteries, pumped hydro) is dispatched in Phase 3 of
the engine (discharge before thermal top-up, charge on surplus). The dispatch is
automatic — no binary commitment decision for storage.

**Why:** Storage doesn't have startup costs or min-run constraints, so the binary
UC framework doesn't apply directly. A proper storage optimisation requires
a continuous charge/discharge schedule optimised jointly with thermal commitment.

**TODO:** Add storage SOC cost-to-go (water value equivalent) as a term in the
GA fitness function, giving the GA a signal about the value of stored energy
at end-of-window. See TODO list below.

---

## Running

```
set JULIA_NUM_THREADS=8
set PYTHON_JULIACALL_HANDLE_SIGNALS=yes
python examples/europe_nordic.py
```

Results: `examples/results_europe_nordic.html` and `.json`

---

## Dispatch Model — Limitations and Simplifications

### What the engine does

**Phase 1 — baseline dispatch:**
Every committed unit is dispatched at its lower bound (`lo = max(min_cap, prev_dispatch - ramp_rate)`).
This is the minimum it must produce given its ramp constraints from the previous hour.
`remain = demand - sum(lo)` is the residual demand the merit order must fill.

**Phase 3 — merit order top-up:**
Remaining demand is filled by iterating through committed units in merit order (cheapest first),
loading each from its current dispatch level up to its ramp-constrained maximum.
The last unit needed sets the marginal clearing price (MCP).

**Fuel cost = sum(dispatch × marginal_cost)** — the GA receives a correct signal
for committed units: only units actually dispatched incur fuel cost.

### Is "fill cheapest unit completely before touching the next one" realistic?

**Yes, for the purpose of this model.** In European day-ahead markets, the merit
order determines which units are price-setters at each hour. All cheaper committed
units are expected to run at their available capacity. This is the standard
**price-taking merit order economic dispatch** that underpins virtually all
academic UC models, and is a good approximation of how EPEX/NordPool day-ahead
markets clear at 24h resolution.

Ramp constraints are applied correctly per scenario: a unit dispatching 100 MW
last hour cannot exceed `prev_dispatch + ramp_rate` this hour (`hi_s` is capped).

`min_cap = 0` means committed units can produce 0 MW when idle (hot standby).
This is correct — the GA's binary decision is "unit is available" not
"unit is running at minimum load."

### Where the model diverges from reality

**1. No strategic bidding / capacity withholding.**
In reality, generators sometimes offer only a fraction of their capacity at
marginal cost and withhold the rest at a higher price (economic withholding).
Our engine dispatches all available capacity at marginal cost. Effect:
prices are slightly too low and too smooth vs. actual markets.
Including strategic bidding would require a game-theoretic equilibrium model —
a fundamentally different research question. Acceptable simplification at this scale.

**2. Cross-border headroom slightly overestimated.**
In Phase 2, export headroom is estimated as `sum(hi - lo)` — total available
capacity above baseline. After Phase 3 domestic top-up, some of that headroom
is consumed. The ATC arbitrage in Phase 2 doesn't know in advance how much
domestic demand will absorb. This can overestimate available export capacity
by ~5–10%. Effect: cross-border flows are slightly too large in congested hours.
Not material for UC decisions but affects flow visualisation accuracy.

**3. Single clearing price per region per hour (zonal pricing).**
Real markets have nodal or zonal prices. Within a region (e.g. DE), we assume
a single price regardless of internal transmission constraints. This is consistent
with the current CWE zonal market design but would need extension for nodal models.

**4. No startup energy consumption.**
Real units consume fuel during startup (warming the boiler). We model startup
cost as a fixed EUR payment but not as an energy/emissions cost. Minor for
gas turbines; more material for large coal/lignite. Acceptable simplification.

**5. No minimum down-time.**
Once a unit is switched off, our model allows it to restart in the next hour.
Real plants have minimum down-time (e.g. a coal unit that shuts down needs 6-8h
before it can restart). This means the GA can cycle units more aggressively than
reality would allow. Could be added as a `min_down_hours` constraint per unit.
Currently mitigated in practice by the overnight block mutation operator which
tends to keep units off for multi-hour blocks.

**6. Storage dispatch not GA-optimised.**
Battery and pumped hydro are dispatched automatically in Phase 3 (discharge when
demand deficit, charge on surplus). The GA doesn't optimise the storage schedule.
This means the GA commitment decisions don't account for storage arbitrage value,
potentially leading to over-commitment of thermal in windows where storage could
cover peak demand. Partially mitigated by adding SOC cost-to-go signal (TODO).

### Summary judgement

The dispatch model is appropriate for a 24h-resolution UC simulator targeting
European market behaviour. The merit order approximation, zonal pricing, and
thermal-first dispatch are consistent with how ENTSO-E day-ahead markets actually
clear. The model correctly captures: startup costs, ramp constraints, min-run
temporal coupling, cross-border ATC/PTDF coupling, and stochastic demand/outage
uncertainty. The gaps (strategic bidding, min down-time, storage GA integration)
are known and manageable for the current research phase.


---

## TODO List

### Fixed in v3 (no longer open)

- [x] **FR_Nuclear near-zero dispatch** — `fixed_units_commit` key missing from
  julia_bridge payload; `N_fixed=0` in engine_batch. Fixed in `julia_bridge.py`.
- [x] **Doubled dispatch arrays (CZ/DK/PL)** — duplicate `OCGT_T3` unit names from
  calling `_add_ocgt_tier(..., tier_idx=2, ...)` twice. Fixed by adding T4 tier to
  `OCGT_TIERS` and using `tier_idx=3` for the extra-capacity blocks.
- [x] **NL structural under-supply (155 EUR/MWh)** — cap/peak was 1.47×; added
  NL OCGT T4 tier (2,000 MW), raising ratio to 1.64×.
- [x] **mean_flows all zero** — `flows_mean_TL` was built from post-dispatch
  `flows_3d` (always ≈0). Fixed: `flows_pre_3d[S,T,L]` records flows during
  the ATC pre-pass; `flows_mean_TL` now averages those.

### Critical (architecture)

- [ ] **ATC coupling in direct-dispatch GA [MAJOR]:**
  The current GA optimises each region to balance supply=demand independently.
  This means net_position ≈ 0 in all regions → ATC trades nothing in the final
  engine.jl dispatch. Regional price spreads remain large (DE=127, FR=46 EUR/MWh)
  when they should partially equalise.
  
  ROOT CAUSE: the GA chromosome controls dispatch fractions per unit, but has no
  variables for cross-border export/import decisions. The ATC phase in
  engine_batch.jl runs after dispatch is set, with no residual to trade.
  
  SOLUTION OPTIONS (in order of complexity):
  A. Add export fraction variables to chromosome: x_export[link,t] ∈ [0,1].
     The GA simultaneously optimises dispatch AND trade. Adds 25×24=600 vars.
     Fitness: when A exports to B, A's "effective demand" increases (costs more)
     and B's decreases (costs less). The net cost is the price differential.
  B. Two-pass ATC: run a preliminary price-based ATC (using merit-order prices)
     to estimate trade volumes, adjust regional demand, then compute dispatch.
     Less clean but architecturally simpler.
  C. Iterative coupling: alternate GA dispatch optimisation and ATC price
     equalisation until convergence. Most rigorous, most complex.
  
  Option A is recommended. It adds a natural market coupling signal to the GA
  and is consistent with how real cross-border capacity is allocated.

- [ ] **PL structural over-pricing:** PL shows mean=200 EUR/MWh, max=1950 EUR/MWh,
  30 scarcity hours. Either PL installed capacity is insufficient or ATC imports
  from DE/CZ are not materialising due to the coupling gap above. Verify PL
  capacity/demand ratio and fix after ATC coupling is resolved.

- [ ] **NO scarcity (1278 EUR/MWh peak):** NO has only hydro + 4 OCGTs. When
  hydro inflow scenarios are low, NO hits scarcity. May need more OCGT capacity
  or more HVDC import capacity from SE/DE.

### High priority

- [ ] **Warm-start validation:** Convergence improving 27-42% per window is good.
  Verify warm-start reduces penalty on first 2-3 windows vs cold-start baseline.

- [ ] **Storage SOC cost-to-go signal:** GA dispatches storage without knowing
  the value of stored energy at end-of-window. Add terminal SOC penalty to GA
  fitness so it preserves storage for next window.

- [ ] **ENTSO-E calibration:** DE=127, FR=46 EUR/MWh is directionally correct
  (Germany gas-marginal, France nuclear-marginal) but prices are too separated.
  **Note:** after the v3 FR_Nuclear fix, FR prices will rise (nuclear was not
  dispatching, so FR was artificially cheap) and DE prices should fall (more FR
  exports will suppress DE prices). Re-benchmark after first v3 run before calibrating.
  Calibrate against ENTSO-E Jan 2024 hourly data. Target: DE 80-100, FR 60-90,
  NL 80-100 (was 155; should improve after capacity + flow fixes).

- [ ] **Startup cost calibration:** 10.9% of total cost is startup (€469M).
  Real European markets: startup costs are 1-3% of total. Either the GA is
  overcycling (likely - Gaussian mutation causes too many threshold crossings)
  or startup costs are too high. Consider adding a smoothing penalty on
  |x[i,t] - x[i,t-1]| to discourage rapid cycling.

- [ ] **Sensitivity analysis runner:** Systematic sweeps of lambda, ATC capacities,
  fleet sizing. Essential for calibration once ATC coupling is working.

- [ ] **Scarcity penalty tuning:** 7.5% penalty suggests the GA is sometimes
  under-dispatching to save fuel cost. The scarcity penalty (3000 EUR/MWh)
  should dominate — increase if needed.

### Medium priority

- [ ] **CO2 shadow price as policy variable:** Expose CO2_PRICE as a sweep
  parameter. The merit-order flip between gas and coal at ~70 EUR/tCO2 is one
  of the most important policy questions. Currently baked into unit marginal costs.

- [ ] **Wind/solar as explicit generation assets:** Currently modelled as
  negative residual demand. Adding explicit wind/solar assets with per-scenario
  output profiles would enable curtailment tracking and separate price effects.

- [ ] **Scenario versioning:** YAML scenario registry with checksums. Essential
  for reproducibility once results are shared or published. Each named scenario
  should be uniquely identified and results tagged to it.

- [ ] **Per-run regression tests:** pytest suite with a 2-region 48h smoke test
  that checks price ranges, zero scarcity, and GA convergence > 1%. Runs in
  < 2 minutes. Guards against calibration regressions when code changes.

- [ ] **Adaptive epoch redistribution:** When a window converges early (adaptive
  stopping), log the saved epochs and optionally reallocate to harder windows
  (identified by poor convergence in previous run).

### Lower priority / future work

- [ ] **Hybrid binary+continuous GA for storage:** Full joint optimisation of
  thermal commitment (binary) and storage charge/discharge schedule (continuous).
  Requires architecture change to GA chromosome and crossover/mutation operators.

- [ ] **Fuel price scenarios as explicit policy sweep:** Currently over 14 days,
  fuel price uncertainty is secondary. For longer horizons (seasonal), gas and
  coal price scenarios become material.

- [ ] **Open-source preparation:** README, pyproject.toml, GitHub Actions CI,
  comprehensive docstrings. Once the model is stable and validated.

- [ ] **Validation module:** Automated comparison against reference metrics.
  D-P correlation > 0.85, mean price within ±25% of reference, no systematic
  scarcity, startup cost / total cost ratio in realistic range.

- [ ] **Cross-run comparison in reporter:** Side-by-side charts of this run vs
  previous run on the same scenario. Critical for tracking the effect of
  each code change.
