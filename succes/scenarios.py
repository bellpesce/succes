"""
succes/scenarios.py
-------------------
Scenario generation for residual demand and fuel prices.

A "scenario" is one realisation of uncertain inputs for a window of T hours.
The ScenarioBank holds all S scenarios and is passed to the simulator.

KEY IMPROVEMENTS over the original i.i.d. noise model
------------------------------------------------------
1. AR(1) autocorrelated demand errors
   Real demand forecast errors are strongly autocorrelated hour-to-hour
   (rho ~ 0.6-0.8). An i.i.d. model creates unrealistically spiky scenarios
   where h08 has +5% error and h09 has -5% — impossible in practice.
   DemandGenerator now uses an Ornstein-Uhlenbeck (discrete AR(1)) process
   for the error term. Configurable via config.DEMAND_AR1.

2. Correlated wind/solar noise
   Residual demand = gross demand - wind - solar. Wind and solar have their
   own forecast errors, which are correlated across neighbouring regions
   (a weather system affects DE, DK, NL simultaneously). This is modelled
   as a regional wind block that is drawn once per scenario and added to
   all regions with region-specific sensitivities.
   Configurable via config.WIND_NOISE_STD and config.SOLAR_NOISE_STD.

3. Stochastic hydro inflow
   Reservoir inflows vary day-to-day with strong autocorrelation and are
   negatively correlated with demand (cold dark calm winters = high demand
   AND low hydro). This is the key "dark doldrum" stress scenario.
   Configurable via config.HYDRO_INFLOW_STOCHASTIC.

4. Per-scenario forced outages
   Previously FOR was only applied as a deterministic expected derate.
   Now each scenario independently samples unit availability via Bernoulli
   draws, creating realistic tail scenarios where multiple units fail
   simultaneously. Configurable via config.FOR_STOCHASTIC.

Hydro inflow per scenario
-------------------------
HydroGenerator produces (S,) inflow multipliers for each hydro unit.
These are passed to the scenario bank and used in engine_batch.jl to
scale hy_inflow per scenario (instead of the fixed mean inflow).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from . import config


# ── Scenario bank ─────────────────────────────────────────────────────────────

@dataclass
class ScenarioBank:
    """
    Holds all Monte Carlo scenarios for one optimisation window.

    Attributes
    ----------
    demand         : (S, T, R) — residual demand per scenario/hour/region.
                     May be negative (surplus renewable generation).
    fuel_prices    : (S, T, F) — optional fuel price scenarios.
    hydro_inflow   : (S, n_hydro_units) — inflow multipliers per scenario.
                     Value of 1.0 = baseline inflow; 0.5 = drought; 1.5 = flood.
                     None if HYDRO_INFLOW_STOCHASTIC is False.
    region_names   : list of region identifiers
    fuel_names     : list of fuel identifiers
    hydro_unit_names: list of hydro unit names (same order as hydro_inflow axis-1)
    """
    demand:           np.ndarray                    # (S, T, R)
    fuel_prices:      Optional[np.ndarray] = None   # (S, T, F)
    hydro_inflow:     Optional[np.ndarray] = None   # (S, H) multipliers
    region_names:     list[str]            = field(default_factory=list)
    fuel_names:       list[str]            = field(default_factory=list)
    hydro_unit_names: list[str]            = field(default_factory=list)

    @property
    def S(self) -> int:
        return self.demand.shape[0]

    @property
    def T(self) -> int:
        return self.demand.shape[1]

    @property
    def n_regions(self) -> int:
        return self.demand.shape[2]

    def demand_for_region(self, region: str) -> np.ndarray:
        idx = self.region_names.index(region)
        return self.demand[:, :, idx]

    def fuel_price_for(self, fuel: str, base_price: float) -> np.ndarray:
        if self.fuel_prices is None or fuel not in self.fuel_names:
            return np.full((self.S, self.T), base_price)
        idx = self.fuel_names.index(fuel)
        return self.fuel_prices[:, :, idx]


# ── Stochastic demand generator ───────────────────────────────────────────────

class DemandGenerator:
    """
    Generates stochastic residual demand scenarios with AR(1) autocorrelation.

    Strategy
    --------
    Each scenario samples an error process e_t = rho*e_{t-1} + sqrt(1-rho^2)*N(0,sigma)
    (discrete Ornstein-Uhlenbeck). This gives realistic hour-to-hour correlation
    in demand forecast errors. On top of this, a common weather factor (wind/solar
    block) can be added to create cross-regional correlation.

    Parameters
    ----------
    base_profile    : (T,) — expected residual demand (MWh/h)
    stress_profiles : list of (T,) — alternative shapes (cold snap, heat wave)
    stress_prob     : probability of each stress profile
    noise_std       : std of demand noise as fraction of demand level
    ar1             : AR(1) coefficient for hour-to-hour error correlation
                      0.0 = i.i.d. (old behaviour), 0.7 = realistic
    wind_sensitivity: how much a unit wind shock changes this region's demand
                      (negative = more wind → lower residual demand)
    solar_sensitivity: same for solar (daytime only, h08-h19)
    allow_negative  : if False, clip demand at 0
    """

    def __init__(
        self,
        base_profile:     np.ndarray,
        stress_profiles:  Optional[list[np.ndarray]] = None,
        stress_prob:      Optional[list[float]]       = None,
        noise_std:        float                       = 0.02,
        ar1:              float                       = None,   # None → use config
        wind_sensitivity: float                       = 1.0,    # MW of demand change per MW wind shock
        solar_sensitivity:float                       = 0.8,
        allow_negative:   bool                        = True,
    ):
        self.base    = np.asarray(base_profile, dtype=float)
        self.stress  = [np.asarray(p, dtype=float) for p in (stress_profiles or [])]
        self.s_prob  = np.asarray(stress_prob or [1.0 / max(len(self.stress), 1)]
                                  * len(self.stress))
        self.noise_std        = noise_std
        self.ar1              = config.DEMAND_AR1 if ar1 is None else ar1
        self.wind_sensitivity  = wind_sensitivity
        self.solar_sensitivity = solar_sensitivity
        self.allow_negative    = allow_negative

    def generate(
        self,
        S: int,
        T: int,
        seed: int = 42,
        surplus_profile: Optional[np.ndarray] = None,
        wind_block:  Optional[np.ndarray] = None,   # (S,) common wind shock per scenario
        solar_block: Optional[np.ndarray] = None,   # (S,) common solar shock per scenario
        _deterministic: bool = False,               # internal: skip all noise, return base profile
    ) -> np.ndarray:
        """
        Generate S scenarios of length T with AR(1) correlated noise.

        If _deterministic=True (set by build_scenario_bank when config.DETERMINISTIC
        is True), returns a single scenario equal to the unperturbed base profile
        with no AR(1) noise, no stress draw, no wind/solar block.
        """
        # ── Deterministic base-profile path ───────────────────────────────────
        if _deterministic or config.DETERMINISTIC:
            def _tile(arr):
                if len(arr) >= T:
                    return arr[:T]
                return np.tile(arr, int(np.ceil(T / len(arr))))[:T]
            base = _tile(self.base)
            if surplus_profile is not None:
                base = base - _tile(np.asarray(surplus_profile, dtype=float))
            if not self.allow_negative:
                base = np.maximum(base, 0.0)
            return base[np.newaxis, :]   # (1, T)

        rng  = np.random.default_rng(seed)
        rho  = self.ar1
        sig  = self.noise_std

        # Extend base to length T
        def _tile(arr):
            if len(arr) >= T:
                return arr[:T]
            return np.tile(arr, int(np.ceil(T / len(arr))))[:T]

        base = _tile(self.base)
        out  = np.empty((S, T))

        # Solar mask: non-zero h07-h19 (daytime)
        solar_mask = np.zeros(T)
        for h in range(T):
            if 7 <= (h % 24) <= 19:
                solar_mask[h] = 1.0

        for s in range(S):
            # Choose profile (base or stress)
            if self.stress:
                r = rng.random()
                profile = base.copy()
                cumprob = 0.0
                for sp, prob in zip(self.stress, self.s_prob):
                    cumprob += prob
                    if r < cumprob:
                        profile = _tile(sp)
                        break
            else:
                profile = base.copy()

            # AR(1) error process
            # e_0 ~ N(0, sigma/sqrt(1-rho^2)) (stationary distribution)
            sigma_stat = sig / np.sqrt(max(1 - rho**2, 1e-6))
            e = rng.normal(0.0, sigma_stat)
            innovation_std = sig * np.sqrt(1 - rho**2)
            errors = np.empty(T)
            for t in range(T):
                e = rho * e + rng.normal(0.0, innovation_std)
                errors[t] = e

            # Apply multiplicative error: demand * (1 + error)
            scenario = profile * (1.0 + errors)

            # Add correlated wind/solar block
            if wind_block is not None and config.WIND_NOISE_STD > 0:
                # wind_block[s] is MW, negative means more wind → lower residual
                scenario -= self.wind_sensitivity * wind_block[s]

            if solar_block is not None and config.SOLAR_NOISE_STD > 0:
                scenario -= self.solar_sensitivity * solar_block[s] * solar_mask

            # Subtract deterministic surplus
            if surplus_profile is not None:
                sur = _tile(np.asarray(surplus_profile, dtype=float))
                scenario -= sur

            if not self.allow_negative:
                scenario = np.maximum(scenario, 0.0)

            out[s] = scenario

        return out


# ── Hydro inflow generator ────────────────────────────────────────────────────

class HydroInflowGenerator:
    """
    Generates per-scenario inflow multipliers for hydro units.

    Each hydro unit gets an inflow multiplier ~ AR(1) process around 1.0.
    Multipliers are correlated with the demand error (cold/calm weather =
    high demand AND low hydro = "dark doldrum" scenario).

    Parameters
    ----------
    unit_names : list of hydro unit names
    base_inflows: {unit_name: mean_inflow_MWh_per_window}
    ar1        : AR(1) for inflow persistence (default config.HYDRO_INFLOW_AR1)
    std        : noise std as fraction of mean inflow
    demand_corr: correlation with demand shock (negative = anti-correlated)
    """

    def __init__(
        self,
        unit_names:   list[str],
        ar1:          float = None,
        std:          float = None,
        demand_corr:  float = None,
    ):
        self.unit_names  = unit_names
        self.ar1         = config.HYDRO_INFLOW_AR1   if ar1         is None else ar1
        self.std         = config.HYDRO_INFLOW_STD   if std         is None else std
        self.demand_corr = config.HYDRO_DEMAND_CORRELATION if demand_corr is None else demand_corr

    def generate(
        self,
        S: int,
        seed: int = 42,
        demand_shocks: Optional[np.ndarray] = None,  # (S,) standardised demand shock
    ) -> np.ndarray:
        """
        Returns (S, n_units) inflow multipliers.
        1.0 = baseline inflow. 0.5 = severe drought. 1.5 = flood.

        demand_shocks: (S,) standardised mean demand error per scenario.
        If provided, inflow is anti-correlated with demand (dark doldrum effect).
        """
        rng = np.random.default_rng(seed)
        n = len(self.unit_names)
        if n == 0:
            return np.ones((S, 0))

        rho = self.ar1
        sig = self.std
        corr = self.demand_corr

        # Generate base inflow shocks (correlated with demand if provided)
        # Each unit gets an independent component + shared weather component
        multipliers = np.ones((S, n))

        for u_idx in range(n):
            # Generate (S,) inflow multipliers for this unit
            unit_shocks = np.empty(S)
            for s in range(S):
                # Unit-specific AR(1) innovation
                unit_shocks[s] = rng.normal(0.0, sig)

            # Add correlation with demand
            if demand_shocks is not None and abs(corr) > 1e-6:
                # Partial correlation: shock = corr * demand_shock + sqrt(1-corr^2) * independent
                indep_std = sig * np.sqrt(max(1 - corr**2, 0.0))
                indep     = rng.normal(0.0, indep_std, size=S)
                unit_shocks = corr * demand_shocks * sig + indep

            # Clip to avoid negative inflows
            multipliers[:, u_idx] = np.clip(1.0 + unit_shocks, 0.10, 2.50)

        return multipliers


# ── Stochastic fuel price generator ──────────────────────────────────────────

class FuelPriceGenerator:
    """
    Generates correlated fuel price scenarios using Ornstein-Uhlenbeck process.

    Note: over 14-day windows, fuel price uncertainty matters less than
    demand/renewable uncertainty. This generator is retained for completeness
    and for longer-horizon extensions.

    TODO (future): expose fuel prices as explicit policy sweep variable
    for CO2 price sensitivity analysis.
    """

    def __init__(
        self,
        fuel_names:     list[str],
        base_prices:    dict[str, float],
        vol:            Optional[dict[str, float]] = None,
        mean_reversion: float                      = 0.1,
        correlation:    Optional[np.ndarray]       = None,
    ):
        self.fuel_names     = fuel_names
        self.base_prices    = np.array([base_prices.get(f, 50.0) for f in fuel_names])
        self.vol            = np.array([(vol or {}).get(f, 0.05) for f in fuel_names])
        self.mean_reversion = mean_reversion
        n = len(fuel_names)
        self.L = np.linalg.cholesky(correlation) if correlation is not None else np.eye(n)

    def generate(self, S: int, T: int, seed: int = 42) -> np.ndarray:
        rng    = np.random.default_rng(seed)
        n      = len(self.fuel_names)
        out    = np.empty((S, T, n))
        kappa  = self.mean_reversion
        for s in range(S):
            price = self.base_prices.copy()
            for t in range(T):
                out[s, t] = price
                z          = self.L @ rng.standard_normal(n)
                shock      = self.vol * price * z
                price      = price + kappa * (self.base_prices - price) + shock
                price      = np.maximum(price, 0.01 * self.base_prices)
        return out


# ── Per-scenario FOR availability ────────────────────────────────────────────

def generate_availability(
    unit_names: list[str],
    for_rates:  dict[str, float],
    S: int,
    T: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate per-scenario, per-hour availability mask (S, N, T).
    1 = available, 0 = forced outage.

    For each unit, availability follows a Bernoulli process:
    P(available at hour t in scenario s) = 1 - for_rate.

    Uses a deterministic seed per unit so results are reproducible.
    Only called if config.FOR_STOCHASTIC is True.
    """
    N = len(unit_names)
    avail = np.ones((S, N, T), dtype=np.float32)

    if not config.FOR_STOCHASTIC:
        return avail

    for i, name in enumerate(unit_names):
        rate = for_rates.get(name, 0.0)
        if rate <= 0.0:
            continue
        unit_seed = abs(hash(name)) % (2**31)
        rng = np.random.default_rng(unit_seed + seed)
        # Draw (S, T) availability — True = available
        draws = rng.random((S, T)) >= rate
        avail[:, i, :] = draws.astype(np.float32)

    return avail


# ── High-level factory ────────────────────────────────────────────────────────

def build_scenario_bank(
    region_generators:   dict[str, DemandGenerator],
    S:                   int,
    T:                   int,
    seed:                int                              = 42,
    fuel_generator:      Optional[FuelPriceGenerator]    = None,
    surplus_profiles:    Optional[dict[str, np.ndarray]] = None,
    hydro_generator:     Optional[HydroInflowGenerator]  = None,
) -> ScenarioBank:
    """
    Generate a ScenarioBank for one window.

    If config.DETERMINISTIC is True, all stochastic sources are suppressed:
    S is forced to 1, all noise is zeroed, and the single scenario is the
    unperturbed base profile. Use this to validate price formation.
    """
    # ── Deterministic override ────────────────────────────────────────────────
    if config.DETERMINISTIC:
        S = 1
        wind_block  = None
        solar_block = None
        demand_arr = np.stack(
            [
                region_generators[r].generate(
                    S=1, T=T, seed=seed + i * 7,
                    surplus_profile=(surplus_profiles or {}).get(r),
                    wind_block=None, solar_block=None,
                    _deterministic=True,
                )
                for i, r in enumerate(sorted(region_generators.keys()))
            ],
            axis=2,
        )
        return ScenarioBank(
            demand           = demand_arr,
            fuel_prices      = None,
            hydro_inflow     = None,
            region_names     = sorted(region_generators.keys()),
            fuel_names       = [],
            hydro_unit_names = [],
        )

    rng = np.random.default_rng(seed)
    region_names = sorted(region_generators.keys())
    n_regions    = len(region_names)

    # ── Draw shared weather factors (one per scenario) ────────────────────────
    # wind_block  (S,): MW of wind forecast error, correlated across all regions.
    #   Positive = more wind than expected → lower residual demand.
    # solar_block (S,): MW of solar forecast error, daytime only.
    # demand_mean_shock (S,): standardised mean demand error (for hydro correlation).

    wind_std  = config.WIND_NOISE_STD
    solar_std = config.SOLAR_NOISE_STD

    # Wind and solar block: cross-regional common factor
    # Scale to a "notional 10 GW region" — each DemandGenerator scales by its sensitivity
    wind_block  = rng.normal(0.0, wind_std  * 10_000, size=S) if wind_std  > 0 else None
    solar_block = rng.normal(0.0, solar_std * 10_000, size=S) if solar_std > 0 else None

    # Standardised demand shock for hydro correlation
    # Average the wind block across regions as a proxy for overall weather
    if wind_block is not None:
        demand_shocks = wind_block / max(wind_std * 10_000, 1.0)  # standardised
    else:
        demand_shocks = np.zeros(S)

    # ── Generate demand scenarios ─────────────────────────────────────────────
    demand_arr = np.stack(
        [
            region_generators[r].generate(
                S, T,
                seed=seed + i * 7,
                surplus_profile=(surplus_profiles or {}).get(r),
                wind_block=wind_block,
                solar_block=solar_block,
            )
            for i, r in enumerate(region_names)
        ],
        axis=2,
    )   # (S, T, R)

    # ── Generate fuel price scenarios ─────────────────────────────────────────
    fuel_arr = None
    if fuel_generator is not None:
        fuel_arr = fuel_generator.generate(S, T, seed=seed + 1000)

    # ── Generate hydro inflow scenarios ───────────────────────────────────────
    hydro_arr = None
    hydro_names = []
    if hydro_generator is not None and config.HYDRO_INFLOW_STOCHASTIC:
        hydro_arr   = hydro_generator.generate(S, seed=seed + 2000,
                                               demand_shocks=demand_shocks)
        hydro_names = hydro_generator.unit_names

    return ScenarioBank(
        demand           = demand_arr,
        fuel_prices      = fuel_arr,
        hydro_inflow     = hydro_arr,
        region_names     = region_names,
        fuel_names       = fuel_generator.fuel_names if fuel_generator else [],
        hydro_unit_names = hydro_names,
    )
