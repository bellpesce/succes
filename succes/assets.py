"""
succes/assets.py
----------------
All dispatchable and storage assets in the fleet.

Design principle: every asset exposes the same interface to the simulator:
    - `capacity_available(t)`  -> float   max MW at hour t
    - `min_output(t)`          -> float   min MW when committed at hour t
    - `fuel_cost_at(t, fuel_prices)` -> float  €/MWh marginal cost at hour t

Availability is modelled as a numpy array `avail` of shape (T,) with values
in [0, 1]. This handles:
  - Planned maintenance (set avail[t] = 0)
  - Partial outages   (set avail[t] = 0.5)
  - Seasonal hydro   (varies smoothly)
  - Forced outage    (Monte Carlo: sample 0/1 per scenario — future extension)

All dataclasses use __slots__ where possible for memory efficiency.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from typing import Optional


class FuelType(str, Enum):
    NUCLEAR  = "nuclear"
    GAS      = "gas"
    COAL     = "coal"
    OIL      = "oil"
    HYDRO    = "hydro"
    BIOMASS  = "biomass"
    ELECTRIC = "electric"   # heat pump / power-to-heat
    NONE     = "none"       # storage, demand response


# ── Base asset ────────────────────────────────────────────────────────────────

@dataclass
class Asset:
    """
    Minimal common interface. All assets inherit from this.

    Parameters
    ----------
    name        : unique string identifier
    region      : which market region this asset belongs to
    max_cap     : nameplate capacity in MW
    avail       : availability profile, shape (T,), values in [0,1].
                  Pass None to use a flat 1.0 profile.
                  Resampled to the window length at solve time.
    """
    name:    str
    region:  str
    max_cap: float
    avail:   Optional[np.ndarray] = field(default=None, repr=False)

    def availability(self, T: int) -> np.ndarray:
        """Return availability array of length T, always in [0, 1]."""
        if self.avail is None:
            return np.ones(T)
        if len(self.avail) >= T:
            return np.clip(self.avail[:T], 0.0, 1.0)
        # tile if shorter than window
        reps = int(np.ceil(T / len(self.avail)))
        return np.clip(np.tile(self.avail, reps)[:T], 0.0, 1.0)


# ── Thermal plant ─────────────────────────────────────────────────────────────

@dataclass
class ThermalPlant(Asset):
    """
    Classic thermal unit: coal, gas, nuclear, oil, biomass.

    Parameters
    ----------
    min_cap         : minimum stable generation (MW) when committed
    fuel_type       : FuelType enum
    base_fuel_cost  : €/MWh at reference fuel price (used when no
                      stochastic fuel price is provided)
    heat_rate       : MWh_fuel / MWh_elec (efficiency inverse).
                      If > 0, actual fuel cost = heat_rate * fuel_price.
                      If 0, base_fuel_cost is used directly.
    startup_cost    : € per cold start
    min_run_hours   : minimum consecutive hours on after startup
    ramp_rate       : max MW change per hour (0 = no ramp limit)
    fixed_on        : if True, always committed (e.g. nuclear must-run)
    offer_price     : EUR/MWh market offer price.  Defaults to None, which
                      means the engine uses base_fuel_cost/heat_rate as the
                      offer.  Set negative for must-run nuclear / wind to
                      represent the avoided cold-start cost the operator bids
                      to stay on (e.g. FR nuclear: -50 EUR/MWh).
                      This is the price that sets the MCP in surplus hours.
    co2_intensity   : tCO2/MWh_elec — for emissions tracking (optional)
    """
    min_cap:             float    = 0.0
    fuel_type:           FuelType = FuelType.GAS
    base_fuel_cost:      float    = 50.0     # €/MWh
    heat_rate:           float    = 0.0      # if 0, use base_fuel_cost directly
    startup_cost:        float    = 0.0      # €
    min_run_hours:       int      = 1
    ramp_rate:           float    = 0.0      # MW/h; 0 = unlimited
    fixed_on:            bool     = False
    offer_price:         Optional[float] = None  # None → use fuel cost as offer
    co2_intensity:       float    = 0.0      # tCO2/MWh
    forced_outage_rate:  float    = 0.0      # fraction [0,1]
    provides_inertia:    bool     = True     # True for synchronous machines
    inertia_constant:    float    = 0.0      # H (seconds): kinetic energy / rated power
                                             # Nuclear/lignite: ~6-7s, CCGT: ~4-5s,
                                             # OCGT: ~2s, wind/solar: 0s
    startup_hours:       int      = 0        # hours to ramp from 0 to min_cap

    def marginal_cost(self, fuel_price: float) -> float:
        """€/MWh given a (possibly stochastic) fuel price."""
        if self.heat_rate > 0:
            return self.heat_rate * fuel_price
        return self.base_fuel_cost

    def effective_offer_price(self, fuel_price: float = 0.0) -> float:
        """Return the market offer price for MCP calculation.

        If offer_price is set explicitly (including negative values for
        must-run nuclear), return it directly.  Otherwise fall back to
        the marginal cost — correct for flexible thermal units.
        """
        if self.offer_price is not None:
            return self.offer_price
        return self.marginal_cost(fuel_price)


# ── Hydro plant ───────────────────────────────────────────────────────────────

@dataclass
class HydroPlant(Asset):
    """
    Dispatchable hydro with a reservoir.

    The reservoir is tracked in MWh. At each window boundary the
    CarryoverState holds `reservoir_mwh`. The inflow profile
    `inflow_mwh` (shape T) is added each hour.

    Parameters
    ----------
    min_cap             : minimum output when dispatched (MW)
    reservoir_capacity  : max reservoir size (MWh)
    initial_reservoir   : starting reservoir level (MWh)
    inflow_mwh          : hourly inflow (MWh). Shape (T,) or scalar.
    water_value         : opportunity cost €/MWh — used as marginal cost
                          so the merit order respects scarcity.
    startup_cost        : € per start (small for hydro)
    min_run_hours       : minimum run hours
    ramp_rate           : MW/h; 0 = unlimited (hydro is usually fast)
    """
    min_cap:            float  = 0.0
    reservoir_capacity: float  = 1000.0   # MWh
    initial_reservoir:  float  = 500.0    # MWh
    inflow_mwh:         float  = 10.0     # MWh/h scalar or array
    water_value:        float  = 30.0     # €/MWh base opportunity cost
    startup_cost:       float  = 0.0
    min_run_hours:      int    = 1
    ramp_rate:          float  = 0.0
    inertia_constant:   float  = 4.0      # H seconds — hydro turbines ~3-6s

    def inflow_array(self, T: int) -> np.ndarray:
        """Return inflow array of length T."""
        if np.isscalar(self.inflow_mwh):
            return np.full(T, float(self.inflow_mwh))
        arr = np.asarray(self.inflow_mwh, dtype=float)
        if len(arr) >= T:
            return arr[:T]
        reps = int(np.ceil(T / len(arr)))
        return np.tile(arr, reps)[:T]

    @property
    def marginal_cost(self) -> float:
        return self.water_value


# ── Variable renewable energy assets ─────────────────────────────────────────

@dataclass
class WindPlant(Asset):
    """
    Wind generation asset with time-varying availability profile.

    RES DISPATCH MODEL (Approach A — net-load pre-subtraction):
    ──────────────────────────────────────────────────────────────
    Wind is NOT a GA decision variable.  At each hour t the output is
    determined deterministically before the GA's thermal commitment loop:

        output[t] = max_cap × avail[t]   if MCP ≥ curtailment_threshold
        output[t] = 0                    otherwise (deep negative price)

    The GA then sees only *residual* (net) demand = gross demand − wind − solar.
    This is the correct market formulation: wind bids at (or below) zero and
    always dispatches unless the market price falls below the operator's
    curtailment threshold.

    curtailment_threshold (EUR/MWh):
        The minimum price at which this plant keeps dispatching.
        - Unsubsidised merchant wind: 0 EUR/MWh (curtail when MCP < 0)
        - CfD / FIT wind: negative value, e.g. -20 EUR/MWh.
          The plant keeps dispatching even at negative prices because
          the fixed strike / feed-in tariff payment exceeds the cost.
          In reality these plants stop at the "floor" set by their contract.
        - Default None: treated as 0 EUR/MWh (unsubsidised).

    offer_price (EUR/MWh):
        The price this plant sets when it is the marginal (last) unit
        in a surplus hour.  Equals curtailment_threshold in practice
        (a plant bidding -20 EUR/MWh sets MCP at -20 EUR/MWh when it
        is the cheapest curtailable unit).  If None, defaults to
        curtailment_threshold.

    No startup cost, no min_run, no ramp constraint.
    min_cap = 0 (can be fully curtailed).
    """
    min_cap:               float         = 0.0
    offer_price:           float         = 0.0      # EUR/MWh — MCP when marginal
    co2_intensity:         float         = 0.0
    curtailment_threshold: float         = 0.0      # EUR/MWh — stop dispatching below this

    @property
    def marginal_cost(self) -> float:
        return self.offer_price

    def effective_offer_price(self, fuel_price: float = 0.0) -> float:
        return self.offer_price

    def available_output(self, T: int) -> np.ndarray:
        """MW available at each hour = max_cap × avail[t].  Shape (T,)."""
        return self.max_cap * self.availability(T)


@dataclass
class SolarPlant(Asset):
    """
    Solar PV asset with time-varying availability (capacity factor) profile.

    Identical dispatch model to WindPlant — RES pre-subtraction, not a GA gene.
    avail[t] = ERA5 solar CF; zero at night, so solar output is automatically
    zero between sunset and sunrise.

    curtailment_threshold: same semantics as WindPlant.
        Solar in Europe is almost entirely subsidy-supported, but DA solar
        bids are typically 0 EUR/MWh (CfD plants receive strike price on top).
        Setting to 0 is correct for most European solar fleets.
    offer_price: EUR/MWh — MCP when solar is the marginal curtailable unit.
    """
    min_cap:               float         = 0.0
    offer_price:           float         = 0.0
    co2_intensity:         float         = 0.0
    curtailment_threshold: float         = 0.0      # EUR/MWh

    @property
    def marginal_cost(self) -> float:
        return self.offer_price

    def effective_offer_price(self, fuel_price: float = 0.0) -> float:
        return self.offer_price

    def available_output(self, T: int) -> np.ndarray:
        """MW available at each hour = max_cap × avail[t].  Shape (T,)."""
        return self.max_cap * self.availability(T)



@dataclass
class HeatPlant(Asset):
    """
    Combined heat and power (CHP) or heat-first plant.

    Produces both electricity (MW_e) and heat (MW_th).
    The ratio is fixed by `power_to_heat_ratio`.
    The plant may have a minimum heat obligation that forces it on
    during cold periods — model this via `must_run_heat_profile`.

    Parameters
    ----------
    min_cap             : minimum electrical output (MW)
    fuel_type           : fuel
    base_fuel_cost      : €/MWh_fuel
    heat_rate           : MWh_fuel / MWh_elec
    startup_cost        : € per start
    min_run_hours       : minimum hours on
    ramp_rate           : MW/h
    power_to_heat_ratio : MW_e / MW_th at full load
    heat_revenue        : €/MWh_th — offsets electricity cost
    must_run_heat       : shape (T,) MW_th obligation; 0 means no obligation
    """
    min_cap:             float    = 0.0
    fuel_type:           FuelType = FuelType.GAS
    base_fuel_cost:      float    = 50.0
    heat_rate:           float    = 0.0
    startup_cost:        float    = 0.0
    min_run_hours:       int      = 1
    ramp_rate:           float    = 0.0
    power_to_heat_ratio: float    = 1.0   # MW_e per MW_th
    heat_revenue:        float    = 30.0  # €/MWh_th
    must_run_heat:       Optional[np.ndarray] = field(default=None, repr=False)

    def effective_fuel_cost(self, fuel_price: float) -> float:
        """Net €/MWh_e after netting off heat revenue."""
        raw = (self.heat_rate * fuel_price) if self.heat_rate > 0 else self.base_fuel_cost
        heat_credit = self.heat_revenue / max(self.power_to_heat_ratio, 1e-6)
        return max(0.0, raw - heat_credit)

    def heat_obligation(self, T: int) -> np.ndarray:
        """MW_th that must be produced each hour."""
        if self.must_run_heat is None:
            return np.zeros(T)
        arr = np.asarray(self.must_run_heat, dtype=float)
        if len(arr) >= T:
            return arr[:T]
        reps = int(np.ceil(T / len(arr)))
        return np.tile(arr, reps)[:T]


# ── Storage / flexibility asset ───────────────────────────────────────────────

@dataclass
class StorageAsset(Asset):
    """
    Battery, pumped hydro, or aggregated flexibility prosumers.

    Storage can charge (consume) or discharge (produce).
    It is not subject to a binary commitment — it is always "available"
    and the simulator decides charge/discharge each hour.

    Parameters
    ----------
    energy_capacity     : MWh of stored energy at full charge
    charge_rate         : max MW of charging power
    discharge_rate      : max MW of discharge power
    charge_efficiency   : fraction retained when charging (0–1)
    discharge_efficiency: fraction retained when discharging (0–1)
    initial_soc         : starting state of charge (MWh)
    marginal_cost       : €/MWh for dispatch (wear/degradation cost)
    """
    energy_capacity:      float = 100.0    # MWh
    charge_rate:          float = 25.0     # MW
    discharge_rate:       float = 25.0     # MW
    charge_efficiency:    float = 0.92
    discharge_efficiency: float = 0.92
    initial_soc:          float = 50.0     # MWh
    marginal_cost:        float = 5.0      # €/MWh


# ── Fleet ─────────────────────────────────────────────────────────────────────

class Fleet:
    """
    Container for all assets in all regions.

    Usage:
        fleet = Fleet()
        fleet.add(ThermalPlant(...))
        fleet.add(StorageAsset(...))
        thermals = fleet.thermals("DE")
    """

    def __init__(self):
        self._assets: list[Asset] = []

    def add(self, asset: Asset) -> None:
        self._assets.append(asset)

    def all(self, region: Optional[str] = None) -> list[Asset]:
        if region is None:
            return list(self._assets)
        return [a for a in self._assets if a.region == region]

    def regions(self) -> list[str]:
        return sorted(set(a.region for a in self._assets))

    def thermals(self, region: Optional[str] = None) -> list[ThermalPlant]:
        return [a for a in self.all(region) if isinstance(a, ThermalPlant)]

    def hydros(self, region: Optional[str] = None) -> list[HydroPlant]:
        return [a for a in self.all(region) if isinstance(a, HydroPlant)]

    def heats(self, region: Optional[str] = None) -> list[HeatPlant]:
        return [a for a in self.all(region) if isinstance(a, HeatPlant)]

    def storages(self, region: Optional[str] = None) -> list[StorageAsset]:
        return [a for a in self.all(region) if isinstance(a, StorageAsset)]

    def res_units(self, region: Optional[str] = None) -> list:
        """Wind and solar plants."""
        return [a for a in self.all(region)
                if isinstance(a, (WindPlant, SolarPlant))]

    def dispatchable(self, region: Optional[str] = None) -> list[Asset]:
        """All assets with a dispatch variable (thermal + hydro + RES), excluding storage."""
        return [a for a in self.all(region)
                if not isinstance(a, StorageAsset)]

    def free_units(self, region: Optional[str] = None) -> list[Asset]:
        """Dispatchable units that are not fixed-on (GA controls dispatch fraction)."""
        return [a for a in self.dispatchable(region)
                if not getattr(a, "fixed_on", False)]

    def fixed_units(self, region: Optional[str] = None) -> list[Asset]:
        """Dispatchable units that are always on (nuclear must-run etc)."""
        return [a for a in self.dispatchable(region)
                if getattr(a, "fixed_on", False)]

    def __len__(self) -> int:
        return len(self._assets)

    def __repr__(self) -> str:
        r = f"Fleet({len(self._assets)} assets, regions={self.regions()})"
        return r
