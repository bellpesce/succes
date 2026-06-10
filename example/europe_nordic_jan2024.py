"""
examples/europe_nordic_jan2024.py  —  SUCCES January 2024 scenario
------------------------------------------------------------------
Full-month simulation: January 1–31 2024, 744 hours, 31 windows.
Same 15-region fleet as europe_nordic_v2.py (all fixes applied).

What is new vs v2
-----------------
Calendar-aware demand
    Each window knows its exact calendar day (Jan 1 = Monday).
    Demand is scaled by a day-specific temperature factor derived from
    actual January 2024 meteorological conditions in Central Europe:
      Week 1 (Jan 1-7):   mild, New Year holiday effect (0.80–0.97×)
      Week 2 (Jan 8-14):  cold snap, −5 to 0°C Germany (1.05–1.10×)
      Week 3 (Jan 15-21): moderate (1.00–1.03×)
      Week 4 (Jan 22-28): mild again (0.95–0.97×)
      Jan 29-31:          slight cooling (0.97–0.98×)
    Differential per region: continental block moves more than GB/Iberia.

Weekday / weekend shape distinction
    Six demand shapes: weekday vs weekend per shape family.
    Weekend factor applied to the exact calendar weekend days.
    Public holiday (Jan 1) treated as Sunday.

AR(1) demand noise
    DemandGenerator noise is now AR(1) with phi=0.60, same unconditional
    std as before (0.020). Consecutive hours within a scenario are
    correlated — a high-demand hour is more likely to be followed by
    another high-demand hour, matching real load autocorrelation.

AR(1) wind noise
    Shared wind block across regions is now AR(1) with phi=0.80.
    Wind episodes last multiple hours (realistic: wind ramps over 3-6h).
    Each scenario draws an independent wind trajectory.

Cross-regional cold-snap correlation
    A shared continental_factor is drawn per scenario: when the
    continental block demand is high (cold snap) all continental regions
    see it simultaneously. Nordic and Iberian blocks have lower loading
    on this factor, matching observed cross-regional demand correlation.

Multiple stress levels
    Three stress scenarios per region:
      1. Mild day  (−8% demand, 15% probability) — warm spells
      2. Cold snap (+10% demand, 5% probability) — deep frost
      3. Dark doldrum (+10% demand AND low wind, 3% probability)
    These compound: a scenario that draws a cold snap AND low wind
    represents the worst-case European power crunch.

Hydro seasonal state
    Norwegian and Swedish reservoirs start at 50–55% fill (typical
    early January — reservoir drawdown from autumn peak). Alpine
    (AT/CH) reservoirs at 35–40% (low point before spring snowmelt).
    Iberian (ES) at 55–60% (post-autumn rainfall).

Run parameters
--------------
TOTAL_HOURS  = 744   (31 days × 24h; set to 336 for quick 14-day test)
WINDOW_HOURS = 24
N_SCENARIOS  = 30
GA_EPOCHS    = 400
GA_POP_SIZE  = 80

Performance estimate
--------------------
31 windows × ~200s = ~100 min on 8 Julia threads.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from succes.assets import Fleet, ThermalPlant, HydroPlant, StorageAsset, FuelType, WindPlant, SolarPlant
from succes.network import Network, TransmissionLink
from succes.scenarios import (DemandGenerator, FuelPriceGenerator,
                                build_scenario_bank, HydroInflowGenerator)
from succes.solver import CoupledRollingHorizonSolver
from succes.julia_bridge import get_bridge
from succes import config

# ── ERA5 weather profiles ─────────────────────────────────────────────────────
# Loaded from data/era5_202401_profiles.npz, produced by data/era5_to_profiles.py.
# If the file does not exist, falls back to synthetic profiles automatically.
# To generate real ERA5 profiles:
#   1. Register at cds.climate.copernicus.eu and configure ~/.cdsapirc
#   2. python data/era5_fetch.py
#   3. python data/era5_to_profiles.py
_ERA5_PATH = ROOT / "data" / "era5_202401_profiles.npz"
_ERA5_SYNTHETIC_FLAG = False

def _load_era5_profiles() -> dict:
    """
    Load ERA5 profiles from disk, or generate synthetic fallback.
    Returns dict with keys: wind_mw, solar_mw, temp_scale — all shape (T, R)
    where T=744, R=15 and regions are sorted alphabetically.
    """
    global _ERA5_SYNTHETIC_FLAG
    if _ERA5_PATH.exists():
        data = np.load(str(_ERA5_PATH), allow_pickle=False)
        file_regions = list(data["regions"].astype(str))
        print(f"  ERA5 profiles loaded: {_ERA5_PATH.name}  "
              f"({int(data['hours'])}h × {len(file_regions)} regions)")
        return {
            "wind_mw":    data["wind_mw"],    # (T, R)
            "solar_mw":   data["solar_mw"],   # (T, R)
            "temp_scale": data["temp_scale"],  # (T, R)
            "regions":    file_regions,
        }
    else:
        _ERA5_SYNTHETIC_FLAG = True
        print(f"  ERA5 file not found: {_ERA5_PATH.name}")
        print(f"  → Using synthetic profiles (run data/era5_fetch.py for real data)")
        # Import the synthetic generator from era5_to_profiles
        sys.path.insert(0, str(ROOT / "data"))
        from era5_to_profiles import (make_synthetic_profiles,
                                      REGIONS as ERA5_REGIONS)
        wc, wm, sc, sm, ts = make_synthetic_profiles(2024, 1)
        return {
            "wind_mw":    wm,
            "solar_mw":   sm,
            "temp_scale": ts,
            "regions":    ERA5_REGIONS,
        }

_ERA5 = _load_era5_profiles()

def _era5_wind_mw(region: str, hour_start: int, T: int) -> np.ndarray:
    """Return wind_mw[hour_start:hour_start+T] for this region."""
    if region not in _ERA5["regions"]:
        return np.zeros(T)
    ri = _ERA5["regions"].index(region)
    profile = _ERA5["wind_mw"][:, ri]
    # Tile if needed (month run: 744h; profile should be 744h exactly)
    if hour_start + T <= len(profile):
        return profile[hour_start : hour_start + T]
    # Fallback: tile
    tiled = np.tile(profile, int(np.ceil((hour_start + T) / len(profile))))
    return tiled[hour_start : hour_start + T]

def _era5_solar_mw(region: str, hour_start: int, T: int) -> np.ndarray:
    if region not in _ERA5["regions"]:
        return np.zeros(T)
    ri = _ERA5["regions"].index(region)
    profile = _ERA5["solar_mw"][:, ri]
    if hour_start + T <= len(profile):
        return profile[hour_start : hour_start + T]
    tiled = np.tile(profile, int(np.ceil((hour_start + T) / len(profile))))
    return tiled[hour_start : hour_start + T]

def _era5_temp_scale(region: str, hour_start: int, T: int) -> np.ndarray:
    if region not in _ERA5["regions"]:
        return np.ones(T)
    ri = _ERA5["regions"].index(region)
    profile = _ERA5["temp_scale"][:, ri]
    if hour_start + T <= len(profile):
        return profile[hour_start : hour_start + T]
    tiled = np.tile(profile, int(np.ceil((hour_start + T) / len(profile))))
    return tiled[hour_start : hour_start + T]

# ── Run parameters ────────────────────────────────────────────────────────────
TOTAL_HOURS  = 336    # set to 744 for full January
WINDOW_HOURS = 24
N_SCENARIOS  = 15    # 30 doubled cost with same epochs → worse quality; 15 optimal for prototype
GA_EPOCHS    = 350   # hard ceiling; adaptive stopping (patience=80, threshold=0.02%) exits early
GA_POP_SIZE  = 80

# ── Calendar: January 2024 ────────────────────────────────────────────────────
# Jan 1 2024 = Monday (weekday 0).  window_idx 0..30 → Jan 1..31.
# Used to select correct demand shape and temperature scaling.
import datetime as _dt
_JAN1_2024    = _dt.date(2024, 1, 1)
_JAN_WEEKDAYS = [(_JAN1_2024 + _dt.timedelta(days=d)).weekday() for d in range(31)]
# weekday(): 0=Mon … 4=Fri, 5=Sat, 6=Sun
IS_WEEKEND    = [wd >= 5 for wd in _JAN_WEEKDAYS]  # True for Sat/Sun
IS_HOLIDAY    = [True] + [False] * 30               # Jan 1 treated as Sunday

# ── Regions ───────────────────────────────────────────────────────────────────
CWE_REGIONS    = ["AT", "BE", "CH", "CZ", "DE", "DK", "FR", "NL", "PL"]
NORDIC_REGIONS = ["GB", "NO", "SE"]
NEW_REGIONS    = ["ES", "HU", "IT"]
REGIONS        = CWE_REGIONS + NORDIC_REGIONS + NEW_REGIONS

# ── Gross demand peaks (MW) ───────────────────────────────────────────────────
# GROSS demand = total consumption before any renewable subtraction.
# Wind and solar are now explicit dispatchable assets in the chromosome.
# The GA dispatches them and the MCP is set by the last unit dispatched.
# Source: ENTSO-E actual Jan 2024 peaks / SOAF 2023 estimates.
DEMAND_PEAKS = {
    "DE": 60_000, "FR": 54_000, "PL": 22_000, "NL": 14_000,
    "BE":  9_500, "CZ":  9_500, "AT":  9_000, "CH":  8_500, "DK":  4_200,
    "NO": 22_000, "SE": 25_000, "GB": 42_000,
    "IT": 50_000, "ES": 38_000, "HU":  6_000,
}

# ── Wind and solar offer prices (EUR/MWh) ─────────────────────────────────────
# Wind/solar bid at or near zero — no fuel cost.
# When wind is the last dispatched unit → MCP = wind offer price.
# Slightly negative for high-subsidy regions (legacy FIT plants).
WIND_OFFER_PRICE = {
    "DE":  0.0, "DK": -5.0, "GB": -3.0, "NL":  0.0, "FR":  0.0,
    "BE":  0.0, "ES":  0.0, "SE":  0.0, "NO":  0.0, "PL":  0.0,
    "IT":  0.0, "AT":  0.0, "CH":  0.0, "CZ":  0.0, "HU":  0.0,
}
SOLAR_OFFER_PRICE = {r: 0.0 for r in ["DE","FR","GB","BE","NL","DK","AT","CH",
                                        "CZ","PL","NO","SE","IT","ES","HU"]}

# ── Installed RES capacity (MW, Jan 2024) ─────────────────────────────────────
WIND_INSTALLED = {
    # Total installed = offshore + inland (see era5_to_profiles.py for the split).
    # era5_to_profiles.py blends two ERA5 points per region weighted by their
    # respective installed capacities, giving a fleet-weighted average CF.
    # Source: WindEurope 2024, ENTSO-E SOAF 2023.
    "DE": 68_000,   # 8 GW offshore (Bight) + 60 GW inland
    "DK":  6_500,   # 2.8 GW offshore + 3.7 GW inland
    "GB": 28_000,   # 14.5 GW offshore (Irish Sea) + 13.5 GW inland
    "NL":  6_500,   # 3.5 GW offshore + 3.0 GW inland
    "BE":  5_500,   # 2.3 GW offshore + 3.2 GW inland
    "FR": 22_000,   # 1.5 GW offshore + 20.5 GW inland
    "NO":  4_000,   # 0.3 GW offshore + 3.7 GW inland
    "SE": 12_000,   # 3.6 GW nearshore + 8.4 GW inland
    "ES": 30_000,   # 10 GW Atlantic coast + 20 GW inland (Castilla)
    "PL":  8_000,   "IT": 12_000,   "AT":  3_500,
    "CH":    100,   "CZ":    350,   "HU":    250,
}

SOLAR_INSTALLED = {
    "DE": 81_000, "IT": 24_000, "ES": 20_000, "FR": 19_000,
    "NL":  9_000, "GB": 15_000, "PL":  9_000, "BE":  7_500,
    "CZ":  3_000, "AT":  4_000, "CH":  4_000, "SE":  2_000,
    "DK":  3_500, "NO":    100, "HU":  4_500,
}

VALLEY_RATIOS = {
    # Overnight minimum as a fraction of GROSS demand peak.
    # When we moved from residual to gross demand these were never updated.
    # Old values were calibrated for residual demand (gross minus typical RES),
    # which has a lower overnight floor than gross demand.
    #
    # Correct values derived from ENTSO-E actual Jan 2024 hourly gross demand:
    #   FR: overnight min ~29 GW / 52 GW gross peak = 0.56
    #   BE: overnight min ~4.5 GW / 9.5 GW = 0.47
    #   GB: overnight min ~19 GW / 42 GW = 0.45
    #   DE: overnight min ~24 GW / 60 GW = 0.40
    #   etc.
    #
    # Raising these values reduces nuclear surplus hours in FR/BE/CH/GB
    # because overnight gross demand is now higher relative to fixed nuclear.
    # This is NOT a quick fix — it is the correct calibration for the gross
    # demand model. The old values caused FR overnight = 18.7 GW vs 38 GW
    # nuclear → guaranteed surplus every night. Real FR overnight = ~29 GW.
    "DE": 0.40, "FR": 0.56, "PL": 0.45, "NL": 0.45, "BE": 0.58,
    # BE: nuclear=5,800 MW, gross_peak=9,500 MW, real overnight min ~5,500 MW
    # valley_ratio = 5,500 / 9,500 = 0.58. Previous 0.47 gave overnight min
    # of 4,465 MW → 1,335 MW nuclear surplus every night → 86 negative hours.
    # At 0.58: overnight min = 5,510 MW → nuclear surplus only ~290 MW
    # → ~15-20 negative hours (physically correct for BE in Jan 2024)
    "CZ": 0.46, "AT": 0.45, "CH": 0.50, "DK": 0.38,
    "NO": 0.55, "SE": 0.50, "GB": 0.45,
    "IT": 0.44, "ES": 0.40, "HU": 0.47,
}

DEMAND_SHAPES = {
    "DE": "continental", "FR": "continental", "PL": "continental",
    "NL": "continental", "BE": "continental", "CZ": "continental",
    "AT": "continental", "CH": "continental", "DK": "danish",
    "NO": "nordic",      "SE": "nordic",      "GB": "british",
    "IT": "continental", "ES": "iberian",     "HU": "continental",
}

# ── January 2024 temperature-driven demand scaling ────────────────────────────
# Day-specific multiplier on base peak demand.  Based on observed temperature
# anomalies for Central Europe Jan 2024:
#   Week 1: anomalously mild (+3–5°C above seasonal norm) → demand below base
#   Week 2: cold snap (−3 to −5°C below norm) → demand above base
#   Week 3: return to mild
#   Week 4: mild, below-average demand
#
# The multiplier is applied to the peak AND valley proportionally so the
# whole profile shifts — not just the peak.
TEMP_SCALE_JAN2024 = [
    # Day:   1     2     3     4     5     6     7
    #        Mon   Tue   Wed   Thu   Fri   Sat   Sun
    # Week 1: mild + New Year public holiday (Jan 1 = national holiday)
           0.88, 0.91, 0.93, 0.95, 0.97, 0.83, 0.80,
    # Week 2: cold snap — peak week of January
    # Jan 8-12 working days see highest demand of the month
           1.05, 1.08, 1.10, 1.10, 1.08, 0.93, 0.91,
    # Week 3: moderate, return to mild
           1.02, 1.03, 1.03, 1.02, 1.00, 0.88, 0.86,
    # Week 4: below average, mild European winter continues
           0.96, 0.97, 0.97, 0.96, 0.95, 0.83, 0.82,
    # Days 29-31: slight cooling, working week
           0.98, 0.99, 0.97,
]

# Regional sensitivity to the temperature scaling.
# Continental block moves most with temperature. GB slightly less.
# Iberia (ES) and NO/SE have lower sensitivity (different climate regime).
TEMP_SENSITIVITY = {
    "DE": 1.00, "FR": 0.95, "PL": 1.00, "CZ": 0.98, "AT": 0.96,
    "CH": 0.94, "BE": 0.97, "NL": 0.98, "HU": 0.99,
    "DK": 0.92, "SE": 0.80, "NO": 0.75,
    "GB": 0.88, "IT": 0.85, "ES": 0.60,
}

def _temp_scale(window_idx: int, region: str) -> float:
    """Return demand multiplier for this calendar day and region."""
    day = min(window_idx, 30)          # clamp to Jan 1–31
    base_scale = TEMP_SCALE_JAN2024[day]
    # Apply regional sensitivity: how much does this region respond to temp?
    sens = TEMP_SENSITIVITY.get(region, 0.90)
    # Interpolate: at sens=1.0 use full scale; at sens=0 no variation (=1.0)
    return 1.0 + (base_scale - 1.0) * sens


def _residual_profile(peak, valley_ratio, shape="continental"):
    """24h residual demand shape, normalised to peak=1.0."""
    if shape == "continental":
        raw = [0.35,0.27,0.21,0.18,0.16,0.20, 0.40,0.65,0.80,0.84,
               0.86,0.88,0.85,0.81,0.84,0.89, 0.93,0.97,1.00,0.97,
               0.90,0.77,0.59,0.44]
    elif shape == "nordic":
        raw = [0.58,0.53,0.50,0.48,0.48,0.51, 0.57,0.69,0.78,0.82,
               0.85,0.87,0.86,0.87,0.88,0.91, 0.94,0.97,1.00,0.98,
               0.93,0.88,0.77,0.67]
    elif shape == "british":
        raw = [0.40,0.31,0.25,0.21,0.19,0.24, 0.44,0.74,0.89,0.91,
               0.86,0.81,0.75,0.74,0.77,0.83, 0.91,0.98,1.00,0.97,
               0.88,0.74,0.58,0.46]
    elif shape == "danish":
        raw = [0.30,0.22,0.17,0.13,0.12,0.16, 0.36,0.68,0.82,0.80,
               0.76,0.73,0.68,0.70,0.74,0.80, 0.88,0.96,1.00,0.95,
               0.83,0.65,0.47,0.34]
    elif shape == "iberian":
        raw = [0.38,0.30,0.24,0.20,0.18,0.21, 0.38,0.58,0.72,0.80,
               0.85,0.88,0.84,0.80,0.82,0.86, 0.90,0.95,1.00,0.98,
               0.92,0.81,0.65,0.50]
    else:
        raw = [0.35,0.27,0.21,0.18,0.16,0.20, 0.40,0.65,0.80,0.84,
               0.86,0.88,0.85,0.81,0.84,0.89, 0.93,0.97,1.00,0.97,
               0.90,0.77,0.59,0.44]
    p = np.array(raw, dtype=float)
    mn, mx = p.min(), p.max()
    p = (p - mn) / (mx - mn)
    p = valley_ratio + (1.0 - valley_ratio) * p
    return peak * p


def _weekend_profile(base_profile: np.ndarray, region: str) -> np.ndarray:
    """Scale weekday profile to weekend shape."""
    valley      = float(base_profile.min())
    wke_factor  = 0.85 if region in ("NO", "SE") else 0.78
    return valley + (base_profile - valley) * wke_factor


# Base weekday profiles (unscaled by temperature — scaling applied per window)
BASE_PROFILES = {
    r: _residual_profile(DEMAND_PEAKS[r], VALLEY_RATIOS[r], DEMAND_SHAPES[r])
    for r in REGIONS
}
RESIDUAL_PROFILES = {r: BASE_PROFILES[r].copy() for r in REGIONS}


# ── Forced outage rates ───────────────────────────────────────────────────────
FOR_BY_FUEL = {
    "nuclear": 0.020, "lignite": 0.055, "coal":    0.045,
    "gas_ccgt": 0.025, "gas_ocgt": 0.030, "hydro":  0.015,
    "biomass":  0.050, "oil":     0.035,
}

def _fuel_for(fuel_type_str, unit_name):
    n = unit_name.lower()
    if "nuclear" in n: return FOR_BY_FUEL["nuclear"]
    if "lignite" in n: return FOR_BY_FUEL["lignite"]
    if "biomass" in n: return FOR_BY_FUEL["biomass"]
    if "oil"     in n: return FOR_BY_FUEL["oil"]
    if "ocgt"    in n: return FOR_BY_FUEL["gas_ocgt"]
    if "ccgt"    in n or "gas" in n: return FOR_BY_FUEL["gas_ccgt"]
    if "coal"    in n: return FOR_BY_FUEL["coal"]
    if "hydro"   in n or "ror" in n or "reservoir" in n: return FOR_BY_FUEL["hydro"]
    return 0.03

CO2_BY_FUEL = {
    "lignite":  1.02, "coal": 0.82, "gas_ccgt": 0.38,
    "gas_ocgt": 0.58, "nuclear": 0.012, "hydro": 0.004,
    "biomass": 0.230, "oil": 0.65,
}

def _co2_intensity(unit_name):
    n = unit_name.lower()
    if "nuclear" in n: return CO2_BY_FUEL["nuclear"]
    if "lignite" in n: return CO2_BY_FUEL["lignite"]
    if "biomass" in n: return CO2_BY_FUEL["biomass"]
    if "oil"     in n: return CO2_BY_FUEL["oil"]
    if "ocgt"    in n: return CO2_BY_FUEL["gas_ocgt"]
    if "ccgt"    in n: return CO2_BY_FUEL["gas_ccgt"]
    if "coal"    in n: return CO2_BY_FUEL["coal"]
    if "hydro"   in n or "ror" in n: return CO2_BY_FUEL["hydro"]
    return 0.50

def _provides_inertia(fuel_type, unit_name):
    n = unit_name.lower()
    return "ocgt" not in n and "battery" not in n


# ── Plant factory helpers ─────────────────────────────────────────────────────

# ── Inertia constants (H seconds) by fuel type ───────────────────────────────
# Source: IEEE Std 2800-2022, CIGRE TB 727, ENTSO-E inertia studies
# H = stored kinetic energy (MJ) / rated power (MVA)
# Large steam turbines (nuclear, lignite, coal) have the highest H.
# Aeroderivative OCGT has the lowest — very light turbine.
_INERTIA_H = {
    "nuclear":  6.5,
    "lignite":  6.0,
    "coal":     5.5,
    "biomass":  4.5,
    "ccgt":     4.5,   # combined cycle — moderate steam section
    "gas":      4.5,   # generic gas → assume CCGT
    "ocgt":     2.0,   # aeroderivative, light rotor
    "oil":      3.5,
    "hydro":    4.0,   # wide range 2-9s; 4s is a conservative mean
    "ror":      3.5,   # run-of-river turbines tend to be smaller
    "wind":     0.0,   # inverter-connected
    "solar":    0.0,
    "storage":  0.0,
    "battery":  0.0,
    "other":    3.0,
}

def _inertia_h(fuel_type, name: str) -> float:
    """Return H (seconds) for a unit given its fuel type and name."""
    n = name.lower()
    # Name-based overrides for specific technology sub-types
    if "ocgt" in n or "np_" in n:  return _INERTIA_H["ocgt"]
    if "ccgt" in n:                return _INERTIA_H["ccgt"]
    if "lignite" in n:             return _INERTIA_H["lignite"]
    if "nuclear" in n:             return _INERTIA_H["nuclear"]
    if "coal" in n:                return _INERTIA_H["coal"]
    if "hydro" in n or "hy_" in n: return _INERTIA_H["hydro"]
    if "wind" in n or "solar" in n: return 0.0
    if "battery" in n or "storage" in n: return 0.0
    # Fall back to fuel type
    ft = str(fuel_type).lower().replace("fueltype.", "")
    return _INERTIA_H.get(ft, 3.0)

# ── Minimum system inertia (H_min seconds) by region ─────────────────────────
# Required: sum(H_i * dispatch_i[t]) >= H_min * demand[t]
# Island systems (GB) are stricter than large synchronous areas (Continental).
H_MIN_SECONDS = {
    "GB": 5.0,   # islanded, strict RoCoF requirements (National Grid ESO)
    "IE": 5.0,   # also islanded
    "NO": 4.5,   # Nordic synchronous area, high hydro → natural inertia
    "SE": 4.5,
    "DK": 4.0,   # connects Continental and Nordic
    "IT": 4.0,   # semi-islanded peninsula
    "ES": 4.0,   # Iberian peninsula, limited French interconnection
    # Continental synchronous area (large coupled system → lower H_min)
    "DE": 3.5, "FR": 3.5, "NL": 3.5, "BE": 3.5,
    "AT": 3.5, "CH": 3.5, "CZ": 3.5, "PL": 3.5,
    "HU": 3.5,
}


def _th(name, region, max_cap, min_cap, fuel_type, base_fuel_cost, heat_rate,
        startup_cost=0, min_run_hours=1, ramp_rate=0, fixed_on=False,
        offer_price=None):
    return ThermalPlant(
        name=name, region=region, max_cap=max_cap, min_cap=min_cap,
        fuel_type=fuel_type, base_fuel_cost=base_fuel_cost, heat_rate=heat_rate,
        startup_cost=startup_cost, min_run_hours=min_run_hours,
        ramp_rate=ramp_rate, fixed_on=fixed_on,
        offer_price=offer_price,
        forced_outage_rate=_fuel_for(str(fuel_type), name),
        provides_inertia=_provides_inertia(fuel_type, name),
        inertia_constant=_inertia_h(fuel_type, name),
        startup_hours=0,
        co2_intensity=_co2_intensity(name))

def _hy(name, region, max_cap, min_cap, reservoir_capacity, initial_reservoir,
        inflow_mwh, water_value, startup_cost=0, min_run_hours=1, ramp_rate=0):
    return HydroPlant(
        name=name, region=region, max_cap=max_cap, min_cap=min_cap,
        reservoir_capacity=reservoir_capacity, initial_reservoir=initial_reservoir,
        inflow_mwh=inflow_mwh, water_value=water_value,
        startup_cost=startup_cost, min_run_hours=min_run_hours, ramp_rate=ramp_rate,
        inertia_constant=_INERTIA_H["hydro"])

def _st(name, region, max_cap, energy_capacity, charge_rate, discharge_rate,
        charge_eff, discharge_eff, initial_soc, marginal_cost):
    return StorageAsset(
        name=name, region=region, max_cap=max_cap,
        energy_capacity=energy_capacity, charge_rate=charge_rate,
        discharge_rate=discharge_rate, charge_efficiency=charge_eff,
        discharge_efficiency=discharge_eff,
        initial_soc=initial_soc, marginal_cost=marginal_cost)


# ══════════════════════════════════════════════════════════════════════════════
# FLEET DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

def build_fleet() -> Fleet:
    """
    Fleet v2 — 16 regions, plant-level DE fleet, IT/ES/HU added.

    CO2 price: 82 EUR/tCO2 (ETS 2024 approximate)

    MC calibration:
      Lignite (BoA):  bf*2.55 + 82*1.02 = bf*2.55 + 83.6  → BoA at 94-100
      Lignite (old):  bf*2.80 + 82*1.02 = bf*2.80 + 83.6  → old at 97-103
      Hard coal T1:   bf*2.70 + 82*0.82 = bf*2.70 + 67.2  → T1 ≈ 101
      CCGT T1:        bf*1.85 + 82*0.38 = bf*1.85 + 31.2  → T1 ≈ 89
      OCGT T1:        bf*2.50 + 82*0.58 = bf*2.50 + 47.6  → T1 ≈ 130
    """
    fleet = Fleet()
    # Attach H_min requirements so julia_bridge can serialise them per region
    fleet._h_min = H_MIN_SECONDS
    GAS, COAL, NUC = FuelType.GAS, FuelType.COAL, FuelType.NUCLEAR

    # ── Shared tier parameter tables ──────────────────────────────────────────

    CCGT_TIERS = [
        # (base_fuel, heat_rate, startup_cost, min_run_hours, label)
        # MC = bf*hr + 82*0.38
        (31.2, 1.85, 12_000, 2, "T1"),   # MC≈89  modern efficient
        (35.0, 1.85, 18_000, 3, "T2"),   # MC≈96  standard
        (39.9, 1.90, 22_000, 4, "T3"),   # MC≈107 older
        (45.3, 1.95, 28_000, 2, "T4"),   # MC≈119 marginal
        (49.5, 2.00, 32_000, 2, "T5"),   # MC≈130 older/marginal — used for capacity fixes
        (54.0, 2.05, 36_000, 2, "T6"),   # MC≈142 very old plant kept for adequacy only
    ]

    COAL_TIERS = [
        # (base_fuel, heat_rate, startup_cost, min_run_hours, co2_int, label)
        # MC = bf*hr + 82*co2_int
        (12.5, 2.70, 60_000,  8, 0.82, "T1"),  # MC≈101 efficient
        (13.5, 2.85, 80_000, 10, 0.85, "T2"),  # MC≈108 standard
        (14.5, 3.00, 95_000, 12, 0.92, "T3"),  # MC≈120 old
        (16.0, 3.10, 110_000,14, 0.95, "T4"),  # MC≈132 very old — adequacy only
    ]

    OCGT_TIERS = [
        # (base_fuel, heat_rate, startup_cost, label)
        # MC = bf*hr + 82*0.58
        (33.0, 2.5, 4_000, "T1"),   # MC≈130
        (36.2, 2.5, 6_000, "T2"),   # MC≈138
        (40.2, 2.5, 8_000, "T3"),   # MC≈148
        (44.5, 2.5, 9_000, "T4"),   # MC≈159  extra-capacity blocks only
    ]

    NP_TIERS = [
        # (base_fuel, startup_cost, label)
        # MC = bf*2.6 + 82*0.58
        (42.5, 10_000, "T1"),  # MC≈158
        (47.1, 12_000, "T2"),  # MC≈170
    ]

    def _add_ccgt_tier(region, tier_idx, n_units, cap_each):
        bf, hr, sc, mr, lbl = CCGT_TIERS[tier_idx]
        min_c = max(1, int(cap_each * 0.15))
        # ramp_rate: modern CCGT ramps ~20% of nameplate per hour (fast-response gas)
        # This ensures ramp_rate * min_hours_for_restart >= min_cap → cyclable
        rr = max(1, int(cap_each * 0.20))
        for i in range(n_units):
            fleet.add(_th(f"{region}_CCGT_{lbl}_{i+1:02d}", region, cap_each,
                min_c, GAS, bf + i*0.1, hr,
                startup_cost=sc, min_run_hours=mr, ramp_rate=rr))

    def _add_coal_tier(region, tier_idx, n_units, cap_each):
        bf, hr, sc, mr, co2i, lbl = COAL_TIERS[tier_idx]
        min_c = max(1, int(cap_each * 0.25))
        # ramp_rate: hard coal ~15% of nameplate per hour (slower than gas)
        rr = max(1, int(cap_each * 0.15))
        for i in range(n_units):
            fleet.add(_th(f"{region}_Coal_{lbl}_{i+1:02d}", region, cap_each,
                min_c, COAL, bf + i*0.2, hr,
                startup_cost=sc + i*2_000, min_run_hours=mr, ramp_rate=rr))

    def _add_ocgt_tier(region, tier_idx, n_units, cap_each):
        bf, hr, sc, lbl = OCGT_TIERS[tier_idx]
        for i in range(n_units):
            fleet.add(_th(f"{region}_OCGT_{lbl}_{i+1:02d}", region, cap_each,
                0, GAS, bf + i*0.15, hr,
                startup_cost=sc + i*200, min_run_hours=1))

    def _add_np_tier(region, tier_idx, n_units, cap_each):
        bf, sc, lbl = NP_TIERS[tier_idx]
        for i in range(n_units):
            fleet.add(_th(f"{region}_NP_{lbl}_{i+1:02d}", region, cap_each,
                0, GAS, bf + i*0.2, 2.6,
                startup_cost=sc + i*500, min_run_hours=1))


    # ══════════════════════════════════════════════════════════════════════════
    # DE — Germany
    # ══════════════════════════════════════════════════════════════════════════
    # Residual peak: 45,000 MW.  Target: ~90,000 MW (2.0x)
    #
    # LIGNITE — named plants, approximate post-2022 operating capacities.
    # Real units are 300-1100 MW blocks; each block modelled individually.
    # Heat rates and MCs calibrated to ENTSO-E SFOP data.
    #
    # min_cap=0 for lignite (they can load-follow to ~30% in reality,
    # but we let the GA decide dispatch fraction — the high startup cost
    # and min_run_hours create the correct inflexibility).
    # ──────────────────────────────────────────────────────────────────────────

    # ── RWE: Neurath (BoA = Braunkohlenkraftwerk mit optimierter Anlagentechnik)
    # BoA units are the newest, most efficient DE lignite (~37% net efficiency).
    # BoA1 (2003) 1,100 MW,  BoA2 (2012) 1,100 MW,  BoA3 (2012) 1,100 MW
    # Plus older Neurath F block 600 MW (higher HR, scheduled for phaseout)
    #
    # Startup cost recalibration (Fix 4):
    # Previous values (120–180k EUR) were "cold start from ambient" — realistic for
    # a unit that has been off for 3+ days, but the rolling horizon model sees units
    # going on/off daily. The correct cost for a "warm start" (8–48h off) is ~50–70%
    # of cold-start cost. Source: ENTSO-E ERAA 2023 cost data, BNetzA cost survey.
    #   BoA modern (high-efficiency, fast ramp): hot/warm start ~45,000 EUR
    #   Standard lignite 600-800 MW:             warm start ~60,000 EUR
    #   Older/smaller blocks:                    warm start ~55,000 EUR
    # This makes overnight lignite commitment genuinely economic in the 48h horizon.
    for name, cap, bf, hr in [
        ("DE_Lignite_Neurath_BoA1", 1100, 4.20, 2.55),
        ("DE_Lignite_Neurath_BoA2", 1100, 4.30, 2.55),
        ("DE_Lignite_Neurath_BoA3", 1100, 4.40, 2.55),
        ("DE_Lignite_Neurath_F",     600, 5.00, 2.80),
    ]:
        sc = 45_000 if "BoA" in name else 55_000
        fleet.add(_th(name, "DE", cap, 0, COAL, bf, hr,
            startup_cost=sc, min_run_hours=16, ramp_rate=int(cap*0.12)))

    # ── RWE: Niederaussem
    # Block K (BoA+, 2003): 1,000 MW,  Blocks G/H/F: 640-650 MW older units
    for name, cap, bf, hr in [
        ("DE_Lignite_Niederauss_K",  1000, 4.80, 2.60),
        ("DE_Lignite_Niederauss_H",   650, 5.20, 2.80),
        ("DE_Lignite_Niederauss_G",   640, 5.30, 2.82),
        ("DE_Lignite_Niederauss_F",   620, 5.40, 2.85),
    ]:
        sc = 50_000 if name.endswith("_K") else 58_000
        fleet.add(_th(name, "DE", cap, 0, COAL, bf, hr,
            startup_cost=sc, min_run_hours=16, ramp_rate=int(cap*0.12)))

    # ── RWE: Weisweiler
    # Two blocks E/F each ~640 MW.  Higher heat rate — older vintage.
    for name, cap, bf, hr in [
        ("DE_Lignite_Weisweiler_E", 640, 5.50, 2.88),
        ("DE_Lignite_Weisweiler_F", 640, 5.60, 2.88),
    ]:
        fleet.add(_th(name, "DE", cap, 0, COAL, bf, hr,
            startup_cost=60_000, min_run_hours=16, ramp_rate=int(cap*0.10)))

    # ── LEAG: Boxberg (Lusatia)
    # Block R (2012, 675 MW BoA),  Blocks P/Q older 900 MW units
    for name, cap, bf, hr in [
        ("DE_Lignite_Boxberg_R",  900, 4.50, 2.60),
        ("DE_Lignite_Boxberg_Q",  900, 4.60, 2.62),
        ("DE_Lignite_Boxberg_P",  700, 5.10, 2.78),
    ]:
        sc = 48_000 if name.endswith("_R") else 62_000
        fleet.add(_th(name, "DE", cap, 0, COAL, bf, hr,
            startup_cost=sc, min_run_hours=18, ramp_rate=int(cap*0.11)))

    # ── LEAG: Schwarze Pumpe
    # Two identical 800 MW blocks (A/B), commissioned 1997/98.
    for name, cap, bf, hr in [
        ("DE_Lignite_SchwarzePumpe_A", 800, 4.70, 2.65),
        ("DE_Lignite_SchwarzePumpe_B", 800, 4.80, 2.68),
    ]:
        fleet.add(_th(name, "DE", cap, 0, COAL, bf, hr,
            startup_cost=60_000, min_run_hours=18, ramp_rate=int(cap*0.11)))

    # ── LEAG: Jänschwalde
    # Six blocks 500 MW each (Brandenburg). High CO2, older design.
    # Three blocks in reserve/standby — model as T3 here with higher HR.
    for name, cap, bf, hr in [
        ("DE_Lignite_Jaenschwalde_A", 500, 5.30, 2.82),
        ("DE_Lignite_Jaenschwalde_B", 500, 5.40, 2.85),
        ("DE_Lignite_Jaenschwalde_C", 500, 5.50, 2.88),
    ]:
        fleet.add(_th(name, "DE", cap, 0, COAL, bf, hr,
            startup_cost=55_000, min_run_hours=18, ramp_rate=int(cap*0.10)))

    # ── DE Hard coal — named plants
    # Datteln 4 (Uniper, 2020): newest and most efficient hard coal in DE, 1100 MW
    # Heyden (Uniper, NRW): 875 MW, ~37% efficiency
    # Staudinger 5 (EnBW): 513 MW
    # Wilhelmshaven (Uniper): 760 MW
    # Rostock (Stadtwerke Rostock): 500 MW
    # Mehrum (Preussen Elektra): 690 MW
    # Lünen (STEAG): 700 MW
    for name, cap, bf, hr, sc in [
        ("DE_Coal_Datteln4",        1100, 11.5, 2.60, 55_000),  # MC≈96 (newest)
        ("DE_Coal_Heyden",           875, 12.0, 2.68, 60_000),  # MC≈98
        ("DE_Coal_Staudinger5",      513, 12.5, 2.72, 62_000),  # MC≈101
        ("DE_Coal_Wilhelmshaven",    760, 13.0, 2.80, 68_000),  # MC≈104
        ("DE_Coal_Rostock",          500, 13.2, 2.82, 70_000),  # MC≈104
        ("DE_Coal_Mehrum",           690, 13.5, 2.85, 75_000),  # MC≈106
        ("DE_Coal_Luenen",           700, 14.0, 2.90, 80_000),  # MC≈108
    ]:
        min_c = max(1, int(cap * 0.25))
        fleet.add(_th(name, "DE", cap, min_c, COAL, bf, hr,
            startup_cost=sc, min_run_hours=8, ramp_rate=int(cap*0.15)))

    # ── DE CCGT: 4 tiers × 12 units × 600 MW = 28,800 MW
    # Germany has ~25 GW of CCGT (Irsching, Marbach, Düsseldorf, Hamm etc).
    # Model as anonymous tiers — individual units don't affect credibility here.
    _add_ccgt_tier("DE", 0, 12, 600)
    _add_ccgt_tier("DE", 1, 12, 600)
    _add_ccgt_tier("DE", 2, 12, 600)
    _add_ccgt_tier("DE", 3, 12, 600)

    # ── DE OCGT: 3 tiers × 20 units × 450 MW = 27,000 MW
    _add_ocgt_tier("DE", 0, 20, 450)
    _add_ocgt_tier("DE", 1, 20, 450)
    _add_ocgt_tier("DE", 2, 20, 450)

    # ── DE NewPeak: 2 tiers × 12 units × 400 MW = 9,600 MW
    _add_np_tier("DE", 0, 12, 400)
    _add_np_tier("DE", 1, 12, 400)

    # ── DE Pumped hydro: 4 named plants
    # Goldisthal (Thüringen, VATTENFALL): 4 × 265 MW = 1,060 MW, 8,480 MWh
    # Markersbach (Saxony, LEAG): 7 × 150 MW = 1,050 MW, 4,200 MWh
    # Wehr (Baden-Württemberg, EnBW): 4 × 228 MW = 910 MW, 7,280 MWh
    # Waldeck 2 (Hessen, E.ON): 2 × 240 MW = 480 MW, 1,800 MWh
    fleet.add(_st("DE_PH_Goldisthal",  "DE", 1_060, 8_480, 1_060, 1_060, 0.88, 0.88, 4_240, 1.5))
    fleet.add(_st("DE_PH_Markersbach", "DE", 1_050, 4_200, 1_050, 1_050, 0.87, 0.87, 2_100, 1.5))
    fleet.add(_st("DE_PH_Wehr",        "DE",   910, 7_280,   910,   910, 0.88, 0.88, 3_640, 1.5))
    fleet.add(_st("DE_PH_Waldeck",     "DE",   480, 1_800,   480,   480, 0.87, 0.87,   900, 1.5))
    fleet.add(_st("DE_Battery_L",      "DE", 2_000, 4_000, 2_000, 2_000, 0.92, 0.92, 2_000, 4.0))
    fleet.add(_st("DE_Battery_M",      "DE", 1_000, 2_000, 1_000, 1_000, 0.92, 0.92, 1_000, 4.5))


    # ══════════════════════════════════════════════════════════════════════════
    # FR — France
    # ══════════════════════════════════════════════════════════════════════════
    # Nuclear calibration — Jan 2024 reality check:
    # ─────────────────────────────────────────────
    # Installed nuclear capacity (EDF fleet): ~63 GW across 56 reactors.
    # BUT: real available output in Jan 2024 was ~35–38 GW due to:
    #   - Planned refuelling outages (~10 reactors offline at any time)
    #   - Corrosion inspection shutdowns (Flamanville/Penly stress-corrosion)
    #   - Typical Jan availability factor ~57–60%
    # Real FR day-ahead prices Jan 2024: ~50–80 EUR/MWh (positive, not -50)
    # Real FR exports Jan 2024: 5,000–8,000 MW average (not 13,000+)
    #
    # Model structure:
    #   Block A (baseload, fixed_on=True):  ~30,000 MW — always-on fleet
    #     max_cap=35,000 (nameplate of always-on fleet)
    #     min_cap=28,000 (fleet minimum at lowest demand)
    #     Represents ~20 reactors in continuous operation
    #
    #   Block B (flexible nuclear, fixed_on=False): ~12,000 MW
    #     The remaining ~8 reactors that cycle seasonally.
    #     GA decides their commitment — correctly OFF in mild hours.
    #     min_cap=4,000 (4 reactors minimum if any committed)
    #
    # With Block A at 30 GW and overnight demand ~32 GW,
    # net thermal demand for dispatchables ≈ 2,000–5,000 MW — realistic.
    fleet.add(_th("FR_Nuclear", "FR", 35_000, 28_000, NUC, 7.0, 3.1,
        startup_cost=2_000_000, min_run_hours=24, ramp_rate=500, fixed_on=True,
        offer_price=-50.0))   # -50 EUR/MWh: EDF bids negative to avoid cold start cost
    fleet.add(_th("FR_Nuclear_B", "FR", 12_000, 1_200, NUC, 7.0, 3.1,
        startup_cost=800_000, min_run_hours=12, ramp_rate=600, fixed_on=False,
        offer_price=-30.0))   # -30 EUR/MWh: flexible nuclear, lower min_cap so GA can
                               # genuinely back it off in overnight surplus hours.
                               # min_cap=1200 MW = 2 reactors at minimum stable load.
                               # Was 4000 (4 reactors) which forced a 4GW floor even
                               # overnight when nuclear A already covers 28GW of ~30GW demand.

    rng_fr = np.random.default_rng(505)
    for i, (cap, minr, res, res0, infl, wv) in enumerate([
        (4_000, 0, 1_129_000, 627_000, 89_600, 30.0),
        (3_500, 0,   989_000, 549_000, 78_400, 55.0),
        (2_800, 0,   791_000, 439_000, 62_700, 78.0),
        (1_700, 0,   480_000, 267_000, 38_100, 90.0),
    ]):
        rain_f = 0.7 + 0.3*rng_fr.random()
        fleet.add(_hy(f"FR_Hydro_{i+1}", "FR", cap, int(cap*minr),
            res, res0, infl*rain_f, wv, startup_cost=2_000+i*300))

    _add_ccgt_tier("FR", 0, 8, 350)
    _add_ccgt_tier("FR", 1, 8, 350)
    _add_ccgt_tier("FR", 2, 8, 350)
    _add_ccgt_tier("FR", 3, 8, 350)
    _add_ocgt_tier("FR", 0, 10, 400)
    _add_ocgt_tier("FR", 1, 10, 400)
    _add_ocgt_tier("FR", 2, 10, 400)
    _add_np_tier("FR", 0, 6, 300)
    _add_np_tier("FR", 1, 6, 300)
    fleet.add(_st("FR_PumpedHydro", "FR", 2_000, 8_000, 2_000, 2_000, 0.87, 0.87, 4_000, 1.5))
    fleet.add(_st("FR_Battery",     "FR",   600, 1_200,   600,   600, 0.92, 0.92,   600, 4.0))


    # ══════════════════════════════════════════════════════════════════════════
    # GB — Great Britain  (unchanged from v1)
    # ══════════════════════════════════════════════════════════════════════════
    fleet.add(_th("GB_Nuclear", "GB", 6_500, 4_000, NUC, 7.5, 3.2,
        startup_cost=400_000, min_run_hours=24, ramp_rate=1_000, fixed_on=True,
        offer_price=-40.0))   # -40 EUR/MWh: UK nuclear (EDF/Hinkley) avoidance bid
    # CCGT: 6 tiers × 14 units × 350 MW = 29,400 MW
    # Real Jan 2024 GB CCGT installed: ~35 GW. 6 tiers × 14 × 350 ≈ 29.4 GW
    # Plus existing coal/biomass/OCGT brings total firm to ~52 GW vs 42 GW gross peak
    _add_ccgt_tier("GB", 0, 14, 350)
    _add_ccgt_tier("GB", 1, 14, 350)
    _add_ccgt_tier("GB", 2, 14, 350)
    _add_ccgt_tier("GB", 3, 14, 350)
    _add_ccgt_tier("GB", 4, 14, 350)   # capacity fix: +4,900 MW
    _add_ccgt_tier("GB", 5, 14, 350)   # capacity fix: +4,900 MW (total CCGT 29.4 GW)
    _add_coal_tier("GB", 0, 4, 250)
    _add_coal_tier("GB", 1, 4, 250)
    # OCGT: 3 tiers × 18 units × 350 MW = 18,900 MW
    _add_ocgt_tier("GB", 0, 18, 350)
    _add_ocgt_tier("GB", 1, 18, 350)
    _add_ocgt_tier("GB", 2, 18, 350)
    _add_np_tier("GB", 0, 8, 250)
    _add_np_tier("GB", 1, 8, 250)
    fleet.add(_st("GB_PumpedHydro_L", "GB", 1_728, 6_912, 1_728, 1_728, 0.87, 0.87, 3_456, 1.5))
    fleet.add(_st("GB_PumpedHydro_M", "GB", 1_072, 4_288, 1_072, 1_072, 0.87, 0.87, 2_144, 1.5))
    fleet.add(_st("GB_Battery",       "GB",   800, 1_600,   800,   800, 0.92, 0.92,   800, 3.5))


    # ══════════════════════════════════════════════════════════════════════════
    # PL — Poland  (unchanged from v1)
    # ══════════════════════════════════════════════════════════════════════════
    # PL — Poland.  Residual peak: 17,000 MW.  Target: ~30,600 MW (1.8x)
    # Poland still heavily coal-dependent (Bełchatów lignite, Kozienice, Opole hard coal)
    _add_ccgt_tier("PL", 0, 6, 350)
    _add_ccgt_tier("PL", 1, 6, 350)
    _add_ccgt_tier("PL", 2, 6, 350)
    _add_ccgt_tier("PL", 3, 6, 350)
    _add_coal_tier("PL", 0, 8, 300)   # Bełchatów, Kozienice, Opole, Jaworzno
    _add_coal_tier("PL", 1, 8, 300)
    _add_coal_tier("PL", 2, 8, 300)
    _add_ocgt_tier("PL", 0, 12, 250)
    _add_ocgt_tier("PL", 1, 12, 250)
    _add_ocgt_tier("PL", 2, 12, 250)
    _add_np_tier("PL", 0, 8, 200)
    _add_np_tier("PL", 1, 8, 200)
    _add_ocgt_tier("PL", 3, 8, 250)   # T4 extra
    fleet.add(_st("PL_Battery", "PL", 600, 1_200, 600, 600, 0.90, 0.90, 600, 5.0))


    # NL — Netherlands
    # Real Jan 2024: ~15 GW CCGT + 3 GW older gas + 0.5 GW nuclear = ~18 GW firm
    # vs 14 GW gross peak → 1.29× (tight, relies on BE/DE/GB imports)
    # Previous fleet (15,223 MW) had only 117 MW headroom vs demand max → scarcity.
    # Adding T5 tier brings firm thermal to ~18,223 MW = 1.30× gross peak.
    _add_ccgt_tier("NL", 0, 6, 250)
    _add_ccgt_tier("NL", 1, 6, 250)
    _add_ccgt_tier("NL", 2, 6, 250)
    _add_ccgt_tier("NL", 3, 6, 250)
    _add_ccgt_tier("NL", 4, 6, 250)
    _add_ccgt_tier("NL", 5, 6, 500)   # +3,000 MW: older gas/steam capacity, MC≈130
    _add_ocgt_tier("NL", 0, 8, 250)
    _add_ocgt_tier("NL", 1, 8, 250)
    _add_ocgt_tier("NL", 2, 8, 250)
    _add_ocgt_tier("NL", 3, 8, 250)
    _add_np_tier("NL", 0, 6, 200)
    _add_np_tier("NL", 1, 6, 200)
    fleet.add(_st("NL_Battery", "NL", 400, 800, 400, 400, 0.92, 0.92, 400, 4.0))


    # ══════════════════════════════════════════════════════════════════════════
    # BE — Belgium
    # ══════════════════════════════════════════════════════════════════════════
    fleet.add(_th("BE_Nuclear", "BE", 5_800, 2_000, NUC, 8.0, 3.2,
        startup_cost=250_000, min_run_hours=12, ramp_rate=500, fixed_on=True,
        offer_price=-30.0))   # -30 EUR/MWh: Doel/Tihange — smaller avoided cost
    # CCGT: 4 tiers × 4 units × 250 MW = 4,000 MW
    _add_ccgt_tier("BE", 0, 4, 250)
    _add_ccgt_tier("BE", 1, 4, 250)
    _add_ccgt_tier("BE", 2, 4, 250)
    _add_ccgt_tier("BE", 3, 4, 250)
    # OCGT: 3 tiers × 8 units × 200 MW = 4,800 MW + T4 extra
    _add_ocgt_tier("BE", 0, 8, 200)
    _add_ocgt_tier("BE", 1, 8, 200)
    _add_ocgt_tier("BE", 2, 8, 200)
    _add_ocgt_tier("BE", 3, 6, 200)   # T4 extra: 1,200 MW → total 6,000 MW
    _add_np_tier("BE", 0, 4, 150)
    _add_np_tier("BE", 1, 4, 150)
    fleet.add(_st("BE_Battery", "BE", 350, 700, 350, 350, 0.92, 0.92, 350, 4.5))


    # ══════════════════════════════════════════════════════════════════════════
    # CZ — Czech Republic
    # ══════════════════════════════════════════════════════════════════════════
    fleet.add(_th("CZ_Nuclear", "CZ", 2_000, 1_200, NUC, 8.5, 3.3,
        startup_cost=150_000, min_run_hours=12, ramp_rate=100, fixed_on=True,
        offer_price=-35.0))   # -35 EUR/MWh: Dukovany/Temelin avoidance bid
    _add_ccgt_tier("CZ", 0, 4, 150)
    _add_ccgt_tier("CZ", 1, 4, 150)
    _add_ccgt_tier("CZ", 2, 4, 150)
    _add_ccgt_tier("CZ", 3, 4, 150)
    _add_coal_tier("CZ", 0, 6, 200)   # Czech brown coal fleet (Počerady, Ledvice, Tušimice)
    _add_coal_tier("CZ", 1, 6, 200)
    _add_ocgt_tier("CZ", 0, 6, 200)
    _add_ocgt_tier("CZ", 1, 6, 200)
    _add_ocgt_tier("CZ", 2, 6, 200)
    _add_np_tier("CZ", 0, 4, 150)
    _add_np_tier("CZ", 1, 4, 150)
    _add_ocgt_tier("CZ", 3, 10, 150)   # T4 extra capacity
    fleet.add(_st("CZ_Battery", "CZ", 300, 600, 300, 300, 0.91, 0.91, 300, 5.0))


    # ══════════════════════════════════════════════════════════════════════════
    # DK — Denmark
    # ══════════════════════════════════════════════════════════════════════════
    fleet.add(_th("DK_Biomass", "DK", 400, 380, COAL, 38.0, 1.1,
        startup_cost=0, min_run_hours=24, ramp_rate=200, fixed_on=True))
    _add_ccgt_tier("DK", 0, 2, 150)
    _add_ccgt_tier("DK", 1, 2, 150)
    _add_ccgt_tier("DK", 2, 2, 150)
    _add_ccgt_tier("DK", 3, 2, 150)
    _add_ocgt_tier("DK", 0, 4, 100)
    _add_ocgt_tier("DK", 1, 4, 100)
    _add_ocgt_tier("DK", 2, 4, 100)
    _add_np_tier("DK", 0, 2, 100)
    _add_np_tier("DK", 1, 2, 100)
    _add_ocgt_tier("DK", 3, 8, 200)   # T4: extra capacity
    _add_np_tier("DK", 0, 4, 100)     # capacity fix: +400 MW via NewPeak
    fleet.add(_st("DK_Battery", "DK", 250, 500, 250, 250, 0.93, 0.93, 250, 3.0))


    # ══════════════════════════════════════════════════════════════════════════
    # AT — Austria (hydro-dominated)
    # ══════════════════════════════════════════════════════════════════════════
    rng_at = np.random.default_rng(101)
    for i, (cap, minr, res, res0, infl, wv) in enumerate([
        (1_200, 0.10, 280_000, 155_000, 43_000, 45.0),
        (1_100, 0.10, 258_000, 143_000, 39_500, 55.0),
        (1_100, 0.10, 258_000, 143_000, 39_500, 65.0),
        (1_050, 0.10, 246_000, 137_000, 37_700, 78.0),
        (1_050, 0.10, 246_000, 137_000, 37_700, 88.0),
        (1_000, 0.10, 234_000, 130_000, 35_900, 98.0),
        (1_000, 0.10, 234_000, 130_000, 35_900, 112.0),
        (  600, 0.12,  66_000,  37_000, 10_100, 128.0),
        (  600, 0.12,  66_000,  37_000, 10_100, 145.0),
    ]):
        rain_f = 0.7 + 0.6*rng_at.random()
        fleet.add(_hy(f"AT_Hydro_{i+1:02d}", "AT", cap, int(cap*minr),
            res, res0, infl*rain_f, wv, startup_cost=1_000+i*100))
    _add_ccgt_tier("AT", 0, 4, 150)
    _add_ccgt_tier("AT", 1, 4, 150)
    _add_ccgt_tier("AT", 2, 4, 150)
    _add_ccgt_tier("AT", 3, 4, 150)
    _add_ocgt_tier("AT", 0, 4, 150)
    _add_ocgt_tier("AT", 1, 4, 150)
    _add_ocgt_tier("AT", 2, 4, 150)
    _add_np_tier("AT", 0, 2, 100)
    _add_np_tier("AT", 1, 2, 100)
    fleet.add(_st("AT_PumpedStorage", "AT", 800, 3_200, 800, 800, 0.88, 0.88, 1_600, 2.0))


    # ══════════════════════════════════════════════════════════════════════════
    # CH — Switzerland (hydro+nuclear dominated)
    # ══════════════════════════════════════════════════════════════════════════
    fleet.add(_th("CH_Nuclear", "CH", 3_000, 900, NUC, 8.0, 3.1,
        startup_cost=220_000, min_run_hours=12, ramp_rate=150, fixed_on=True,
        offer_price=-35.0))   # -35 EUR/MWh: Gösgen/Leibstadt/Mühleberg avoidance bid
    rng_ch = np.random.default_rng(202)
    for i, (cap, minr, res, res0, infl, wv) in enumerate([
        (1_400, 0.10, 252_000, 140_000, 65_300, 50.0),
        (1_350, 0.10, 243_000, 135_000, 62_900, 65.0),
        (1_250, 0.10, 225_000, 125_000, 58_200, 78.0),
        (1_200, 0.10, 216_000, 120_000, 55_900, 90.0),
        (1_200, 0.10, 216_000, 120_000, 55_900, 102.0),
        (1_150, 0.10, 207_000, 115_000, 53_600, 115.0),
        (1_100, 0.10,  81_000,  45_000, 25_200, 132.0),
        (1_150, 0.12,  81_000,  45_000, 25_200, 155.0),
    ]):
        rain_f = 0.4 + 0.4*rng_ch.random()
        fleet.add(_hy(f"CH_Hydro_{i+1:02d}", "CH", cap, int(cap*minr),
            res, res0, infl*rain_f, wv, startup_cost=1_000+i*100))
    _add_ccgt_tier("CH", 0, 4, 150)
    _add_ccgt_tier("CH", 1, 4, 150)
    _add_ccgt_tier("CH", 2, 4, 150)
    _add_ccgt_tier("CH", 3, 4, 150)
    _add_ocgt_tier("CH", 0, 4, 100)
    _add_ocgt_tier("CH", 1, 4, 100)
    _add_ocgt_tier("CH", 2, 4, 100)
    _add_np_tier("CH", 0, 2, 100)
    _add_np_tier("CH", 1, 2, 100)
    fleet.add(_st("CH_PumpedStorage", "CH", 1_500, 6_000, 1_500, 1_500, 0.87, 0.87, 3_000, 1.5))


    # ══════════════════════════════════════════════════════════════════════════
    # SE — Sweden
    # ══════════════════════════════════════════════════════════════════════════
    for cap, sc, rp, idx in [(3_350,200_000,130,1),(2_800,180_000,130,2),(2_750,170_000,130,3)]:
        fleet.add(_th(f"SE_Nuclear_{idx}", "SE", cap, int(cap*0.45), NUC, 7.5, 3.2,
            startup_cost=sc, min_run_hours=12, ramp_rate=rp, fixed_on=True,
            offer_price=-35.0))  # -35 EUR/MWh: Ringhals/Forsmark/Oskarshamn avoidance bid
    fleet.add(_th("SE_Biomass", "SE", 1_200, 1_100, COAL, 36.0, 1.1,
        startup_cost=0, min_run_hours=24, ramp_rate=300, fixed_on=True))
    rng_se = np.random.default_rng(404)
    for i, (cap, minr, res, res0, infl, wv) in enumerate([
        (2_200, 0.10, 541_000, 300_000, 73_700, 55.0),
        (2_000, 0.10, 491_000, 272_000, 67_000, 68.0),
        (1_800, 0.10, 442_000, 245_000, 60_300, 78.0),
        (1_600, 0.10, 393_000, 218_000, 53_600, 88.0),
        (1_400, 0.10, 344_000, 191_000, 46_900, 96.0),
        (  900, 0.10, 221_000, 122_000, 30_200, 118.0),
        (  800, 0.10, 197_000, 109_000, 26_800, 130.0),
    ]):
        rain_f = 0.5 + 0.5*rng_se.random()
        fleet.add(_hy(f"SE_Hydro_L{i+1:02d}", "SE", cap, int(cap*minr),
            res, res0, infl*rain_f, wv, startup_cost=600+i*60))
    for i, (cap, minr, res, res0, infl, wv) in enumerate([
        (600, 0.12, 147_000, 81_000, 20_200, 138.0),
        (500, 0.12, 123_000, 68_000, 16_800, 142.0),
        (400, 0.12,  98_000, 54_000, 13_400, 145.0),
        (300, 0.12,  74_000, 41_000, 10_100, 148.0),
    ]):
        rain_f = 0.5 + 0.5*rng_se.random()
        fleet.add(_hy(f"SE_Hydro_M{i+1:02d}", "SE", cap, int(cap*minr),
            res, res0, infl*rain_f, wv, startup_cost=400+i*60))
    _add_ccgt_tier("SE", 0, 6, 200)
    _add_ccgt_tier("SE", 1, 6, 200)
    _add_ccgt_tier("SE", 2, 6, 200)
    _add_ccgt_tier("SE", 3, 6, 200)
    _add_ocgt_tier("SE", 0, 8, 150)
    _add_ocgt_tier("SE", 1, 8, 150)
    _add_ocgt_tier("SE", 2, 8, 150)
    _add_ocgt_tier("SE", 3, 8, 150)   # T4 extra to reach 1.8x
    _add_np_tier("SE", 0, 4, 150)
    _add_np_tier("SE", 1, 4, 150)
    fleet.add(_st("SE_PumpedStorage", "SE", 600, 2_400, 600, 600, 0.87, 0.87, 1_200, 2.0))


    # ══════════════════════════════════════════════════════════════════════════
    # NO — Norway (all hydro + emergency OCGT)
    # ══════════════════════════════════════════════════════════════════════════
    rng_no = np.random.default_rng(303)
    for i, (cap, minr, res, res0, infl, wv) in enumerate([
        (4_000, 0.10, 1_290_000, 718_000, 155_000, 60.0),
        (3_800, 0.10, 1_226_000, 682_000, 147_000, 62.0),
        (3_500, 0.10, 1_129_000, 628_000, 136_000, 64.0),
        (3_200, 0.10, 1_032_000, 574_000, 124_000, 66.0),
        (3_000, 0.10,   968_000, 538_000, 116_000, 68.0),
        (2_800, 0.10,   903_000, 502_000, 108_000, 70.0),
        (2_500, 0.10,   806_000, 448_000,  96_500, 73.0),
        (2_200, 0.10,   710_000, 394_000,  85_000, 76.0),
    ]):
        rain_f = 0.5 + 0.5*rng_no.random()
        fleet.add(_hy(f"NO_Hydro_L{i+1:02d}", "NO", cap, int(cap*minr),
            res, res0, infl*rain_f, wv, startup_cost=600+i*80))
    for i, (cap, minr, res, res0, infl, wv) in enumerate([
        (1_800, 0.12, 420_000, 233_000, 61_800, 100.0),
        (1_600, 0.12, 373_000, 207_000, 54_900, 102.0),
        (1_500, 0.12, 350_000, 194_000, 51_500, 104.0),
        (1_300, 0.12, 303_000, 168_000, 44_600, 106.0),
        (1_200, 0.12, 280_000, 155_000, 41_200, 108.0),
    ]):
        rain_f = 0.45 + 0.55*rng_no.random()
        fleet.add(_hy(f"NO_Hydro_M{i+1:02d}", "NO", cap, int(cap*minr),
            res, res0, infl*rain_f, wv, startup_cost=500+i*60))
    for i, (cap, minr, res, res0, infl, wv) in enumerate([
        (1_200, 0.60, 2_400, 1_200, 1_008, 68.0),
        (  800, 0.60, 1_600,   800,   672, 72.0),
    ]):
        rain_f = 0.6 + 0.4*rng_no.random()
        fleet.add(_hy(f"NO_RoR_{i+1}", "NO", cap, int(cap*minr),
            res, res0, infl*rain_f, wv, startup_cost=200))
    _add_ocgt_tier("NO", 0, 4, 100)
    fleet.add(_st("NO_PumpedStorage", "NO", 1_400, 5_600, 1_400, 1_400, 0.87, 0.87, 2_800, 2.0))


    # ══════════════════════════════════════════════════════════════════════════
    # IT — Italy
    # ══════════════════════════════════════════════════════════════════════════
    # Residual peak: 46,000 MW.  Target: ~86,000 MW (1.87x + pumped)
    #
    # Italy's generation mix (winter 2024 approximate):
    #   CCGT / combined cycle: ~43 GW installed (gas-dominant)
    #   Alpine hydro (reservoir): ~14 GW
    #   Pumped storage: ~7.7 GW (mainly Edolo, San Fiorano, Presenzano)
    #   Hard coal: ~7.5 GW (Brindisi, La Spezia — being phased out)
    #   Oil/distillate peakers: ~4 GW (emergency, very high MC)
    #   Biomass/waste: ~3.5 GW
    #
    # MC calibration note: Italian gas import cost is ~15% higher than TTF
    # due to pipeline import dependence (TAP, TANAP). Add 5 EUR/MWh to base_fuel.
    #
    # CCGT: 4 tiers × 12 units × 650 MW = 31,200 MW
    _add_ccgt_tier("IT", 0, 12, 650)
    _add_ccgt_tier("IT", 1, 12, 650)
    _add_ccgt_tier("IT", 2, 12, 650)
    _add_ccgt_tier("IT", 3, 12, 650)

    # Hard coal: 2 tiers × 5 units × 600 MW = 6,000 MW
    # (Brindisi Nord, Porto Tolle, Torrevaldaliga, La Spezia, Fusina)
    _add_coal_tier("IT", 0, 5, 600)
    _add_coal_tier("IT", 1, 5, 600)

    # OCGT: 3 tiers × 14 units × 500 MW = 21,000 MW
    _add_ocgt_tier("IT", 0, 14, 500)
    _add_ocgt_tier("IT", 1, 14, 500)
    _add_ocgt_tier("IT", 2, 14, 500)

    # Oil peakers (emergency, MC≈200+): NewPeak with high base_fuel
    # These only dispatch in scarcity.  Represent Italian SNAM emergency capacity.
    # Use NP tier 2 with additional base_fuel premium for oil (~0.65 co2_int)
    _add_np_tier("IT", 0, 6, 450)
    _add_np_tier("IT", 1, 6, 450)

    # Alpine hydro — 12 reservoir/RoR units
    # Diga del Vajont (Sade group), Diga di Cancano, Lago di Santa Croce etc.
    # Water values 50–130 EUR/MWh reflecting Italy's position as hydro price-setter
    rng_it = np.random.default_rng(606)
    for i, (cap, minr, res, res0, infl, wv) in enumerate([
        (2_000, 0.10, 380_000, 210_000, 55_000, 50.0),
        (1_800, 0.10, 342_000, 190_000, 49_500, 60.0),
        (1_600, 0.10, 304_000, 169_000, 44_000, 70.0),
        (1_500, 0.10, 285_000, 158_000, 41_200, 78.0),
        (1_400, 0.10, 266_000, 148_000, 38_400, 88.0),
        (1_200, 0.10, 228_000, 127_000, 32_900, 98.0),
        (1_000, 0.10, 190_000, 106_000, 27_400, 108.0),
        (  800, 0.12, 108_000,  60_000, 18_000, 118.0),
        (  700, 0.12,  95_000,  53_000, 15_800, 125.0),
        (  600, 0.12,  81_000,  45_000, 13_500, 130.0),
    ]):
        rain_f = 0.5 + 0.5*rng_it.random()
        fleet.add(_hy(f"IT_Hydro_{i+1:02d}", "IT", cap, int(cap*minr),
            res, res0, infl*rain_f, wv, startup_cost=1_200+i*100))

    # Pumped storage: Edolo (1,000 MW), San Fiorano (1,000 MW),
    #                 Presenzano (1,000 MW), Roncovalgrande (1,000 MW)
    fleet.add(_st("IT_PH_Edolo",        "IT", 1_000, 4_000, 1_000, 1_000, 0.87, 0.87, 2_000, 1.5))
    fleet.add(_st("IT_PH_SanFiorano",   "IT", 1_000, 4_000, 1_000, 1_000, 0.87, 0.87, 2_000, 1.5))
    fleet.add(_st("IT_PH_Presenzano",   "IT", 1_000, 4_000, 1_000, 1_000, 0.88, 0.88, 2_000, 1.5))
    fleet.add(_st("IT_PH_Roncovalgrande","IT",  700, 2_800,   700,   700, 0.87, 0.87, 1_400, 1.5))
    fleet.add(_st("IT_Battery",         "IT",  800, 1_600,   800,   800, 0.92, 0.92,   800, 4.5))


    # ══════════════════════════════════════════════════════════════════════════
    # ES — Spain
    # ══════════════════════════════════════════════════════════════════════════
    # Residual peak: 32,000 MW.  Target: ~62,000 MW (1.91x + pumped)
    #
    # Spain's generation mix (winter 2024 approximate):
    #   Nuclear: 7,100 MW (6 plants: Almaraz ×2, Ascó ×2, Cofrentes, Vandellós)
    #   CCGT: ~24 GW installed (Sagunto, Besós, Tarragona etc.)
    #   Hard coal: ~3 GW remaining (Aboño, Soto de Ribera)
    #   OCGT: ~13 GW peakers
    #   Hydro: ~17 GW reservoir + run-of-river
    #   Pumped: ~6.8 GW (La Muela, Aldeadávila, Aguayo, Bolarque)
    #
    # Nuclear: 6 plants modelled as two blocks (3+3) — different vintages
    fleet.add(_th("ES_Nuclear_Block1", "ES", 3_600, 2_200, NUC, 7.8, 3.15,
        startup_cost=250_000, min_run_hours=24, ramp_rate=200, fixed_on=True,
        offer_price=-40.0))   # -40 EUR/MWh: Almaraz/Ascó avoidance bid
    fleet.add(_th("ES_Nuclear_Block2", "ES", 3_500, 2_100, NUC, 8.0, 3.20,
        startup_cost=250_000, min_run_hours=24, ramp_rate=200, fixed_on=True,
        offer_price=-40.0))   # -40 EUR/MWh: Cofrentes/Vandellós avoidance bid

    # CCGT: 4 tiers × 8 units × 550 MW = 17,600 MW
    _add_ccgt_tier("ES", 0, 8, 550)
    _add_ccgt_tier("ES", 1, 8, 550)
    _add_ccgt_tier("ES", 2, 8, 550)
    _add_ccgt_tier("ES", 3, 8, 550)

    # Hard coal: 2 tiers × 3 units × 500 MW = 3,000 MW
    _add_coal_tier("ES", 0, 3, 500)
    _add_coal_tier("ES", 1, 3, 500)

    # OCGT: 3 tiers × 10 units × 450 MW = 13,500 MW
    _add_ocgt_tier("ES", 0, 10, 450)
    _add_ocgt_tier("ES", 1, 10, 450)
    _add_ocgt_tier("ES", 2, 10, 450)

    # NewPeak: 2 tiers × 5 units × 400 MW = 4,000 MW
    _add_np_tier("ES", 0, 5, 400)
    _add_np_tier("ES", 1, 5, 400)

    # Hydro — Iberian reservoir hydro (Tajo, Duero, Ebro basins)
    # Water values lower than central Europe (rain-fed, less seasonal)
    rng_es = np.random.default_rng(707)
    for i, (cap, minr, res, res0, infl, wv) in enumerate([
        (2_500, 0.10, 600_000, 333_000, 82_000, 35.0),
        (2_200, 0.10, 528_000, 293_000, 72_000, 45.0),
        (2_000, 0.10, 480_000, 267_000, 65_500, 52.0),
        (1_800, 0.10, 432_000, 240_000, 59_000, 60.0),
        (1_600, 0.10, 384_000, 213_000, 52_400, 68.0),
        (1_400, 0.10, 336_000, 187_000, 45_900, 75.0),
        (1_200, 0.10, 288_000, 160_000, 39_300, 82.0),
        (1_000, 0.12, 138_000,  77_000, 22_000, 90.0),
        (  900, 0.12, 124_000,  69_000, 19_800, 96.0),
        (  800, 0.12, 110_000,  61_000, 17_600, 103.0),
        (  700, 0.12,  97_000,  54_000, 15_400, 110.0),
        (  600, 0.12,  83_000,  46_000, 13_200, 118.0),
        (  500, 0.12,  69_000,  38_000, 11_000, 125.0),
        (  400, 0.12,  55_000,  31_000,  8_800, 132.0),
    ]):
        rain_f = 0.5 + 0.5*rng_es.random()
        fleet.add(_hy(f"ES_Hydro_{i+1:02d}", "ES", cap, int(cap*minr),
            res, res0, infl*rain_f, wv, startup_cost=1_000+i*80))

    # Pumped storage: La Muela (832 MW), Aldeadávila-II (810 MW),
    #                 Aguayo (750 MW), Bolarque-II (223 MW)
    fleet.add(_st("ES_PH_LaMuela",    "ES",   832, 3_328, 832, 832, 0.87, 0.87, 1_664, 1.5))
    fleet.add(_st("ES_PH_Aldeadavila","ES",   810, 3_240, 810, 810, 0.88, 0.88, 1_620, 1.5))
    fleet.add(_st("ES_PH_Aguayo",     "ES",   750, 3_000, 750, 750, 0.87, 0.87, 1_500, 1.5))
    fleet.add(_st("ES_PH_Bolarque",   "ES",   223,   892, 223, 223, 0.87, 0.87,   446, 1.5))
    fleet.add(_st("ES_Battery",       "ES",   600, 1_200, 600, 600, 0.92, 0.92,   600, 4.0))


    # ══════════════════════════════════════════════════════════════════════════
    # HU — Hungary
    # ══════════════════════════════════════════════════════════════════════════
    # Residual peak: 5,500 MW.  Target: ~10,000 MW (1.8x)
    #
    # Hungary's generation mix (2024):
    #   Nuclear: Paks NPP — 4 × 500 MW = 2,000 MW (VVER-440).
    #            Paks II under construction (2030+), not modelled.
    #   CCGT: ~3.5 GW (MFGT, Dunamenti, Kelenfold, Kelenföld gas blocks)
    #   Coal/lignite: Mátrai Erőmű ~0.8 GW (Mátra lignite, being phased out)
    #   OCGT/gas: ~1.5 GW backup
    #
    fleet.add(_th("HU_Nuclear_Paks", "HU", 2_000, 1_200, NUC, 8.5, 3.30,
        startup_cost=200_000, min_run_hours=24, ramp_rate=100, fixed_on=True,
        offer_price=-40.0))   # -40 EUR/MWh: Paks NPP avoidance bid

    # Mátrai lignite (Mátra Power Plant) — being phased out 2025-2030
    fleet.add(_th("HU_Lignite_Matra", "HU", 800, 0, COAL, 5.8, 2.90,
        startup_cost=120_000, min_run_hours=16, ramp_rate=80))

    # CCGT: 4 tiers × 3 units × 300 MW = 3,600 MW
    _add_ccgt_tier("HU", 0, 3, 300)
    _add_ccgt_tier("HU", 1, 3, 300)
    _add_ccgt_tier("HU", 2, 3, 300)
    _add_ccgt_tier("HU", 3, 3, 300)

    # OCGT: 3 tiers × 4 units × 200 MW = 2,400 MW
    _add_ocgt_tier("HU", 0, 4, 200)
    _add_ocgt_tier("HU", 1, 4, 200)
    _add_ocgt_tier("HU", 2, 4, 200)   # T3 extra to cover 1.8x

    # NewPeak: 2 tiers × 3 units × 150 MW = 900 MW
    _add_np_tier("HU", 0, 3, 150)
    _add_np_tier("HU", 1, 3, 150)

    fleet.add(_st("HU_Battery", "HU", 200, 400, 200, 200, 0.91, 0.91, 200, 5.0))

    # ══════════════════════════════════════════════════════════════════════════
    # RES ASSETS — Wind and Solar (one of each per region)
    # ══════════════════════════════════════════════════════════════════════════
    # Each WindPlant / SolarPlant has:
    #   max_cap  = installed capacity (MW nameplate)
    #   avail    = ERA5 capacity factor profile — set per-window in make_fn
    #              via the _era5_wind_cf / _era5_solar_cf accessors
    #   offer_price = DA market bid (0 or slightly negative for subsidised plants)
    #
    # The avail profile is updated each window in build_scenario_fns via
    # fleet.res_units(r)[n].avail = era5_cf_for_this_window
    # This is handled cleanly in make_fn without any demand subtraction.
    #
    # Gross demand (updated in DEMAND_PEAKS) means the GA sees the full
    # consumer demand and must dispatch wind + thermal to cover it.
    # When wind fills demand → MCP = wind offer_price (0 EUR/MWh).
    # When wind is curtailed → GA reduces dispatch fraction below 1.0.
    for r in REGIONS:
        w_cap  = WIND_INSTALLED.get(r, 0)
        s_cap  = SOLAR_INSTALLED.get(r, 0)
        w_off  = WIND_OFFER_PRICE.get(r, 0.0)
        s_off  = SOLAR_OFFER_PRICE.get(r, 0.0)
        if w_cap > 0:
            fleet.add(WindPlant(
                name                 = f"{r}_Wind",
                region               = r,
                max_cap              = float(w_cap),
                offer_price          = float(w_off),
                # curtailment_threshold = price below which this plant stops dispatching.
                # Equals offer_price: a plant that bids -5 EUR/MWh only curtails when
                # MCP falls below -5 EUR/MWh (i.e. when even its subsidy doesn't cover costs).
                # For unsubsidised plants (w_off=0): curtail at exactly 0 EUR/MWh.
                curtailment_threshold = float(w_off),
            ))
        if s_cap > 0:
            fleet.add(SolarPlant(
                name                 = f"{r}_Solar",
                region               = r,
                max_cap              = float(s_cap),
                offer_price          = float(s_off),
                # Solar is overwhelmingly CfD/FIT subsidised in Europe but still
                # bids at 0 EUR/MWh in the DA market (strike price paid separately).
                # Curtailment threshold = 0: curtail only when MCP would go negative.
                curtailment_threshold = float(s_off),
            ))

    return fleet


# ══════════════════════════════════════════════════════════════════════════════
# NETWORK
# ══════════════════════════════════════════════════════════════════════════════

def build_network() -> Network:
    """
    16-region network.  New links for IT, ES, HU.

    IT links — winter NTC approximations:
      IT-CH: 1,800 MW  (Lavorgo/San Bernardino corridors)
      IT-AT: 800 MW   (Brenner/Tauern axis)
      IT-FR: 1,000 MW  (Moncenisio/Fréjus)
      IT-SI: not modelled (Slovenia out of scope — treated as IT-AT flow)

    ES links:
      ES-FR: 2,800 MW  (Pyrenean AC corridors — historically congested)
      ES-PT: 3,000 MW  (Iberian internal, very wide)

    HU links:
      HU-AT: 1,000 MW  (TenneT/MAVIR, AC corridor Kittsee-Győr)
      HU-CZ: 500 MW   (approximated; real flows via SK not modelled)

    Loss factors remain as v1 for existing links.
    """
    net = Network()

    for ra, rb, ab, ba, lf in [
        # CWE-9 core (unchanged from v1)
        ("DE","FR", 3_200, 3_200, 0.010), ("DE","NL", 3_000, 3_000, 0.010),
        ("DE","BE", 3_200, 3_200, 0.010), ("DE","AT", 4_000, 4_000, 0.010),
        ("DE","CH", 3_000, 3_000, 0.010), ("DE","CZ", 2_200, 2_200, 0.010),
        ("DE","DK", 1_800, 1_800, 0.010), ("DE","PL",   800,   800, 0.015),
        ("FR","BE", 3_000, 3_000, 0.010), ("FR","CH", 2_500, 2_500, 0.010),
        ("BE","NL", 2_500, 2_500, 0.010), ("AT","CH", 3_500, 3_500, 0.010),
        ("AT","CZ", 1_500, 1_500, 0.010), ("CZ","PL", 1_500, 1_500, 0.010),
        ("NL","DK",   700,   700, 0.015),
        # Nordic HVDC
        ("NO","DK", 1_400, 1_400, 0.015), ("NO","DE", 1_400, 1_400, 0.025),
        ("NO","SE", 3_600, 3_600, 0.010), ("SE","DK",   900,   900, 0.015),
        ("SE","DE",   600,   600, 0.015), ("SE","PL",   600,   600, 0.015),
        # GB HVDC cables
        ("GB","FR", 2_000, 2_000, 0.020), ("GB","NL", 1_000, 1_000, 0.020),
        ("GB","BE", 1_000, 1_000, 0.020), ("GB","DK", 1_400, 1_400, 0.025),
        # IT — cross-border corridors
        ("IT","CH", 1_800, 1_800, 0.010), ("IT","AT",   800,   800, 0.010),
        ("IT","FR", 1_000, 1_000, 0.010),
        # ES — Pyrenean and Iberian
        ("ES","FR", 2_800, 2_800, 0.010),
        # HU — central European AC
        ("HU","AT", 1_000, 1_000, 0.010), ("HU","CZ",   500,   500, 0.010),
    ]:
        net.add_link(TransmissionLink(ra, rb, ab, ba, lf, 0.0))

    # PTDF for AC continental grid (CWE + IT + ES + HU); GB/NO/SE via HVDC only
    ac_regions = ["AT","BE","CH","CZ","DE","DK","ES","FR","HU","IT","NL","PL"]
    net.build_ptdf(ac_regions, slack_region="DE", ram_fraction=0.75)

    return net


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def build_scenario_fns(fleet):
    """
    January 2024 scenario generators — calendar-aware, AR(1) noise, multi-stress.

    Key improvements over v2 generic scenario
    ------------------------------------------
    1. Calendar demand scaling: each window uses the correct Jan 2024 temperature
       multiplier. Cold snap week (Jan 8-14) pushes DE demand to ~49 GW peak;
       mild weeks (Jan 1-7, Jan 22-28) drop to ~40 GW.

    2. AR(1) demand noise (phi=0.60): consecutive hours within a scenario are
       correlated. A cold morning hour is more likely followed by another cold
       morning hour — matching real load autocorrelation structure.

    3. Three stress levels per region:
       - Mild spell   (−8% demand, prob 0.15) — warm anomalies
       - Cold snap    (+10% demand, prob 0.05) — frost episodes
       - Dark doldrum (+10% demand, prob 0.03) — worst case

    4. Seasonal solar: January solar sensitivity reduced (0.1–0.6 vs 0.3–0.9)
       to reflect short days and low sun angle in Northern Europe.

    5. Exact calendar weekday detection: IS_WEEKEND[day] uses real Jan 2024
       calendar, so Jan 6/7, 13/14, 20/21, 27/28 get weekend shape.
       Jan 1 (New Year) treated as a public holiday (Sunday shape).
    """

    # Fuel price generator
    co2_corr = np.array([
        [1.0, 0.3, 0.0, 0.4],
        [0.3, 1.0, 0.0, 0.6],
        [0.0, 0.0, 1.0, 0.0],
        [0.4, 0.6, 0.0, 1.0],
    ])
    fuel_gen = FuelPriceGenerator(
        fuel_names=["gas", "coal", "nuclear", "co2"],
        base_prices={"gas": 35.0, "coal": 12.0, "nuclear": 4.5, "co2": 82.0},
        vol={"gas": 0.08, "coal": 0.04, "nuclear": 0.01, "co2": 0.12},
        mean_reversion=0.15,
        correlation=co2_corr,
    )

    WIND_SENS = {
        "DE": 1.2, "DK": 1.5, "NL": 1.3, "GB": 1.4,
        "FR": 0.8, "BE": 0.9, "PL": 0.7,
        "AT": 0.3, "CH": 0.2, "NO": 0.4, "SE": 0.6, "CZ": 0.5,
        "IT": 0.6, "ES": 1.0, "HU": 0.4,
    }
    # January solar: much lower than annual average everywhere
    SOLAR_SENS = {
        "DE": 0.15, "FR": 0.20, "IT": 0.35, "ES": 0.40,
        "GB": 0.10, "NL": 0.15, "BE": 0.15, "DK": 0.08,
        "AT": 0.15, "CH": 0.15, "NO": 0.03, "SE": 0.08,
        "PL": 0.15, "CZ": 0.15, "HU": 0.20,
    }

    # AR(1) noise parameters
    PHI_DEMAND  = 0.60
    NOISE_STD   = 0.020
    # Innovation std keeps unconditional std = NOISE_STD
    INNOV_STD_D = NOISE_STD * np.sqrt(1.0 - PHI_DEMAND**2)

    # Pre-build one DemandGenerator per (region, calendar_day)
    # so each window gets the correctly temperature-scaled base profile
    day_gens = {}   # keyed by (region, day_idx 0..30)

    for r in REGIONS:
        base_wday = BASE_PROFILES[r]
        base_wend = _weekend_profile(base_wday, r)

        for day in range(31):
            ts    = _temp_scale(day, r)
            is_we = IS_WEEKEND[day] or IS_HOLIDAY[day]
            base  = (base_wend if is_we else base_wday) * ts

            # Three stress scenarios around this day's base
            mild    = base * (1.0 - 0.08)
            cold    = base * (1.0 + 0.10)
            doldrum = cold.copy()  # same demand as cold snap + low wind drawn externally

            gen = DemandGenerator(
                base_profile   = base,
                stress_profiles= [mild, cold, doldrum],
                stress_prob    = [0.15, 0.05, 0.03],
                noise_std      = INNOV_STD_D,
                ar1            = PHI_DEMAND,
                wind_sensitivity  = WIND_SENS.get(r, 0.6),
                solar_sensitivity = SOLAR_SENS.get(r, 0.4),
                allow_negative    = True,
            )
            day_gens[(r, day)] = gen

    hydro_gens = {}
    for r in REGIONS:
        hydro_units = fleet.hydros(r)
        if hydro_units:
            hydro_gens[r] = HydroInflowGenerator(
                unit_names=[h.name for h in hydro_units],
                ar1=None, std=None, demand_corr=None,
            )

    def make_fn(r):
        def fn(window_idx: int, hour_start: int, T: int):
            day = min(window_idx, 30)

            # ── ERA5 temperature scaling for gross demand ─────────────────────
            # Temperature scales gross demand (no wind subtraction — wind is now
            # a dispatchable asset in the chromosome).
            t_sc = _era5_temp_scale(r, hour_start, T)  # (T,) scaling

            # Update RES asset availability profiles for this window.
            # The availability profile IS the ERA5 capacity factor — it sets the
            # maximum dispatchable output per hour for each wind/solar plant.
            # julia_bridge multiplies the GA chromosome fraction by this profile:
            #   actual_output[t] = ga_fraction[t] × max_cap × avail[t]
            # ga_fraction=1.0 → full dispatch; ga_fraction<1.0 → curtailment.
            wind_cf  = _era5_wind_mw(r, hour_start, T)  / max(WIND_INSTALLED.get(r, 1), 1)
            solar_cf = _era5_solar_mw(r, hour_start, T) / max(SOLAR_INSTALLED.get(r, 1), 1)
            for u in fleet.res_units(r):
                if isinstance(u, WindPlant):
                    u.avail = np.clip(wind_cf[:T], 0.0, 1.0)
                elif isinstance(u, SolarPlant):
                    u.avail = np.clip(solar_cf[:T], 0.0, 1.0)

            # ── Gross demand profile (temperature-scaled, no RES subtraction) ─
            base_synth = day_gens[(r, day)].base
            if len(base_synth) < T:
                base_synth_T = np.tile(base_synth, int(np.ceil(T / len(base_synth))))[:T]
            else:
                base_synth_T = base_synth[:T]

            # Gross demand = base shape × temperature scaling.
            # No wind/solar subtraction — RES are explicit assets.
            #
            # ERA5 temperature already captures cold snaps fully:
            # at -15°C, temp_scale = 1 + 0.006×30 = 1.18 (18% above base).
            # Adding a synthetic ±10% cold-snap stress on top double-counts
            # the cold signal and pushes demand above the physical gross peak.
            # Fix: reduce stress magnitudes to ±4% — representing residual
            # uncertainty (economic activity, behavioural) not temperature.
            base_gross = base_synth_T * t_sc

            # Hard cap on base and stress profiles.
            # DemandGenerator adds AR1 noise (std≈1.6%) on top of these profiles,
            # Hard cap: 1.05× gross peak.
            # ERA5 cold-snap week (day 8-14) reliably pushes demand to the cap.
            # At 1.08× PL demand reached 23,738 MW — only 22 MW below ceiling —
            # causing scarcity when the GA hadn't pre-committed all OCGT.
            # 1.05× gives 23,100 MW for PL, comfortably below fleet capacity.
            peak_cap   = DEMAND_PEAKS[r] * 1.05
            base_gross = np.minimum(base_synth_T * t_sc, peak_cap)
            mild    = np.minimum(base_gross * (1.0 - 0.04), peak_cap)
            cold    = np.minimum(base_gross * (1.0 + 0.04), peak_cap)
            doldrum = np.minimum(base_gross * (1.0 + 0.04), peak_cap)

            gen_window = DemandGenerator(
                base_profile    = base_gross,
                stress_profiles = [mild, cold, doldrum],
                stress_prob     = [0.15, 0.05, 0.03],
                noise_std       = INNOV_STD_D * 0.5,  # halved: keeps AR1 tail inside peak_cap
                ar1             = PHI_DEMAND,
                wind_sensitivity  = 0.0,   # wind is an explicit asset now
                solar_sensitivity = 0.0,
                allow_negative    = False,  # gross demand is always positive
            )

            hydro_g = hydro_gens.get(r, None)
            return build_scenario_bank(
                region_generators = {r: gen_window},
                S                 = 1 if config.DETERMINISTIC else N_SCENARIOS,
                T                 = T,
                seed              = 42 + window_idx * 100,
                fuel_generator    = None if config.DETERMINISTIC else fuel_gen,
                hydro_generator   = None if config.DETERMINISTIC else hydro_g,
            )
        return fn

    return {r: make_fn(r) for r in REGIONS}



# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def check_capacity(fleet):
    """Print capacity vs residual demand diagnostics for all 16 regions."""
    from collections import Counter
    print("\n  Capacity check:")
    print(f"  {'R':>4}  {'Valley':>7}  {'ActualCap':>10}  {'Ratio':>6}  "
          f"{'Free':>5}  {'Fixed':>5}  {'Status':>8}")
    for r in REGIONS:
        res    = RESIDUAL_PROFILES[r]
        valley = float(res.min())
        actcap = sum(u.max_cap for u in fleet.thermals(r) + fleet.hydros(r))
        ratio  = actcap / res.max()
        n_free = len(fleet.free_units(r))
        n_fix  = len(fleet.fixed_units(r))
        status = "✓" if 1.6 <= ratio <= 2.4 else ("↑ high" if ratio > 2.4 else "↓ low")
        print(f"  {r:>4}  {valley:>7,.0f}  {actcap:>10,}  {ratio:>5.2f}x  "
              f"{n_free:>5}  {n_fix:>5}  {status}")

    # Duplicate name check
    all_names = [u.name for u in fleet.free_units() + fleet.fixed_units()]
    counts = Counter(all_names)
    dups = {k: v for k, v in counts.items() if v > 1}
    if dups:
        print(f"\n  *** DUPLICATE UNIT NAMES: {dups} ***")
    else:
        print(f"\n  No duplicate unit names ✓")

    total_free = sum(len(fleet.free_units(r)) for r in REGIONS)
    print(f"\n  Total free units: {total_free} × {WINDOW_HOURS}h "
          f"= {total_free*WINDOW_HOURS:,} vars/window")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 78)
    print("  SUCCES — January 2024  (15 regions, 744h, calendar-aware demand)")
    print("=" * 78)

    fleet   = build_fleet()
    network = build_network()

    get_bridge(verbose=True)
    check_capacity(fleet)

    total_free = sum(len(fleet.free_units(r)) for r in REGIONS)
    total_fix  = sum(len(fleet.fixed_units(r)) for r in REGIONS)
    print(f"\n  Fleet: {len(fleet)} total assets | "
          f"{total_free} free + {total_fix} fixed | "
          f"{total_free*WINDOW_HOURS:,} vars/window (GA: {total_free*48:,})\n")
    for r in sorted(fleet.regions()):
        print(f"    {r}: {len(fleet.thermals(r))} thermal, {len(fleet.hydros(r))} hydro, "
              f"{len(fleet.storages(r))} storage  "
              f"[{len(fleet.free_units(r))} free + {len(fleet.fixed_units(r))} fixed]")

    print(f"\n  Network: {len(network.links())} links | "
          f"Scenarios: {N_SCENARIOS} | GA: {GA_EPOCHS}×{GA_POP_SIZE}")
    print(f"  Calendar: Jan 2024, {TOTAL_HOURS}h ({TOTAL_HOURS//24} windows)")
    print(f"  Weekends: {[d+1 for d in range(31) if IS_WEEKEND[d]]} (days of month)")
    era5_status = "REAL ERA5 Jan 2024" if not _ERA5_SYNTHETIC_FLAG else "SYNTHETIC (ERA5 not downloaded)"
    print(f"  Weather:  {era5_status}\n")

    solver = CoupledRollingHorizonSolver(
        fleet=fleet, scenario_fn_map=build_scenario_fns(fleet),
        network=network, ga_epochs=GA_EPOCHS, ga_pop_size=GA_POP_SIZE,
        lambda_risk=0.15, seed=42, verbose=True,
    )

    out_path = Path(__file__).parent / "results_europe_jan2024"
    results  = solver.run(total_hours=TOTAL_HOURS, window_hours=WINDOW_HOURS,
                          report_path=str(out_path) + ".html")
    results.print_summary()


if __name__ == "__main__":
    main()
