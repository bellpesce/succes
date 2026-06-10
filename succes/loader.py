"""
succes/loader.py
----------------
Load fleet definitions from YAML and timeseries profiles from Parquet or CSV.

Two entry points
----------------
load_fleet(yaml_path)
    Reads a YAML file describing plants and returns a Fleet object.
    Identical to building the fleet in Python — all the same asset types
    and parameters are supported.

load_profiles(path)
    Reads a Parquet or CSV file of hourly timeseries.
    Returns a dict {column_name: np.ndarray}.
    Used for demand profiles, availability profiles, inflow profiles, etc.

build_scenario_bank_from_profiles(...)
    High-level factory that combines a loaded fleet with loaded profiles
    to build a ScenarioBank ready for the solver.

YAML structure
--------------
See examples/data/fleet_1region.yaml for a full example.

Minimal plant entry:
    - name: ccgt_berlin
      type: thermal        # thermal | hydro | heat | storage
      region: DE
      max_cap: 500
      min_cap: 150
      fuel_type: gas       # gas | coal | nuclear | oil | hydro | biomass
      base_fuel_cost: 55.0
      startup_cost: 8000
      min_run_hours: 4
      ramp_rate: 60
      fixed_on: false
      avail_profile: null  # or column name in avail.parquet

Timeseries profile files (Parquet / CSV)
-----------------------------------------
Rows = hours (e.g. 8760 for a full year).
Columns = one per plant or region, named to match plant names or region names.

Example demand.parquet columns: DE, FR, NL
Example avail.parquet columns:  ccgt_berlin, peaker_munich, hydro_france

Dependencies
------------
pip install pyyaml pandas pyarrow
These are optional — the core succes package works without them.
The loader will raise a clear ImportError if they are missing.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Union
import numpy as np


# ── Lazy imports — only required if loader is used ────────────────────────────

def _require_yaml():
    try:
        import yaml
        return yaml
    except ImportError:
        raise ImportError(
            "pyyaml is required for YAML loading. "
            "Install it with: pip install pyyaml"
        )

def _require_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        raise ImportError(
            "pandas is required for profile loading. "
            "Install it with: pip install pandas pyarrow"
        )


# ── Profile loader ────────────────────────────────────────────────────────────

def load_profiles(path: Union[str, Path]) -> dict[str, np.ndarray]:
    """
    Load hourly timeseries from a Parquet or CSV file.

    Parameters
    ----------
    path : path to .parquet or .csv file

    Returns
    -------
    dict {column_name: np.ndarray of shape (n_hours,)}

    Parquet is strongly preferred for files with many columns (plants):
    - 1000-plant CSV at 8760 rows: ~5-15 seconds to read
    - 1000-plant Parquet:          ~0.2-0.5 seconds

    To convert an existing CSV to Parquet once:
        import pandas as pd
        pd.read_csv("profiles.csv").to_parquet("profiles.parquet")
    """
    pd   = _require_pandas()
    path = Path(path)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in (".csv", ".tsv"):
        sep = "\t" if path.suffix == ".tsv" else ","
        df  = pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use .parquet or .csv")

    return {col: df[col].to_numpy(dtype=float) for col in df.columns}


def save_profiles(profiles: dict[str, np.ndarray], path: Union[str, Path]) -> None:
    """
    Save a dict of profiles to Parquet or CSV.

    Parameters
    ----------
    profiles : {column_name: np.ndarray}
    path     : output path (.parquet recommended)
    """
    pd   = _require_pandas()
    path = Path(path)
    df   = pd.DataFrame(profiles)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


# ── YAML fleet loader ─────────────────────────────────────────────────────────

def load_fleet(
    yaml_path:     Union[str, Path],
    avail_profiles: Optional[dict[str, np.ndarray]] = None,
) -> "Fleet":
    """
    Load a Fleet from a YAML file.

    Parameters
    ----------
    yaml_path      : path to fleet YAML file
    avail_profiles : optional dict of availability arrays loaded from Parquet/CSV.
                     Keys must match the `avail_profile` field in the YAML.
                     If None, all plants use flat availability = 1.0.

    Returns
    -------
    Fleet object — identical to one built in Python.
    """
    yaml = _require_yaml()

    # Import here to avoid circular import
    from .assets import (
        Fleet, ThermalPlant, HydroPlant, HeatPlant, StorageAsset, FuelType
    )

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    fleet = Fleet()

    for entry in data.get("plants", []):
        # Resolve availability profile
        avail = None
        profile_key = entry.get("avail_profile")
        if profile_key and avail_profiles and profile_key in avail_profiles:
            avail = avail_profiles[profile_key]

        plant_type = entry.get("type", "thermal").lower()

        if plant_type == "thermal":
            fleet.add(ThermalPlant(
                name           = entry["name"],
                region         = entry["region"],
                max_cap        = float(entry.get("max_cap", 0)),
                min_cap        = float(entry.get("min_cap", 0)),
                fuel_type      = FuelType(entry.get("fuel_type", "gas")),
                base_fuel_cost = float(entry.get("base_fuel_cost", 50.0)),
                heat_rate      = float(entry.get("heat_rate", 0.0)),
                startup_cost   = float(entry.get("startup_cost", 0.0)),
                min_run_hours  = int(entry.get("min_run_hours", 1)),
                ramp_rate      = float(entry.get("ramp_rate", 0.0)),
                fixed_on       = bool(entry.get("fixed_on", False)),
                co2_intensity  = float(entry.get("co2_intensity", 0.0)),
                avail          = avail,
            ))

        elif plant_type == "hydro":
            fleet.add(HydroPlant(
                name               = entry["name"],
                region             = entry["region"],
                max_cap            = float(entry.get("max_cap", 0)),
                min_cap            = float(entry.get("min_cap", 0)),
                reservoir_capacity = float(entry.get("reservoir_capacity", 1000.0)),
                initial_reservoir  = float(entry.get("initial_reservoir", 500.0)),
                inflow_mwh         = float(entry.get("inflow_mwh", 10.0)),
                water_value        = float(entry.get("water_value", 30.0)),
                startup_cost       = float(entry.get("startup_cost", 0.0)),
                min_run_hours      = int(entry.get("min_run_hours", 1)),
                ramp_rate          = float(entry.get("ramp_rate", 0.0)),
                avail              = avail,
            ))

        elif plant_type == "heat":
            fleet.add(HeatPlant(
                name                = entry["name"],
                region              = entry["region"],
                max_cap             = float(entry.get("max_cap", 0)),
                min_cap             = float(entry.get("min_cap", 0)),
                fuel_type           = FuelType(entry.get("fuel_type", "gas")),
                base_fuel_cost      = float(entry.get("base_fuel_cost", 50.0)),
                heat_rate           = float(entry.get("heat_rate", 0.0)),
                startup_cost        = float(entry.get("startup_cost", 0.0)),
                min_run_hours       = int(entry.get("min_run_hours", 1)),
                ramp_rate           = float(entry.get("ramp_rate", 0.0)),
                power_to_heat_ratio = float(entry.get("power_to_heat_ratio", 1.0)),
                heat_revenue        = float(entry.get("heat_revenue", 30.0)),
                avail               = avail,
            ))

        elif plant_type == "storage":
            fleet.add(StorageAsset(
                name                 = entry["name"],
                region               = entry["region"],
                max_cap              = float(entry.get("max_cap", 25)),
                energy_capacity      = float(entry.get("energy_capacity", 100)),
                charge_rate          = float(entry.get("charge_rate", 25)),
                discharge_rate       = float(entry.get("discharge_rate", 25)),
                charge_efficiency    = float(entry.get("charge_efficiency", 0.92)),
                discharge_efficiency = float(entry.get("discharge_efficiency", 0.92)),
                initial_soc          = float(entry.get("initial_soc", 50)),
                marginal_cost        = float(entry.get("marginal_cost", 5.0)),
                avail                = avail,
            ))
        else:
            raise ValueError(
                f"Unknown plant type '{plant_type}' for plant '{entry.get('name')}'."
                f" Must be one of: thermal, hydro, heat, storage"
            )

    return fleet


def load_network(yaml_path: Union[str, Path]) -> "Network":
    """
    Load a Network from YAML.

    YAML structure:
        links:
          - region_a: DE
            region_b: FR
            max_mw_ab: 3000
            max_mw_ba: 2500
            loss_factor: 0.015
    """
    yaml = _require_yaml()
    from .network import Network, TransmissionLink

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    net = Network()
    for entry in data.get("links", []):
        net.add_link(TransmissionLink(
            region_a    = entry["region_a"],
            region_b    = entry["region_b"],
            max_mw_ab   = float(entry.get("max_mw_ab", 0)),
            max_mw_ba   = float(entry.get("max_mw_ba", entry.get("max_mw_ab", 0))),
            loss_factor = float(entry.get("loss_factor", 0.0)),
        ))
    return net


# ── Scenario bank builder from loaded profiles ────────────────────────────────

def build_scenario_bank_from_profiles(
    demand_profiles:  dict[str, np.ndarray],   # {region: (n_hours,)}
    region_names:     list[str],
    hour_start:       int,
    T:                int,
    S:                int,
    seed:             int                              = 42,
    noise_std:        float                            = 0.02,
    fuel_profiles:    Optional[dict[str, np.ndarray]] = None,  # {fuel: (n_hours,)}
    fuel_names:       Optional[list[str]]              = None,
    surplus_profiles: Optional[dict[str, np.ndarray]] = None,  # {region: (n_hours,)}
    allow_negative:   bool                             = True,
) -> "ScenarioBank":
    """
    Build a ScenarioBank for hours [hour_start, hour_start+T) using
    loaded profiles as the base demand and fuel prices.

    The loaded profile provides the deterministic base shape.
    Multiplicative noise (N(1, noise_std)) is added per scenario
    to produce S distinct realisations.

    Parameters
    ----------
    demand_profiles  : {region: array of shape (n_hours,)} — base demand
    region_names     : which regions to include (must be keys in demand_profiles)
    hour_start       : first hour of the window in the annual profile
    T                : window length (hours)
    S                : number of scenarios
    seed             : random seed
    noise_std        : hourly noise standard deviation
    fuel_profiles    : optional {fuel_name: (n_hours,)} deterministic fuel prices
    fuel_names       : list of fuel names (must match keys in fuel_profiles)
    surplus_profiles : optional {region: (n_hours,)} renewable surplus to subtract
    allow_negative   : if False, demand is clipped to 0

    Returns
    -------
    ScenarioBank ready for the solver
    """
    from .scenarios import ScenarioBank

    rng = np.random.default_rng(seed)
    h0  = hour_start
    h1  = hour_start + T

    # Demand: (S, T, R)
    demand_arr = np.zeros((S, T, len(region_names)))
    for r_idx, r in enumerate(region_names):
        base = demand_profiles[r][h0:h1]
        if surplus_profiles and r in surplus_profiles:
            base = base - surplus_profiles[r][h0:h1]
        noise        = rng.normal(1.0, noise_std, size=(S, T))
        demand_arr[:, :, r_idx] = noise * base[np.newaxis, :]
        if not allow_negative:
            demand_arr[:, :, r_idx] = np.maximum(0.0, demand_arr[:, :, r_idx])

    # Fuel prices: (S, T, F) — deterministic profile + small noise
    fuel_arr  = None
    fnames    = fuel_names or []
    if fuel_profiles and fnames:
        fuel_arr = np.zeros((S, T, len(fnames)))
        for f_idx, f in enumerate(fnames):
            if f in fuel_profiles:
                base_f = fuel_profiles[f][h0:h1]
                fnoise = rng.normal(1.0, noise_std * 0.5, size=(S, T))
                fuel_arr[:, :, f_idx] = np.maximum(
                    0.01 * base_f, fnoise * base_f[np.newaxis, :]
                )

    return ScenarioBank(
        demand       = demand_arr,
        fuel_prices  = fuel_arr,
        region_names = region_names,
        fuel_names   = fnames,
    )


# ── Generate example data files (called by toy_from_yaml.py) ─────────────────

def generate_example_data(output_dir: Union[str, Path]) -> None:
    """
    Write example YAML and Parquet files to output_dir.
    Call this once to set up the examples/data/ directory.
    """
    pd   = _require_pandas()
    yaml = _require_yaml()
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── fleet_1region.yaml ────────────────────────────────────────────────────
    fleet_1r = {
        "plants": [
            {
                "name": "nuclear_de", "type": "thermal", "region": "DE",
                "max_cap": 800, "min_cap": 800, "fuel_type": "nuclear",
                "base_fuel_cost": 12.0, "startup_cost": 0, "min_run_hours": 24,
                "ramp_rate": 0, "fixed_on": True, "co2_intensity": 0.0,
                "avail_profile": None,
            },
            {
                "name": "ccgt_de", "type": "thermal", "region": "DE",
                "max_cap": 500, "min_cap": 150, "fuel_type": "gas",
                "base_fuel_cost": 55.0, "heat_rate": 1.8, "startup_cost": 8000,
                "min_run_hours": 4, "ramp_rate": 60, "fixed_on": False,
                "co2_intensity": 0.35, "avail_profile": "ccgt_de",
            },
            {
                "name": "peaker_de", "type": "thermal", "region": "DE",
                "max_cap": 150, "min_cap": 20, "fuel_type": "gas",
                "base_fuel_cost": 90.0, "startup_cost": 1500,
                "min_run_hours": 1, "ramp_rate": 150, "fixed_on": False,
                "co2_intensity": 0.45, "avail_profile": None,
            },
            {
                "name": "battery_de", "type": "storage", "region": "DE",
                "max_cap": 25, "energy_capacity": 100, "charge_rate": 25,
                "discharge_rate": 25, "charge_efficiency": 0.92,
                "discharge_efficiency": 0.92, "initial_soc": 50,
                "marginal_cost": 4.0,
            },
        ]
    }
    with open(out / "fleet_1region.yaml", "w") as f:
        yaml.dump(fleet_1r, f, default_flow_style=False, sort_keys=False)

    # ── fleet_2region.yaml ────────────────────────────────────────────────────
    fleet_2r = {
        "plants": [
            # DE
            {
                "name": "ccgt_de", "type": "thermal", "region": "DE",
                "max_cap": 500, "min_cap": 150, "fuel_type": "gas",
                "base_fuel_cost": 55.0, "heat_rate": 1.8, "startup_cost": 8000,
                "min_run_hours": 4, "ramp_rate": 60, "fixed_on": False,
                "co2_intensity": 0.35, "avail_profile": "ccgt_de",
            },
            {
                "name": "peaker_de", "type": "thermal", "region": "DE",
                "max_cap": 150, "min_cap": 20, "fuel_type": "gas",
                "base_fuel_cost": 90.0, "startup_cost": 1500,
                "min_run_hours": 1, "ramp_rate": 150, "fixed_on": False,
                "co2_intensity": 0.45,
            },
            {
                "name": "battery_de", "type": "storage", "region": "DE",
                "max_cap": 25, "energy_capacity": 100, "charge_rate": 25,
                "discharge_rate": 25, "charge_efficiency": 0.92,
                "discharge_efficiency": 0.92, "initial_soc": 50,
                "marginal_cost": 4.0,
            },
            # FR
            {
                "name": "nuclear_fr", "type": "thermal", "region": "FR",
                "max_cap": 1200, "min_cap": 1200, "fuel_type": "nuclear",
                "base_fuel_cost": 10.0, "startup_cost": 0, "min_run_hours": 24,
                "ramp_rate": 0, "fixed_on": True, "co2_intensity": 0.0,
            },
            {
                "name": "hydro_fr", "type": "hydro", "region": "FR",
                "max_cap": 400, "min_cap": 0, "reservoir_capacity": 8000,
                "initial_reservoir": 4000, "inflow_mwh": 50.0,
                "water_value": 25.0, "startup_cost": 500,
                "min_run_hours": 2, "ramp_rate": 200,
            },
            {
                "name": "gas_peaker_fr", "type": "thermal", "region": "FR",
                "max_cap": 200, "min_cap": 30, "fuel_type": "gas",
                "base_fuel_cost": 70.0, "startup_cost": 3000,
                "min_run_hours": 2, "ramp_rate": 100, "co2_intensity": 0.40,
            },
        ]
    }
    with open(out / "fleet_2region.yaml", "w") as f:
        yaml.dump(fleet_2r, f, default_flow_style=False, sort_keys=False)

    # ── network_2region.yaml ──────────────────────────────────────────────────
    network_yaml = {
        "links": [
            {
                "region_a": "DE", "region_b": "FR",
                "max_mw_ab": 1500, "max_mw_ba": 1500,
                "loss_factor": 0.015,
            }
        ]
    }
    with open(out / "network_2region.yaml", "w") as f:
        yaml.dump(network_yaml, f, default_flow_style=False, sort_keys=False)

    # ── demand.parquet ────────────────────────────────────────────────────────
    # 72 hours of synthetic demand for DE and FR
    hours = 72
    t     = np.arange(hours)

    # DE: typical day shape repeated, with slight upward drift
    de_shape = np.array([
        840, 800, 760, 740, 750, 800, 910, 1010, 1060, 1080, 1090, 1100,
        1080, 1060, 1050, 1060, 1100, 1140, 1120, 1080, 1020, 980, 940, 900
    ], dtype=float)
    fr_shape = np.array([
        600, 570, 550, 540, 545, 570, 630, 700, 750, 770, 780, 785,
        775, 760, 750, 760, 790, 820, 810, 780, 740, 710, 670, 630
    ], dtype=float)

    de_demand = np.tile(de_shape, int(np.ceil(hours / 24)))[:hours]
    fr_demand = np.tile(fr_shape, int(np.ceil(hours / 24)))[:hours]

    # ── avail.parquet ─────────────────────────────────────────────────────────
    # ccgt_de: planned maintenance hours 20-23 on day 2 (hours 44-47)
    ccgt_avail = np.ones(hours)
    ccgt_avail[44:48] = 0.0   # forced outage window

    # ── solar_surplus.parquet ─────────────────────────────────────────────────
    solar_shape = np.array([
        0, 0, 0, 0, 0, 15, 100, 280, 420, 550, 630, 660,
        640, 570, 480, 360, 210, 80, 15, 0, 0, 0, 0, 0
    ], dtype=float)
    de_solar = np.tile(solar_shape, int(np.ceil(hours / 24)))[:hours]

    # ── gas_price.parquet ─────────────────────────────────────────────────────
    # Slightly trending gas price with daily cycle
    np.random.seed(42)
    gas_price = 35.0 + np.cumsum(np.random.normal(0, 0.3, hours))
    gas_price = np.clip(gas_price, 20.0, 80.0)

    # Save all profile files
    pd.DataFrame({"DE": de_demand, "FR": fr_demand}).to_parquet(
        out / "demand.parquet", index=False
    )
    pd.DataFrame({"ccgt_de": ccgt_avail}).to_parquet(
        out / "avail.parquet", index=False
    )
    pd.DataFrame({"DE": de_solar}).to_parquet(
        out / "solar_surplus.parquet", index=False
    )
    pd.DataFrame({"gas": gas_price}).to_parquet(
        out / "fuel_prices.parquet", index=False
    )

    print(f"Example data written to {out}/")
    print(f"  fleet_1region.yaml   — 1-region fleet (nuclear + ccgt + peaker + battery)")
    print(f"  fleet_2region.yaml   — 2-region fleet (DE: ccgt+peaker+battery, FR: nuclear+hydro+gas)")
    print(f"  network_2region.yaml — DE<->FR link 1500 MW, 1.5% losses")
    print(f"  demand.parquet       — 72h demand profiles for DE and FR")
    print(f"  avail.parquet        — ccgt_de availability (outage h44-47)")
    print(f"  solar_surplus.parquet— DE solar surplus (midday peak)")
    print(f"  fuel_prices.parquet  — 72h gas price path")
