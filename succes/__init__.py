"""
SUCCES — Stochastic Unit Commitment for Competitive Energy Systems
==================================================================
Public API. Import from here rather than from submodules.

Quick start (code-first)
-------------------------
    from succes import Fleet, ThermalPlant, StorageAsset, FuelType
    from succes import DemandGenerator, build_scenario_bank
    from succes import CoupledRollingHorizonSolver, Network, TransmissionLink

Quick start (config-first, from YAML + Parquet)
------------------------------------------------
    from succes.loader import load_fleet, load_network, load_profiles
    from succes.loader import build_scenario_bank_from_profiles
    from succes import CoupledRollingHorizonSolver

See examples/ for full runnable examples.
"""

from .assets import (
    Fleet,
    Asset,
    ThermalPlant,
    HydroPlant,
    HeatPlant,
    StorageAsset,
    FuelType,
)
from .scenarios import (
    ScenarioBank,
    DemandGenerator,
    FuelPriceGenerator,
    build_scenario_bank,
)
from .simulator import (
    simulate_window,
    simulate_coupled,
    compute_cvar,
    CarryoverState,
    build_objective,
    build_coupled_objective,
    must_run_soft_penalty,
)
from .network import Network, TransmissionLink
from .solver import (
    RollingHorizonSolver,
    CoupledRollingHorizonSolver,
    Results,
    WindowResult,
)
from .julia_bridge import JuliaBridge, get_bridge
from .reporter import generate_report
from . import config

__version__ = "0.2.0"
__all__ = [
    # assets
    "Fleet", "Asset", "ThermalPlant", "HydroPlant", "HeatPlant",
    "StorageAsset", "FuelType",
    # scenarios
    "ScenarioBank", "DemandGenerator", "FuelPriceGenerator",
    "build_scenario_bank",
    # simulator
    "simulate_window", "simulate_coupled", "compute_cvar",
    "CarryoverState", "build_objective", "build_coupled_objective",
    "must_run_soft_penalty",
    # network
    "Network", "TransmissionLink",
    # solver
    "RollingHorizonSolver", "CoupledRollingHorizonSolver",
    "Results", "WindowResult",
    # julia bridge
    "JuliaBridge", "get_bridge",
    # reporter
    "generate_report",
    # config
    "config",
]
