"""
succes/julia_bridge.py
----------------------
Python ↔ Julia interface for the SUCCES simulation engine.

Two execution modes:
────────────────────
1. juliacall  (pip install juliacall, Julia already installed separately)
   In-process Julia. JIT-compiled once on first call, stays warm for all
   subsequent windows. No JSON, no subprocesses — data lives in memory.

   IMPORTANT: juliacall will NOT download Julia automatically. You must
   have Julia installed and on PATH before using this mode. Install from
   https://julialang.org/downloads/ or via winget/brew/apt.

2. Python (always available)
   Pure-Python simulate_coupled(). No Julia needed.

Mode selection (priority order):
  1. SUCCES_ENGINE env var: set to "juliacall" or "python"
  2. SUCCES_FORCE_PYTHON=1 → always Python
  3. Auto: try juliacall, fall back to Python instantly if unavailable
"""

from __future__ import annotations

import logging
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .assets import Fleet, ThermalPlant, HydroPlant, HeatPlant, StorageAsset, FuelType, WindPlant, SolarPlant
from .scenarios import ScenarioBank
from .network import Network
from .simulator import CarryoverState, simulate_coupled
from . import config

log = logging.getLogger(__name__)
_ENGINE_PATH = Path(__file__).parent / "engine.jl"


# ── Mode detection ─────────────────────────────────────────────────────────────

def _detect_mode() -> str:
    env = os.environ.get("SUCCES_ENGINE", "").lower().strip()
    if env in ("juliacall", "python"):
        return env
    if os.environ.get("SUCCES_FORCE_PYTHON", "0") == "1":
        return "python"
    if _juliacall_usable():
        return "juliacall"
    return "python"


def _juliacall_usable() -> bool:
    """
    Return True if juliacall is installed and Julia is reachable.
    juliacall manages its own Julia via juliapkg — it does NOT need
    julia on PATH. We just check the package is importable without
    triggering a fresh install (JULIAPKG_OFFLINE=true means: use what
    is already provisioned, error if nothing is there yet).
    """
    try:
        import importlib.util
        if importlib.util.find_spec("juliacall") is None:
            return False
        return True
    except Exception:
        return False


# ── juliacall singleton ────────────────────────────────────────────────────────

_jl_engine = None


def _get_juliacall_engine():
    """Load engine.jl into Julia once; module is cached for JIT-warm reuse."""
    global _jl_engine
    if _jl_engine is not None:
        return _jl_engine

    import juliacall
    jl = juliacall.newmodule("SuccesHost")
    engine_path_str = str(_ENGINE_PATH).replace("\\", "/")
    jl.seval(f'include("{engine_path_str}")')
    jl.seval("import .SuccesEngine")
    _jl_engine = jl.SuccesEngine
    return _jl_engine


# ── JuliaBridge ────────────────────────────────────────────────────────────────

class JuliaBridge:
    """
    Routes simulation calls to Julia (juliacall) or the Python fallback.

    Parameters
    ----------
    mode        : "auto" | "juliacall" | "python"
    flow_passes : ATC redistribution passes (2 is enough for most networks)
    verbose     : print mode and per-window timing
    """

    _LABELS = {
        "juliacall": "juliacall (in-process, JIT-warm) ✓",
        "python":    "Python simulator (Julia not available or not requested)",
    }

    def __init__(self, mode: str = "auto", flow_passes: int = 2, verbose: bool = True):
        self.flow_passes = flow_passes
        self.verbose     = verbose

        if mode == "auto":
            self.mode = _detect_mode()
        elif mode in ("juliacall", "python"):
            self.mode = mode
        else:
            raise ValueError(f"Unknown mode {mode!r}. Use 'auto', 'juliacall', or 'python'.")

        if self.verbose:
            print(f"  [bridge] {self._LABELS[self.mode]}")

        # Pre-warm Julia on construction so the first GA window isn't slow
        if self.mode == "juliacall":
            try:
                if self.verbose:
                    print("  [bridge] Loading Julia engine (first time: JIT compile ~20s)…",
                          flush=True)
                t0 = time.perf_counter()
                _get_juliacall_engine()
                if self.verbose:
                    print(f"  [bridge] Julia ready ({time.perf_counter()-t0:.1f}s)", flush=True)
            except Exception as e:
                log.warning("juliacall init failed (%s) — falling back to Python.", e)
                self.mode = "python"
                if self.verbose:
                    print(f"  [bridge] juliacall failed ({e})\n"
                          f"           Is Julia installed and on PATH? "
                          f"(https://julialang.org/downloads)\n"
                          f"           Continuing with Python simulator.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def simulate_coupled(
        self,
        commitments:  dict[str, np.ndarray],
        fleet:        Fleet,
        scenario_map: dict[str, ScenarioBank],
        network:      Network,
        carryover:    Optional[CarryoverState] = None,
        flow_passes:  int = None,
    ) -> dict:
        """Drop-in for simulator.simulate_coupled(). Julia owns all computation."""
        if flow_passes is None:
            flow_passes = config.FLOW_PASSES if hasattr(config, 'FLOW_PASSES') else 3
        if self.mode == "python":
            return simulate_coupled(
                commitments, fleet, scenario_map, network,
                carryover=carryover, flow_passes=flow_passes,
            )

        t0      = time.perf_counter()
        payload = self._build_payload(
            commitments, fleet, scenario_map, network, carryover, flow_passes
        )
        raw    = self._call_juliacall(payload)
        result = self._parse_result(raw, fleet, scenario_map, network)

        if self.verbose:
            S = next(iter(scenario_map.values())).S
            print(f"  [bridge] Julia dispatch: {time.perf_counter()-t0:.2f}s "
                  f"S={S} R={len(scenario_map)}", flush=True)
        return result

    # ── juliacall call ─────────────────────────────────────────────────────────

    def _call_juliacall(self, payload: dict) -> dict:
        engine = _get_juliacall_engine()
        try:
            import juliacall
            jl_payload = juliacall.convert(
                juliacall.Main.seval("Dict{Any,Any}"), payload
            )
        except Exception:
            jl_payload = payload   # juliacall may handle it natively on some versions
        jl_result = engine.run_simulation(jl_payload)
        # Extract flows_mean BEFORE _jl_to_python — Julia Matrix{Float64} is
        # column-major; _jl_to_python iterates columns giving wrong shape.
        # np.array on a juliacall matrix reads it correctly (Fortran order).
        flows_raw = None
        try:
            jl_flows = jl_result["flows"]
            if jl_flows is not None:
                import numpy as _np
                fm = _np.array(jl_flows["flows_mean"])   # Julia (T,L) col-major → numpy (L,T)
                np_m = _np.array(jl_flows["net_pos_mean"])  # Julia (T,R) col-major → numpy (R,T)
                # Julia column-major Matrix{T}(T_rows, L_cols): numpy reads as (L, T)
                # We want (T, L) — just transpose
                flows_raw = {"flows_mean": fm.T if fm.ndim==2 else fm,
                             "net_pos_mean": np_m.T if np_m.ndim==2 else np_m}
        except Exception:
            flows_raw = None
        result = _jl_to_python(jl_result)
        if flows_raw is not None:
            result["flows"] = flows_raw
        return result

    # ── Payload builder ────────────────────────────────────────────────────────

    @staticmethod
    def _hydro_mult_for_static(r, bank, hydros):
        """Extract per-scenario hydro inflow multipliers for region r."""
        if (bank is None or not hasattr(bank, 'hydro_inflow')
                or bank.hydro_inflow is None or len(hydros) == 0):
            return []
        hydro_names = [h.name for h in hydros]
        bank_names  = getattr(bank, 'hydro_unit_names', [])
        if not bank_names:
            return []
        # (S, H) slice for units in this region
        result = []
        for hname in hydro_names:
            if hname in bank_names:
                idx = bank_names.index(hname)
                result.append(bank.hydro_inflow[:, idx].tolist())
            else:
                result.append([])   # unit not in bank → use baseline
        return result  # list of S-length lists, one per hydro unit

    @staticmethod
    def _compute_res_net_demand(
        res_units: list,
        gross_demand: np.ndarray,   # (S, T)
        T: int,
    ) -> tuple:
        """
        Compute RES output (deterministic) and subtract from gross demand.

        RES DISPATCH RULE (Approach A — pro-rata curtailment by threshold):
        ────────────────────────────────────────────────────────────────────
        Each RES unit dispatches at its full available capacity:
            output[t] = max_cap × avail[t]   (MW)

        when mean gross demand >= total RES output.  If there is a surplus,
        units are curtailed in descending order of curtailment_threshold
        (highest threshold = cheapest to curtail = curtail first), pro-rata
        within each tier.

        curtailment_threshold semantics:
          - 0 EUR/MWh  : unsubsidised merchant plant — curtail when MCP → 0
          - negative   : subsidised (FIT/CfD) — keeps running at negative prices,
                         curtailed only when price falls below the threshold
          Default: 0 EUR/MWh (conservative; scenario file can override per unit)

        Returns
        ───────
        net_demand  : (S, T) net demand for the thermal GA dispatch engine
        res_output  : (T,) deterministic RES MW after curtailment (for reporting)
        """
        S, T_full = gross_demand.shape
        T = min(T, T_full)

        if not res_units:
            return gross_demand[:, :T].copy(), np.zeros(T)

        # Available output per unit per hour: (T, N_res)
        avail_mw = np.stack(
            [u.max_cap * u.availability(T) for u in res_units], axis=1
        )
        total_res_full = avail_mw.sum(axis=1)   # (T,)

        # Threshold per unit — array shape (N_res,)
        thresh = np.array([
            float(getattr(u, "curtailment_threshold", 0.0)) for u in res_units
        ])

        # Use mean gross demand as the curtailment signal.
        # We curtail when total RES > mean demand to avoid clipping away
        # RES that is genuinely needed in high-demand scenarios.
        gross_mean = gross_demand[:, :T].mean(axis=0)   # (T,)

        # dispatch_fraction[t, i] ∈ [0, 1] per unit per hour
        dispatch_fraction = np.ones((T, len(res_units)))

        # Sort units by descending threshold (curtail high-threshold first)
        sort_idx = np.argsort(-thresh)

        for t in range(T):
            surplus = total_res_full[t] - gross_mean[t]
            if surplus <= 1.0:
                continue
            remaining = surplus
            for i in sort_idx:
                if remaining <= 0.5:
                    break
                avail_i = avail_mw[t, i]
                if avail_i < 0.5:
                    continue
                curtail_i = min(avail_i, remaining)
                dispatch_fraction[t, i] = 1.0 - curtail_i / avail_i
                dispatch_fraction[t, i] = max(0.0, dispatch_fraction[t, i])
                remaining -= curtail_i

        res_output = (avail_mw * dispatch_fraction).sum(axis=1)   # (T,)

        # Net demand: gross - RES output, clipped at 0
        # (negative net demand = curtailed surplus; MCP handled by engine)
        net_d = np.maximum(gross_demand[:, :T] - res_output[np.newaxis, :], 0.0)
        return net_d, res_output

    def _build_payload(
        self,
        commitments:  dict[str, np.ndarray],
        fleet:        Fleet,
        scenario_map: dict[str, ScenarioBank],
        network:      Network,
        carryover:    Optional[CarryoverState],
        flow_passes:  int,
    ) -> dict:
        """
        Build the Julia payload dict — Approach A: RES as net-load pre-subtraction.

        Wind and solar are NOT GA genes.  Their output is computed from ERA5
        availability profiles and subtracted from gross demand.  The GA dispatches
        thermal + hydro units to cover the residual (net) demand only.

        Fixes in this version vs. previous:
        ─────────────────────────────────────
        1. RES removed from GA chromosome  → smaller search space, cleaner fitness signal
        2. export_avail in coupling pre-pass uses actual RES-subtracted headroom,
           not flat nameplate → no phantom-solar inflation of coupling signals at night
        3. Demand passed to Julia is net demand → Julia MCP logic works correctly:
           hours with net_demand ≈ 0 price at nuclear/hydro offer price (can be negative);
           hours with net_demand > 0 price at marginal thermal unit
        """
        co      = carryover or CarryoverState()
        regions = sorted(fleet.regions())
        S       = next(iter(scenario_map.values())).S
        T       = next(iter(scenario_map.values())).T
        rdata: dict = {}
        self._last_res_output: dict = {}   # store for _parse_result (reporting)

        for r in regions:
            bank     = scenario_map[r]
            res_u    = fleet.res_units(r)
            # GA chromosome units = thermal + hydro only (no wind/solar)
            free_u   = [u for u in fleet.free_units(r)
                        if not isinstance(u, (WindPlant, SolarPlant))]
            fixed_u  = fleet.fixed_units(r)
            all_u    = fixed_u + free_u
            storages = fleet.storages(r)
            hydros   = fleet.hydros(r)
            N        = len(all_u)

            # ── RES pre-subtraction ────────────────────────────────────────────
            gross_demand = bank.demand_for_region(r)           # (S, T)
            net_demand, res_output = self._compute_res_net_demand(
                res_u, gross_demand, T
            )
            self._last_res_output[r] = res_output              # (T,) for reporting

            # ── Commitment matrix (thermal + hydro only) ──────────────────────
            fixed_com = np.ones((len(fixed_u), T), dtype=float)
            free_com  = commitments.get(r, np.zeros((len(free_u), T), dtype=float))
            if len(fixed_u) and len(free_u):
                commitment = np.vstack([fixed_com, free_com])
            elif len(fixed_u):
                commitment = fixed_com
            elif len(free_u):
                commitment = free_com
            else:
                commitment = np.zeros((0, T), dtype=float)

            # Thermal/hydro availability (planned outage, forced outage)
            avail = np.array([u.availability(T) for u in all_u])
            if commitment.shape[0]:
                commitment = commitment * avail

            from . import config as _cfg
            for_stochastic = bool(_cfg.FOR_STOCHASTIC)

            fc_mat    = self._build_fuel_cost_matrix(all_u, bank, S)
            prev_disp = np.zeros((S, N), dtype=float)
            prev_on_s = np.zeros((S, N), dtype=float)
            for i, u in enumerate(all_u):
                was_on = 1.0 if co.get_plant_on(u.name, True) else 0.0
                if was_on < 0.5:
                    default_disp = 0.0
                else:
                    rr = float(getattr(u, "ramp_rate", 0.0))
                    mc = float(u.max_cap)
                    if rr > 0 and rr / mc < 0.10:
                        default_disp = float(getattr(u, "min_cap", 0.0))
                    else:
                        default_disp = mc * 0.7
                prev_disp[:, i] = co.get_last_dispatch(u.name, default_disp)
                prev_on_s[:, i] = 1.0 if co.get_plant_on(u.name, True) else 0.0

            must_run = []
            for u in all_u:
                if getattr(u, "fixed_on", False):
                    must_run.append(True)
                elif hasattr(u, "must_run_heat") and u.must_run_heat is not None:
                    must_run.append(bool(u.heat_obligation(T).max() > 0))
                else:
                    must_run.append(False)

            hy_uidx: dict = {}
            for h in hydros:
                for idx, u in enumerate(all_u):
                    if u.name == h.name:
                        hy_uidx[h.name] = idx + 1
                        break

            rdata[r] = {
                # net demand after RES subtraction — the GA dispatches against this
                "demand":                 net_demand.tolist(),
                "fuel_cost_matrix":       fc_mat.tolist(),
                "min_caps":               [float(getattr(u, "min_cap", 0.0)) for u in all_u],
                "max_caps":               [float(u.max_cap) for u in all_u],
                "ramp_rates":             [float(getattr(u, "ramp_rate", 0.0)) for u in all_u],
                "fixed_units_commit":     fixed_com.tolist(),
                "commitment":             commitment.tolist(),
                "prev_dispatch":          prev_disp.tolist(),
                "prev_on":                prev_on_s.tolist(),
                "startup_cost_vec":       [float(getattr(u, "startup_cost", 0.0)) for u in all_u],
                "prev_on_startup":        [1.0 if co.get_plant_on(u.name, True) else 0.0 for u in all_u],
                "must_run_mask":          [bool(x) for x in must_run],
                "provides_inertia":       [bool(getattr(u, "provides_inertia", True) and
                                           getattr(u, "fuel_type", None) not in ("gas",) or
                                           (hasattr(u, "fuel_type") and
                                            str(u.fuel_type).lower() not in ("gas","none","electric")))
                                          for u in all_u],
                "for_rates":              [float(getattr(u, "forced_outage_rate", 0.0)) for u in all_u],
                "co2_intensity":          [float(getattr(u, "co2_intensity", 0.0)) for u in all_u],
                "inertia_constants":      [
                    float(getattr(u, "inertia_constant", 0.0))
                    for u in all_u
                ],
                # offer_prices: thermal + hydro only (no RES)
                "offer_prices":           [
                    float(u.offer_price)      if getattr(u, "offer_price", None) is not None
                    else float(u.water_value) if isinstance(u, HydroPlant)
                    else float(getattr(u, "base_fuel_cost", 50.0))
                    for u in all_u
                ],
                "storage_names":          [st.name for st in storages],
                "storage_soc":            {st.name: co.get_soc(st.name, S, st.initial_soc).tolist() for st in storages},
                "storage_charge_rate":    {st.name: float(st.charge_rate) for st in storages},
                "storage_discharge_rate": {st.name: float(st.discharge_rate) for st in storages},
                "storage_charge_eff":     {st.name: float(st.charge_efficiency) for st in storages},
                "storage_discharge_eff":  {st.name: float(st.discharge_efficiency) for st in storages},
                "storage_capacity":       {st.name: float(st.energy_capacity) for st in storages},
                "storage_marginal":       {st.name: float(st.marginal_cost) for st in storages},
                "hydro_names":            [h.name for h in hydros],
                "hydro_soc":              {h.name: co.get_reservoir(h.name, S, h.initial_reservoir).tolist() for h in hydros},
                "hydro_inflow":           {h.name: h.inflow_array(T).tolist() for h in hydros},
                "hydro_capacity":         {h.name: float(h.reservoir_capacity) for h in hydros},
                "hydro_unit_idx":         hy_uidx,
                "for_stochastic":         for_stochastic,
                "h_min_seconds":          float(getattr(fleet, "_h_min", {}).get(r, 3.5)),
                "hydro_inflow_mult":      self._hydro_mult_for_static(r, bank, hydros),
                # RES metadata (not consumed by Julia — for _parse_result reporting)
                "res_output":             res_output.tolist(),
                "res_unit_names":         [u.name for u in res_u],
            }

        return {
            "S":           S,
            "T":           T,
            "regions":     regions,
            "region_data": rdata,
            "links": [
                {"region_a": lk.region_a, "region_b": lk.region_b,
                 "atc_ab": float(lk.max_mw_ab), "atc_ba": float(lk.max_mw_ba),
                 "loss_factor": float(lk.loss_factor)}
                for lk in network.links()
            ],
            "flow_passes": flow_passes,
            **network.ptdf_payload(regions),
        }

    # ── Result parser ──────────────────────────────────────────────────────────

    def _parse_result(self, raw, fleet, scenario_map, network) -> dict:
        regions = sorted(fleet.regions())
        regional: dict = {}
        for r in regions:
            jr = raw["regional"][r]

            S_r  = next(iter(scenario_map.values())).S
            T_r  = next(iter(scenario_map.values())).T
            # N_r excludes RES: Julia only has thermal+hydro in its unit list.
            res_u_r = fleet.res_units(r)
            N_r  = (len(fleet.fixed_units(r))
                    + len([u for u in fleet.free_units(r)
                           if not isinstance(u, (WindPlant, SolarPlant))]))
            Ns_r = len(fleet.storages(r))

            # Julia matrices are column-major. A Julia (N, T) matrix arrives in
            # Python as a nested list of T columns each of length N, so
            # np.asarray gives shape (T, N). We must reshape as (T_r, N_r)
            # then transpose to (N_r, T_r).
            prices_raw   = np.asarray(jr["prices"],       dtype=float).reshape(T_r, S_r).T
            dispatch_log = np.asarray(jr["dispatch_log"], dtype=float).reshape(T_r, N_r).T
            net_pos      = np.asarray(jr["net_position"], dtype=float).reshape(T_r, S_r).T
            mean_prices  = np.asarray(jr["mean_prices"],  dtype=float).reshape(T_r)
            st_raw = jr.get("storage_log", [])
            storage_log  = np.asarray(st_raw, dtype=float).reshape(T_r, Ns_r).T if Ns_r > 0 else np.zeros((0, T_r))

            dispatch_mean = np.asarray(jr.get("dispatch_mean", jr["dispatch_log"]),
                                         dtype=float).reshape(T_r, N_r).T

            # Re-inject RES output into dispatch_mean for reporting.
            # Julia never saw these units, so we append their deterministic
            # output (res_output, shape T_r) as extra rows at the end of
            # dispatch_mean.  This keeps the reporter's generation stacks correct.
            res_output_r = getattr(self, "_last_res_output", {}).get(r)
            if res_output_r is not None and len(res_u_r) > 0:
                # Build one row per RES unit proportional to their share of total
                # RES output.  Total output at each hour = sum of all RES units.
                res_total = np.maximum(
                    np.array([u.max_cap * u.availability(T_r) for u in res_u_r]).sum(axis=0),
                    1e-9,
                )
                # Actual dispatched output (after curtailment) = res_output_r
                # Apportion to each unit proportionally to their available output
                res_rows = np.stack([
                    res_output_r * (u.max_cap * u.availability(T_r)) / res_total
                    for u in res_u_r
                ], axis=0)   # (N_res, T_r)
                dispatch_mean_full = np.vstack([dispatch_mean, res_rows])
                dispatch_log_full  = np.vstack([dispatch_log,  res_rows])
            else:
                dispatch_mean_full = dispatch_mean
                dispatch_log_full  = dispatch_log

            regional[r] = {
                "costs":        np.asarray(jr["costs"],         dtype=float),
                "fuel_costs":   np.asarray(jr["fuel_costs"],    dtype=float),
                "co2_emissions":np.asarray(jr.get("co2_emissions", np.zeros(len(jr["costs"]))), dtype=float),
                "startup_costs":np.asarray(jr["startup_costs"], dtype=float),
                "penalty_costs":np.asarray(jr["penalty_costs"], dtype=float),
                "pen_scarcity":   np.asarray(jr.get("pen_scarcity",    [0.0]*len(jr["costs"])), dtype=float),
                "pen_curtailment":np.asarray(jr.get("pen_curtailment", [0.0]*len(jr["costs"])), dtype=float),
                "pen_must_run":   np.asarray(jr.get("pen_must_run",    [0.0]*len(jr["costs"])), dtype=float),
                "pen_inertia":    np.asarray(jr.get("pen_inertia",     [0.0]*len(jr["costs"])), dtype=float),
                "pen_ramp":       np.asarray(jr.get("pen_ramp",        [0.0]*len(jr["costs"])), dtype=float),
                "storage_soc":  {str(k): np.asarray(v, dtype=float)
                                 for k, v in jr["storage_soc"].items()},
                "hydro_res":    {str(k): np.asarray(v, dtype=float)
                                 for k, v in jr["hydro_soc"].items()},
                "dispatch_exp":  dispatch_log_full,    # scenario-1 + RES, for carryover
                "dispatch_mean": dispatch_mean_full,   # mean over S + RES, for reporting
                "storage_log":  storage_log,
                "storage_names":list(jr.get("storage_names", [])),
                "net_position": net_pos,
                "prices":       prices_raw,
                "mean_prices":  mean_prices,
            }
        total_costs = np.asarray(raw["total_costs"], dtype=float)
        flows_out   = None
        if raw.get("flows") is not None:
            fdata = raw["flows"]
            L = len(network.links()) if hasattr(network, "links") else 1
            R_n = len(regions)
            T_n = next(iter(scenario_map.values())).T
            # Reshape robustly — juliacall may flatten singleton dims
            fm = fdata["flows_mean"]
            # If already a numpy array (from pre-extraction), use directly
            if hasattr(fm, 'shape'):
                fmean = fm.astype(float)
                if fmean.shape != (T_n, L):
                    fmean = fmean.reshape(T_n, L)
            else:
                fmean = np.asarray(fm, dtype=float).reshape(T_n, L)
            flows_out = fmean[np.newaxis, :, :]                     # (1, T, L)
            npm = fdata["net_pos_mean"]
            np_mean = npm.astype(float) if hasattr(npm,'shape') else np.asarray(npm, dtype=float).reshape(T_n, R_n)
            if np_mean.shape != (T_n, R_n):
                np_mean = np_mean.reshape(T_n, R_n)
            for ri, r in enumerate(regions):
                regional[r]["net_pos_mean"] = np_mean[:, ri]
        return {"regions": regional, "total_costs": total_costs, "flows": flows_out}

    @staticmethod
    def _build_fuel_cost_matrix(units, bank: ScenarioBank, S: int) -> np.ndarray:
        from .simulator import _build_fuel_cost_matrix
        return _build_fuel_cost_matrix(units, bank, S)


# ── juliacall type converter ───────────────────────────────────────────────────

def _jl_to_python(obj) -> object:
    """Recursively unwrap juliacall proxy objects to plain Python types."""
    if hasattr(obj, "items"):
        return {str(k): _jl_to_python(v) for k, v in obj.items()}
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        try:
            return [_jl_to_python(x) for x in obj]
        except Exception:
            pass
    if hasattr(obj, "__float__"):
        try: return float(obj)
        except Exception: pass
    if hasattr(obj, "__int__"):
        try: return int(obj)
        except Exception: pass
    return obj


# ── Module-level singleton ─────────────────────────────────────────────────────

_default_bridge: Optional[JuliaBridge] = None


def get_bridge(verbose: bool = True, mode: str = "auto") -> JuliaBridge:
    """Get or create the module-level JuliaBridge singleton."""
    global _default_bridge
    if _default_bridge is None:
        _default_bridge = JuliaBridge(mode=mode, verbose=verbose)
    return _default_bridge
