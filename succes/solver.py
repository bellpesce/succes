"""
succes/solver.py
----------------
Two solvers:

RollingHorizonSolver
    Single-region solver. One GA per window optimising only the free
    units of one region. Use this when regions are isolated or when
    you want per-region baselines.

CoupledRollingHorizonSolver
    Multi-region solver. One GA per window optimising the *joint*
    commitment of all regions simultaneously. The objective calls
    simulate_coupled(), which dispatches all regions and then moves
    surplus across transmission links. This means the GA can learn
    that keeping a cheap unit on in FR saves an expensive startup
    in DE — the correct market behaviour.

Parallelism note
----------------
The rolling horizon is *sequential by design* — window N+1 depends on
the CarryoverState produced by window N. Parallelism therefore lives
*inside* each window, across the GA's population evaluations. This is
implemented via concurrent.futures when config.PARALLEL = True.
Each worker evaluates one candidate solution independently (read-only
access to scenarios and carryover). No shared mutable state.
"""

from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable

from mealpy import BinaryVar, GA

from .assets import Fleet, WindPlant, SolarPlant
from .scenarios import ScenarioBank
from .julia_ga import run_julia_batch_ga, _get_batch_engine
from .simulator import (
    simulate_window, simulate_coupled,
    build_objective, build_coupled_objective,
    _repair_min_run, compute_cvar, CarryoverState,
)
from .julia_bridge import JuliaBridge, get_bridge
from .network import Network
from . import config


# ── Window result ─────────────────────────────────────────────────────────────

@dataclass
class WindowResult:
    window_idx:   int
    hour_start:   int
    hour_end:     int
    region:       str          # "coupled" for multi-region windows

    mean_cost:    float
    cvar_cost:    float
    obj_value:    float

    fuel_cost:    float
    startup_cost: float
    penalty_cost: float
    co2_t:        float   # tCO2 total for this window (scenario-mean)

    commitment:   dict[str, np.ndarray]   # {region: (N_free, T)}
    dispatch_exp: dict[str, np.ndarray]   # {region: (N_all, T)}
    costs_all:    np.ndarray              # (S,) total cost distribution

    solve_time_s: float
    convergence:  list[float]

    # Per-type penalty breakdown (summed over regions and T, mean over S)
    # These have defaults so they must come after all non-default fields.
    pen_scarcity:    float = 0.0
    pen_curtailment: float = 0.0
    pen_must_run:    float = 0.0
    pen_inertia:     float = 0.0
    pen_ramp:        float = 0.0

    # cross-border flows: (S, T, n_links) — only set by coupled solver
    flows:          Optional[np.ndarray] = None
    # per-region real MCP prices (mean over scenarios): {region: (T,)}
    mean_prices:    Optional[dict] = None
    # per-region storage net dispatch log: {region: (Ns, T)}
    storage_log:    Optional[dict] = None
    # per-region storage asset names: {region: [str]}
    storage_names:  Optional[dict] = None
    # per-region scenario-mean demand (T,): the actual load the units faced
    demand_mean:    Optional[dict] = None
    # per-region scenario-mean dispatch per unit: {region: (N_all, T)}
    dispatch_mean_map: Optional[dict] = None
    # per-region scenario-mean net position after ATC flows: {region: (T,)}
    net_pos_mean_map: Optional[dict] = None
    # per-region deterministic RES output pre-subtracted from demand: {region: (T,)}
    res_output_map: Optional[dict] = None

    @property
    def T(self) -> int:
        return self.hour_end - self.hour_start


# ── Results container ─────────────────────────────────────────────────────────

class Results:
    """
    Aggregates WindowResult objects. Provides summary reporting and
    export to dict / JSON.
    """

    def __init__(self, region: str, fleet: Fleet):
        self.region  = region
        self.fleet   = fleet
        self._windows: list[WindowResult] = []

    def add(self, wr: WindowResult) -> None:
        self._windows.append(wr)

    @property
    def windows(self) -> list[WindowResult]:
        return list(self._windows)

    @property
    def n_windows(self) -> int:
        return len(self._windows)

    @property
    def total_hours(self) -> int:
        return sum(w.T for w in self._windows)

    def total_cost(self) -> float:
        return sum(w.mean_cost for w in self._windows)

    def total_fuel_cost(self) -> float:
        return sum(w.fuel_cost for w in self._windows)

    def total_startup_cost(self) -> float:
        return sum(w.startup_cost for w in self._windows)

    def total_penalty_cost(self) -> float:
        return sum(w.penalty_cost for w in self._windows)

    def total_co2_t(self) -> float:
        """Total CO2 emissions in tonnes for the simulation horizon."""
        return sum(getattr(w, 'co2_t', 0.0) for w in self._windows)

    def mean_cvar(self) -> float:
        if not self._windows:
            return 0.0
        return float(np.mean([w.cvar_cost for w in self._windows]))

    def commitment_schedule(self, region: Optional[str] = None) -> np.ndarray:
        """Concatenated commitment (N_free, total_T) for one region."""
        r = region or self.region
        arrays = []
        for w in self._windows:
            if r in w.commitment:
                arrays.append(w.commitment[r])
        return np.hstack(arrays) if arrays else np.array([])

    def print_summary(self) -> None:
        sep = "=" * 76
        print(f"\n{sep}")
        print(f"  SUCCES — Results  |  {self.region}  |  "
              f"{self.n_windows} windows  ({self.total_hours}h)")
        print(sep)

        print(f"\n{'Win':>4} {'Hours':>10} {'E[cost]':>12} {'CVaR_5%':>12} "
              f"{'Fuel':>12} {'Startup':>9} {'Penalty':>10} {'Time':>6}")
        print("-" * 76)
        for w in self._windows:
            print(
                f"{w.window_idx:>4} "
                f"{w.hour_start:>4}-{w.hour_end:<5}"
                f"€{w.mean_cost:>10,.0f} "
                f"€{w.cvar_cost:>10,.0f} "
                f"€{w.fuel_cost:>10,.0f} "
                f"€{w.startup_cost:>7,.0f} "
                f"€{w.penalty_cost:>8,.0f} "
                f"{w.solve_time_s:>5.1f}s"
            )
        print("-" * 76)
        print(
            f"{'TOT':>4} {'':>10} "
            f"€{self.total_cost():>10,.0f} "
            f"€{self.mean_cvar():>10,.0f} "
            f"€{self.total_fuel_cost():>10,.0f} "
            f"€{self.total_startup_cost():>7,.0f} "
            f"€{self.total_penalty_cost():>8,.0f}"
        )
        co2 = self.total_co2_t()
        if co2 > 0:
            print(f"\n  CO2 emissions: {co2/1e3:,.1f} kt  "
                  f"({co2/max(self.total_hours,1):.0f} t/h avg)  "
                  f"intensity: {co2/max(self.total_fuel_cost()/50,1):.3f} tCO2/MWh_e approx")

        # Per-unit dispatch summary (thermal + hydro only — RES excluded)
        # Shows: avg MW dispatch, % of max_cap, min/max across the run horizon.
        # This is more informative than commitment fractions because it shows
        # whether the GA is actually using capacity efficiently:
        #   - baseload (coal, nuclear): should be 60-100% of max_cap
        #   - mid-merit (CCGT): 30-80% depending on season
        #   - peakers (OCGT): 5-30% (used only at peak demand hours)
        #   - hydro: variable, driven by inflow and water value
        for r in sorted(self.fleet.regions()):
            fixed_u  = self.fleet.fixed_units(r)
            free_u   = [u for u in self.fleet.free_units(r)
                        if not isinstance(u, (WindPlant, SolarPlant))]
            all_u    = fixed_u + free_u
            if not all_u:
                continue
            # Gather mean dispatch per unit across all windows from dispatch_mean_map
            unit_dispatch: dict[str, list[float]] = {u.name: [] for u in all_u}
            for w in self._windows:
                dm = (w.dispatch_mean_map or {}).get(r)
                if dm is None:
                    continue
                T_w = w.T
                n_th = len(all_u)
                for i, u in enumerate(all_u):
                    if i < dm.shape[0]:
                        unit_dispatch[u.name].extend(dm[i, :T_w].tolist())

            if not any(unit_dispatch.values()):
                continue

            print(f"\n  Dispatch summary — {r}:")
            print(f"    {'Unit':<28} {'AvgMW':>7}  {'%Cap':>5}  {'MinMW':>7}  {'MaxMW':>7}")
            for u in all_u:
                vals = unit_dispatch.get(u.name, [])
                if not vals:
                    continue
                avg_mw = sum(vals) / len(vals)
                min_mw = min(vals)
                max_mw = max(vals)
                pct    = avg_mw / u.max_cap * 100 if u.max_cap > 0 else 0
                print(f"    {u.name:<28} {avg_mw:7.1f}  {pct:5.1f}%  {min_mw:7.1f}  {max_mw:7.1f}")
        print(sep)

    def print_window_detail(self, window_idx: int) -> None:
        w    = self._windows[window_idx]
        T    = w.T
        sep  = "-" * 64

        print(f"\n{sep}")
        print(f"Window {w.window_idx}  h{w.hour_start}–h{w.hour_end}  {w.region}")
        print(sep)
        print(f"  E[cost]    = €{w.mean_cost:,.0f}")
        print(f"  CVaR_5%    = €{w.cvar_cost:,.0f}")
        print(f"  Fuel       = €{w.fuel_cost:,.0f}")
        print(f"  Startup    = €{w.startup_cost:,.0f}")
        print(f"  Penalty    = €{w.penalty_cost:,.0f}")
        print(f"  Solve time = {w.solve_time_s:.1f}s")

        for r, com in w.commitment.items():
            free_names  = [u.name for u in self.fleet.free_units(r)]
            fixed_names = [u.name for u in self.fleet.fixed_units(r)]
            all_names   = fixed_names + free_names

            print(f"\n  [{r}] Commitment:")
            for i, name in enumerate(free_names):
                row = "".join("1" if com[i, t] else "." for t in range(T))
                print(f"    {name:<20} {row}  [{int(com[i].sum())}h on]")
            for name in fixed_names:
                print(f"    {name:<20} {'1'*T}  [fixed]")

            if r in w.dispatch_exp:
                d = w.dispatch_exp[r]
                print(f"\n  [{r}] Dispatch — expected scenario (first 12h):")
                nh = min(T, 12)
                hdr = f"    {'Unit':<20} " + "".join(f"{t:>6}" for t in range(nh))
                print(hdr)
                for i, name in enumerate(all_names):
                    row = "".join(f"{d[i, t]:>6.0f}" for t in range(nh))
                    print(f"    {name:<20} {row}")

        if w.flows is not None and w.flows.size > 0:
            links = self.fleet  # can't access network here, just show totals
            mean_flow = w.flows.mean(axis=0)   # (T, n_links)
            print(f"\n  Cross-border flows (mean over scenarios, MW):")
            print(f"    {'Hour':<6} " +
                  "  ".join(f"Link{l}" for l in range(mean_flow.shape[1])))
            for t in range(min(T, 12)):
                row = "  ".join(f"{mean_flow[t, l]:>6.0f}"
                                for l in range(mean_flow.shape[1]))
                print(f"    {t:<6} {row}")

        print(sep)

    def _auto_report(self, report_path=None) -> None:
        """
        Build rich result dict from window data and generate HTML report.
        Writes a companion JSON with full per-hour detail alongside the HTML.
        """
        import traceback, os, json
        from pathlib import Path
        from datetime import datetime
        try:
            from .reporter import generate_report

            regions = sorted(self.fleet.regions())
            T_total = self.total_hours
            hours   = list(range(T_total))

            # ── Per-region per-hour data ───────────────────────────────────────
            region_data: dict = {}
            for r in regions:
                # Unit list order must match dispatch_mean row order from _parse_result.
                # julia_bridge._parse_result appends RES rows AFTER thermal+hydro rows:
                #   rows 0..N_thermal+hydro-1 = fixed_u + free_u (no RES)
                #   rows N_thermal+hydro..    = res_units (wind/solar)
                fixed_u  = self.fleet.fixed_units(r)
                th_free_u = [u for u in self.fleet.free_units(r)
                             if not isinstance(u, (WindPlant, SolarPlant))]
                res_u    = self.fleet.res_units(r)
                all_u    = fixed_u + th_free_u + res_u   # matches dispatch_mean row order
                storages = self.fleet.storages(r)

                gen: dict = {u.name: [] for u in all_u}
                storage_net: dict = {st.name: [] for st in storages}
                prices: list = []
                demand: list = []
                imports_h: list = []
                exports_h: list = []

                for w in self._windows:
                    T = w.T
                    # Use scenario-mean dispatch for chart (not scenario-1)
                    # dispatch_mean is the average MW across all S scenarios,
                    # which matches the mean_prices and mean_demand we also show.
                    disp_m = getattr(w, "dispatch_mean_map", {}).get(r)
                    disp_1 = w.dispatch_exp.get(r)  # fallback
                    disp   = disp_m if disp_m is not None else disp_1
                    for i, u in enumerate(all_u):
                        if disp is not None and i < disp.shape[0]:
                            gen[u.name].extend(disp[i, :].tolist())
                        else:
                            gen[u.name].extend([0.0] * T)

                    # Storage net dispatch (+ = discharge, - = charge)
                    st_log  = getattr(w, "storage_log",  {}).get(r)
                    st_names_w = getattr(w, "storage_names", {}).get(r, [st.name for st in storages])
                    for j, st in enumerate(storages):
                        if st_log is not None and j < st_log.shape[0]:
                            storage_net[st.name].extend(st_log[j, :].tolist())
                        else:
                            storage_net[st.name].extend([0.0] * T)

                    # Real MCP prices from Julia (mean over scenarios)
                    mp = getattr(w, "mean_prices", {}).get(r)
                    if mp is not None and len(mp) == T:
                        prices.extend(mp.tolist())
                    else:
                        prices.extend([0.0] * T)

                    # Demand for the report = GROSS demand (what consumers actually needed).
                    # dm is NET demand (what Julia saw = gross - RES output).
                    # Adding RES output back gives gross demand so that the stacked-area
                    # chart (thermal + RES vs gross demand) balances correctly.
                    dm = (w.demand_mean or {}).get(r)
                    res_out_w = (w.res_output_map or {}).get(r)
                    if dm is not None and len(dm) == T:
                        if res_out_w is not None and len(res_out_w) == T:
                            demand.extend([float(dm[t_]) + float(res_out_w[t_]) for t_ in range(T)])
                        else:
                            demand.extend(dm.tolist())
                    else:
                        # Fallback: sum of dispatch
                        demand.extend([
                            sum(gen[u.name][len(gen[u.name])-T+t_] for u in all_u)
                            for t_ in range(T)
                        ])

                    # Imports/exports from energy balance: accumulated each window.
                    # IMPORTANT: dm_w is NET demand (gross - RES output).
                    # local_gen must therefore also exclude RES, because RES was
                    # already subtracted from demand before the GA.
                    # Using (thermal+hydro gen only) against net demand gives the
                    # correct trade signal: positive = import needed, negative = export.
                    dm_w = (w.demand_mean or {}).get(r)
                    if dm_w is not None and len(dm_w) == T:
                        # N_thermal_hydro = number of non-RES units in all_u
                        n_thermal = len(fixed_u) + len(th_free_u)
                        offset_base = len(demand) - T
                        for t_ in range(T):
                            offset_t = offset_base + t_
                            # Only thermal+hydro dispatch (rows 0..n_thermal-1)
                            local_gen_t = sum(
                                gen[u.name][offset_t]
                                for u in (fixed_u + th_free_u)
                                if gen[u.name] and offset_t < len(gen[u.name])
                            )
                            st_net_t = sum(
                                (storage_net[st.name][offset_t]
                                 if storage_net[st.name] and offset_t < len(storage_net[st.name])
                                 else 0.0)
                                for st in storages
                            )
                            net_trade = float(dm_w[t_]) - local_gen_t - st_net_t
                            if net_trade > 1.0:
                                imports_h.append(net_trade)
                                exports_h.append(0.0)
                            elif net_trade < -1.0:
                                imports_h.append(0.0)
                                exports_h.append(-net_trade)
                            else:
                                imports_h.append(0.0)
                                exports_h.append(0.0)
                    else:
                        imports_h.extend([0.0] * T)
                        exports_h.extend([0.0] * T)

                region_data[r] = {
                    "generation":  gen,
                    "storage_net": storage_net,
                    "prices":      prices,
                    "demand":      demand,
                    "imports":     imports_h,
                    "exports":     exports_h,
                }

            # ── Full result dict ───────────────────────────────────────────────
            data = self.to_dict()
            data["regions"]     = regions
            data["hours"]       = hours
            data["region_data"] = region_data

            # ── Write companion JSON with full detail ──────────────────────────
            if report_path is not None:
                json_path = Path(report_path).with_suffix(".json")
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                json_path = Path(f"succes_results_{ts}.json")

            with open(json_path, "w") as f:
                json.dump(data, f, indent=2, default=lambda x:
                    x.tolist() if hasattr(x, "tolist") else float(x)
                    if hasattr(x, "__float__") else str(x))
            print(f"  ✓ Results JSON → {json_path.absolute()}")

            generate_report(data, out_path=report_path, verbose=True)

        except Exception as exc:
            print(f"  [reporter] Could not generate report: {exc}")
            if os.environ.get("SUCCES_DEBUG"):
                traceback.print_exc()

    def to_dict(self) -> dict:
        return {
            "region":            self.region,
            "n_windows":         self.n_windows,
            "total_hours":       self.total_hours,
            # Added for PTDF flow visualisation in reporter
            "network_links":     getattr(self, "_link_names", []),
            "total_cost":        self.total_cost(),
            "total_fuel_cost":   self.total_fuel_cost(),
            "total_startup_cost":self.total_startup_cost(),
            "total_penalty_cost":self.total_penalty_cost(),
            "total_co2_t":       self.total_co2_t(),
            "mean_cvar":         self.mean_cvar(),
            # Total penalty breakdown across all windows
            "total_pen_scarcity":    sum(getattr(w, 'pen_scarcity',    0.0) for w in self._windows),
            "total_pen_curtailment": sum(getattr(w, 'pen_curtailment', 0.0) for w in self._windows),
            "total_pen_must_run":    sum(getattr(w, 'pen_must_run',    0.0) for w in self._windows),
            "total_pen_inertia":     sum(getattr(w, 'pen_inertia',     0.0) for w in self._windows),
            "total_pen_ramp":        sum(getattr(w, 'pen_ramp',        0.0) for w in self._windows),
            "windows": [
                {
                    "window_idx":   w.window_idx,
                    "hour_start":   w.hour_start,
                    "hour_end":     w.hour_end,
                    "mean_cost":    w.mean_cost,
                    "cvar_cost":    w.cvar_cost,
                    "fuel_cost":    w.fuel_cost,
                    "startup_cost": w.startup_cost,
                    "penalty_cost": w.penalty_cost,
                    # Per-type penalty breakdown for this window
                    "penalty_breakdown": {
                        "scarcity":    getattr(w, 'pen_scarcity',    0.0),
                        "curtailment": getattr(w, 'pen_curtailment', 0.0),
                        "must_run":    getattr(w, 'pen_must_run',    0.0),
                        "inertia":     getattr(w, 'pen_inertia',     0.0),
                        "ramp":        getattr(w, 'pen_ramp',        0.0),
                    },
                    "co2_t":        getattr(w, 'co2_t', 0.0),
                    "solve_time_s": w.solve_time_s,
                    "convergence":  w.convergence,
                    # Commitment fractions {region: [[f per unit per hour]]}
                    # Maps directly to free_unit_names for diagnosis.
                    # Use this to find which units had low fractions at scarcity hours.
                    "commitment": {
                        r: com.tolist()
                        for r, com in w.commitment.items()
                        if com.size > 0
                    },
                    "free_unit_names": {
                        r: [u.name for u in self.fleet.free_units(r)
                            if not isinstance(u, (WindPlant, SolarPlant))]
                        for r in w.commitment
                        if r in [reg for reg in w.commitment]
                    },
                    # Mean flows over scenarios: list of (T,) per link, or []
                    "mean_flows":   (
                        w.flows.mean(axis=0).T.tolist()   # (n_links, T) as list
                        if w.flows is not None and w.flows.size > 0
                        else []
                    ),
                }
                for w in self._windows
            ],
        }


# ── Carryover builder (shared between both solvers) ───────────────────────────

def _build_carryover(
    fleet:    Fleet,
    regions:  list[str],
    window_result: dict,       # output of simulate_window or simulate_coupled
    commitments: dict[str, np.ndarray],
    T:        int,
) -> CarryoverState:
    """
    Extract end-of-window state into a new CarryoverState.

    Fix B — warm carryover for cycling units:
    When a unit is OFF at the last hour but was ON within the last ramp_time
    hours, it is mid-restart at window end. We carry it over as plant_on=True
    with last_dispatch = hours_since_restart × ramp_rate (ramp-consistent).
    This prevents the next window from treating it as cold-start and generating
    an h0 scarcity spike due to ramp constraints.
    """
    new_co = CarryoverState()

    for r in regions:
        # Exclude RES: wind/solar have no UC state to carry over.
        # dispatch_exp rows are ordered: fixed_units + th_free_units + res_units.
        # last_col covers only fixed + th_free (same as chromosome), so we must
        # exclude RES here to keep indices aligned.
        free_units  = [u for u in fleet.free_units(r)
                       if not isinstance(u, (WindPlant, SolarPlant))]
        fixed_units = fleet.fixed_units(r)
        all_disp    = fixed_units + free_units
        N_fixed     = len(fixed_units)
        com         = commitments.get(r, np.zeros((0, T)))

        # Last-hour commitment vector
        last_fixed = np.ones(N_fixed)
        last_free  = com[:, -1] if com.shape[1] > 0 else np.zeros(0)
        last_col   = np.concatenate([last_fixed, last_free])

        # Last dispatch from regional dispatch_exp
        if r in window_result:
            disp = window_result[r]["dispatch_exp"]
        elif "dispatch_exp" in window_result:
            disp = window_result["dispatch_exp"]
        else:
            disp = np.zeros((len(all_disp), T))

        for i, u in enumerate(all_disp):
            ramp = float(getattr(u, "ramp_rate", 0.0))
            min_cap = float(getattr(u, "min_cap", 0.0))
            mc = u.max_cap

            on_last = last_col[i] > 0.5
            d = float(disp[i, -1])

            # Fix B (extended): detect mid-restart — unit OFF at last hour but was ON
            # within the last max(9, ceil(min_cap/ramp)) hours.
            # The max(9, ...) catches fast CCGTs that were evening-cycled (off h16-h23):
            # those have tiny ceil(min_cap/ramp) (e.g. 1h for CCGT) but were clearly
            # cycling — the 9h lookback captures the restart regardless of speed.
            is_free = i >= N_fixed
            if is_free and not on_last and ramp > 0 and min_cap > 0:
                restart_hours = int(np.ceil(min_cap / ramp))
                lookback = min(max(9, restart_hours), T)
                unit_idx = i - N_fixed
                if com.shape[1] > 0:
                    recent_slice = com[unit_idx, max(0, T - lookback):]
                    hours_restarted = int(np.sum(recent_slice))  # hours ON in lookback
                    if hours_restarted > 0:
                        # Unit is mid-restart: treat as ON with ramp-consistent dispatch
                        on_last = True
                        d = min(hours_restarted * ramp, mc)

            # Slow-ramp fixed units (nuclear): cap carryover at 70% max
            if getattr(u, "fixed_on", False) and ramp > 0 and ramp / mc < 0.05:
                d = min(d, mc * 0.70)

            new_co.plant_on[u.name]     = on_last
            new_co.last_dispatch[u.name] = d

        # Hours on: count backwards from window end
        for i, u in enumerate(free_units):
            h = 0
            for t in range(T - 1, -1, -1):
                if com.shape[1] > 0 and com[i, t] == 1:
                    h += 1
                else:
                    break
            new_co.hours_on[u.name] = h

        # Storage and hydro state
        if r in window_result:
            new_co.storage_soc.update(window_result[r].get("storage_soc", {}))
            new_co.hydro_reservoir.update(window_result[r].get("hydro_res", {}))

    return new_co


# ── Single-region rolling horizon solver ─────────────────────────────────────

class RollingHorizonSolver:
    """
    Solves one region per window. Regions are independent.
    Use for single-region problems or per-region baselines.
    """

    def __init__(
        self,
        fleet:         Fleet,
        region:        str,
        scenario_fn:   Callable[[int, int, int], ScenarioBank],
        network:       Optional[Network] = None,
        n_scenarios:   int   = config.N_SCENARIOS,
        ga_epochs:     int   = config.GA_EPOCHS,
        ga_pop_size:   int   = config.GA_POP_SIZE,
        ga_pc:         float = config.GA_PC,
        ga_pm:         float = config.GA_PM,
        lambda_risk:   float = config.LAMBDA_RISK,
        seed:          int   = config.RANDOM_SEED,
        verbose:       bool  = True,
    ):
        self.fleet       = fleet
        self.region      = region
        self.scenario_fn = scenario_fn
        self.network     = network
        self.ga_epochs   = ga_epochs
        self.ga_pop_size = ga_pop_size
        self.ga_pc       = ga_pc
        self.ga_pm       = ga_pm
        self.lambda_risk = lambda_risk
        self.seed        = seed
        self.verbose     = verbose

    def solve_window(
        self, window_idx: int, hour_start: int, T: int, carryover: CarryoverState
    ) -> tuple[WindowResult, CarryoverState]:

        free_units = self.fleet.free_units(self.region)
        N_free     = len(free_units)
        scenarios  = self.scenario_fn(window_idx, hour_start, T)

        if self.verbose:
            print(f"\n  Window {window_idx:>3}  h{hour_start}–h{hour_start+T}  "
                  f"{N_free} free × {T}h = {N_free*T} vars", flush=True)

        t0 = time.time()
        if N_free == 0:
            best_solution = np.ones(0)
            convergence   = []
        else:
            obj_fn  = build_objective(
                self.fleet, scenarios, self.region, carryover, self.lambda_risk
            )
            problem = {
                "obj_func": obj_fn,
                "bounds":   BinaryVar(n_vars=N_free * T),
                "minmax":   "min",
                "log_to":   None,
            }
            opt     = GA.BaseGA(epoch=self.ga_epochs, pop_size=self.ga_pop_size,
                                pc=self.ga_pc, pm=self.ga_pm)
            g_best  = opt.solve(problem, seed=self.seed + window_idx)
            best_solution = np.asarray(g_best.solution, dtype=float)
            convergence   = list(opt.history.list_global_best_fit)

        solve_time = time.time() - t0

        # No post-repair: soft penalty in the objective guides the GA
        commitment = best_solution.reshape(N_free, T) if N_free > 0 else np.zeros((0, T))
        result   = simulate_window(
            commitment, self.fleet, scenarios, self.region, carryover=carryover
        )
        costs    = result["costs"]
        mean_c   = float(costs.mean())
        cvar_c   = compute_cvar(costs, config.CVAR_ALPHA)

        if self.verbose:
            print(f"         E=€{mean_c:,.0f}  CVaR=€{cvar_c:,.0f}  "
                  f"({solve_time:.1f}s)", flush=True)

        commitments = {self.region: commitment}
        new_co      = _build_carryover(
            self.fleet, [self.region],
            {self.region: result}, commitments, T
        )

        wr = WindowResult(
            window_idx   = window_idx,
            hour_start   = hour_start,
            hour_end     = hour_start + T,
            region       = self.region,
            mean_cost    = mean_c,
            cvar_cost    = cvar_c,
            obj_value    = mean_c + self.lambda_risk * cvar_c,
            fuel_cost    = float(result["fuel_costs"][0]),
            startup_cost = float(result["startup_costs"][0]),
            penalty_cost = float(result["penalty_costs"][0]),
            commitment   = commitments,
            dispatch_exp = {self.region: result["dispatch_exp"]},
            costs_all    = costs,
            solve_time_s = solve_time,
            convergence  = convergence,
        )
        return wr, new_co

    def run(self, total_hours: int, window_hours: int = config.WINDOW_HOURS, report_path=None) -> Results:
        results   = Results(self.region, self.fleet)
        carryover = CarryoverState()
        n_windows = int(np.ceil(total_hours / window_hours))

        _print_header(
            label       = "Single-region",
            regions     = [self.region],
            fleet       = self.fleet,
            n_windows   = n_windows,
            window_hours= window_hours,
            total_hours = total_hours,
            ga_epochs   = self.ga_epochs,
            ga_pop_size = self.ga_pop_size,
            n_scenarios = "from scenario_fn",
        )
        t0 = time.time()
        for w in range(n_windows):
            h_start   = w * window_hours
            T         = min(window_hours, total_hours - h_start)
            wr, carryover = self.solve_window(w, h_start, T, carryover, total_hours=total_hours)
            results.add(wr)

        print(f"\n  Total: {time.time()-t0:.1f}s")
        # Store link names for PTDF flow visualisation
        if hasattr(self, "network") and self.network is not None:
            results._link_names = [
                f"{l.region_a}-{l.region_b}"
                for l in self.network.links()
            ]
        results._auto_report(report_path)
        return results


# ── Coupled multi-region rolling horizon solver ───────────────────────────────

class CoupledRollingHorizonSolver:
    """
    Solves all regions jointly per window using one GA with a combined
    binary decision vector: [region_A_bits | region_B_bits | ...].

    The objective calls simulate_coupled(), so cross-border flows are
    part of the evaluation. The GA learns commitment schedules that
    exploit cheap inter-regional surplus rather than spinning up
    expensive local units.

    Parallelism
    -----------
    Windows are solved sequentially (carryover is required).
    Population evaluations within each window can be parallelised
    by setting config.PARALLEL = True. Each evaluation is a
    stateless call to simulate_coupled() and is safe to parallelise.

    Note: MEALPY's built-in parallelism (n_workers) handles this
    automatically — set it in config or pass n_workers here.
    """

    def __init__(
        self,
        fleet:          Fleet,
        scenario_fn_map: dict[str, Callable[[int, int, int], ScenarioBank]],
        network:        Network,
        ga_epochs:      int   = config.GA_EPOCHS,
        ga_pop_size:    int   = config.GA_POP_SIZE,
        ga_pc:          float = config.GA_PC,
        ga_pm:          float = config.GA_PM,
        lambda_risk:    float = config.LAMBDA_RISK,
        seed:           int   = config.RANDOM_SEED,
        verbose:        bool  = True,
    ):
        self.fleet           = fleet
        self.scenario_fn_map = scenario_fn_map
        self.network         = network
        self.ga_epochs       = ga_epochs
        self.ga_pop_size     = ga_pop_size
        self.ga_pc           = ga_pc
        self.ga_pm           = ga_pm
        self.lambda_risk     = lambda_risk
        self.seed            = seed
        self.verbose         = verbose
        # Bridge for final evaluation — Julia owns all dispatch computation
        self._bridge         = get_bridge(verbose=verbose)

    def solve_window(
        self, window_idx: int, hour_start: int, T: int, carryover: CarryoverState,
        total_hours: int = 0,
        prev_best: "np.ndarray | None" = None,
    ) -> tuple[WindowResult, CarryoverState]:
        """
        Solve one window with MPC look-ahead overlap (Fix C).

        The GA optimises T_opt = T + OVERLAP_HOURS hours so it can see past the
        window boundary and avoid cycling units at window-end that would cause
        cold-start scarcity in the next window. Only the first T hours of the
        resulting commitment schedule are committed and passed to carryover.
        """
        overlap   = getattr(config, 'OVERLAP_HOURS', 0)
        # Full overlap applies to all windows including W0.
        # W0 with 12h overlap sees h24-h36 in the next window's demand realisation,
        # correctly pricing the cold-start restart cost at h24 so the GA avoids
        # cycling too many units simultaneously in the last hours of W0.
        # Don't extend past total simulation horizon
        T_opt     = T + min(overlap, max(0, total_hours - hour_start - T))

        regions      = sorted(self.fleet.regions())
        region_sizes = [(r, len([u for u in self.fleet.free_units(r)
                                 if not isinstance(u, (WindPlant, SolarPlant))])) for r in regions]
        total_vars   = sum(n * T_opt for _, n in region_sizes)

        # ── Scenario construction with correct seed split ─────────────────────
        # Bug-fix: previously the entire T_opt scenario (binding + overlap) used
        # window_idx+1 as the seed, so the GA optimised the binding zone (hours 0..T)
        # against the wrong demand realisation. The final dispatch corrected it, but
        # the chromosome was fitted to mismatched demand.
        #
        # Fix: build two scenario banks and stitch them together:
        #   • hours 0..T-1   (binding)  → seed = window_idx    (matches final dispatch)
        #   • hours T..T_opt-1 (overlap) → seed = window_idx+1 (next window's realisation)
        #
        # We achieve this by building T_opt-length scenarios using the binding seed,
        # then overwriting the overlap portion with next-window scenarios.
        # Because build_scenario_bank draws a single shared weather factor for all S
        # scenarios, we stitch at the ScenarioBank level rather than per-region.
        #
        # Simple approximation (sufficient for the GA's heuristic purpose):
        #   If T_opt == T (last window, no overlap), use window_idx seed throughout.
        #   Otherwise build T_opt with window_idx seed for binding zone coherence,
        #   and accept that the overlap zone uses the same seed — the key benefit
        #   (seeing future min-run/startup cost structure) is scenario-seed-independent.
        #   Full stitching is a future improvement (see SUCCES_DOCS.md).
        scenario_map = {
            r: self.scenario_fn_map[r](window_idx, hour_start, T_opt)
            for r in regions
        }

        if self.verbose:
            parts = " + ".join(f"{n}×{r}" for r, n in region_sizes)
            overlap_str = f" (+{T_opt-T}h overlap)" if T_opt > T else ""
            print(f"\n  Window {window_idx:>3}  h{hour_start}–h{hour_start+T}{overlap_str}  "
                  f"[{parts}] = {total_vars} vars  coupled", flush=True)

        t0 = time.time()
        if total_vars == 0:
            best_solution = np.ones(0)
            convergence   = []
        elif _get_batch_engine() is not None:
            # ── Julia batch mode: entire population evaluated in Julia per epoch ──
            best_solution, convergence = run_julia_batch_ga(
                fleet        = self.fleet,
                scenario_map = scenario_map,
                network      = self.network,
                carryover    = carryover,
                region_sizes = region_sizes,
                T            = T_opt,       # GA sees T_opt hours
                epochs       = self.ga_epochs,
                pop_size     = self.ga_pop_size,
                pc           = self.ga_pc,
                pm           = self.ga_pm,
                lambda_risk  = self.lambda_risk,
                cvar_alpha   = config.CVAR_ALPHA,
                seed         = self.seed + window_idx,
                verbose      = self.verbose,
                prev_best    = prev_best,
            )
        else:
            # ── Python fallback: mealpy GA ────────────────────────────────────
            obj_fn = build_coupled_objective(
                self.fleet, scenario_map, self.network, carryover,
                region_sizes, T_opt, self.lambda_risk
            )
            problem = {
                "obj_func": obj_fn,
                "bounds":   BinaryVar(n_vars=total_vars),
                "minmax":   "min",
                "log_to":   None,
            }
            opt    = GA.BaseGA(epoch=self.ga_epochs, pop_size=self.ga_pop_size,
                               pc=self.ga_pc, pm=self.ga_pm)
            g_best = opt.solve(problem, seed=self.seed + window_idx)
            best_solution = np.asarray(g_best.solution, dtype=float)
            convergence   = list(opt.history.list_global_best_fit)

        solve_time = time.time() - t0

        # Decode solution — slice to first T hours only (MPC: commit only binding horizon)
        commitments: dict[str, np.ndarray] = {}
        offset = 0
        for region, N_free in region_sizes:
            chunk = best_solution[offset: offset + N_free * T_opt] if total_vars > 0 \
                    else np.zeros(0)
            offset += N_free * T_opt
            com_full = chunk.astype(float).reshape(N_free, T_opt) if N_free > 0 else np.zeros((0, T_opt))
            commitments[region] = com_full[:, :T]   # commit only first T hours

        # Final evaluation uses T-hour commitments and T-hour scenario
        scenario_map_T = {
            r: self.scenario_fn_map[r](window_idx, hour_start, T)
            for r in regions
        }
        coupled_result = self._bridge.simulate_coupled(
            commitments, self.fleet, scenario_map_T, self.network,
            carryover=carryover,
            flow_passes=config.FLOW_PASSES if hasattr(config, 'FLOW_PASSES') else 3,
        )
        costs    = coupled_result["total_costs"]
        mean_c   = float(costs.mean())
        cvar_c   = compute_cvar(costs, config.CVAR_ALPHA)

        if self.verbose:
            print(f"         E=€{mean_c:,.0f}  CVaR=€{cvar_c:,.0f}  "
                  f"({solve_time:.1f}s)", flush=True)

        # Aggregate cost breakdown across regions
        fuel_c  = sum(float(coupled_result["regions"][r]["fuel_costs"][0])
                      for r in regions)
        start_c = sum(float(coupled_result["regions"][r]["startup_costs"][0])
                      for r in regions)
        pen_c   = float(costs[0]) - fuel_c - start_c
        co2_total = sum(float(coupled_result["regions"][r].get("co2_emissions",
                         np.zeros(1))[0])
                        for r in regions)

        def _pen_sum(key):
            return sum(float(coupled_result["regions"][r].get(key, np.zeros(1))[0])
                       for r in regions)

        pen_scarcity_c    = _pen_sum("pen_scarcity")
        pen_curtailment_c = _pen_sum("pen_curtailment")
        pen_must_run_c    = _pen_sum("pen_must_run")
        pen_inertia_c     = _pen_sum("pen_inertia")
        pen_ramp_c        = _pen_sum("pen_ramp")

        new_co = _build_carryover(
            self.fleet, regions,
            coupled_result["regions"], commitments, T
        )

        wr = WindowResult(
            window_idx   = window_idx,
            hour_start   = hour_start,
            hour_end     = hour_start + T,
            region       = "coupled:" + "+".join(regions),
            mean_cost    = mean_c,
            cvar_cost    = cvar_c,
            obj_value    = mean_c + self.lambda_risk * cvar_c,
            fuel_cost    = fuel_c,
            startup_cost = start_c,
            penalty_cost = pen_c,
            pen_scarcity    = pen_scarcity_c,
            pen_curtailment = pen_curtailment_c,
            pen_must_run    = pen_must_run_c,
            pen_inertia     = pen_inertia_c,
            pen_ramp        = pen_ramp_c,
            co2_t        = co2_total,
            commitment   = commitments,
            dispatch_exp = {r: coupled_result["regions"][r]["dispatch_exp"]
                            for r in regions},
            costs_all    = costs,
            solve_time_s = solve_time,
            convergence  = convergence,
            flows        = coupled_result["flows"],
            mean_prices  = {r: coupled_result["regions"][r].get("mean_prices",
                               np.zeros(T)) for r in regions},
            storage_log  = {r: coupled_result["regions"][r].get("storage_log",
                               np.zeros((0, T))) for r in regions},
            storage_names= {r: list(coupled_result["regions"][r].get("storage_names",
                               [st.name for st in self.fleet.storages(r)]))
                            for r in regions},
            demand_mean  = {r: scenario_map[r].demand_for_region(r).mean(axis=0)
                            for r in regions},
            res_output_map = {r: np.array(self._bridge._last_res_output.get(r, np.zeros(T)))
                              for r in regions},
            dispatch_mean_map = {r: coupled_result["regions"][r].get("dispatch_mean",
                                    coupled_result["regions"][r].get("dispatch_exp",
                                    np.zeros((0, T))))
                                 for r in regions},
            net_pos_mean_map  = {r: coupled_result["regions"][r].get("net_pos_mean",
                                    coupled_result["regions"][r]["net_position"].mean(axis=0))
                                 for r in regions},
        )
        # Attach best_solution for warm-start in next window
        wr._best_solution = best_solution
        return wr, new_co

    def run(self, total_hours: int, window_hours: int = config.WINDOW_HOURS, report_path=None) -> Results:
        regions   = sorted(self.fleet.regions())
        label     = "coupled:" + "+".join(regions)
        results   = Results(label, self.fleet)
        carryover = CarryoverState()
        n_windows = int(np.ceil(total_hours / window_hours))

        _print_header(
            label       = "Coupled multi-region",
            regions     = regions,
            fleet       = self.fleet,
            n_windows   = n_windows,
            window_hours= window_hours,
            total_hours = total_hours,
            ga_epochs   = self.ga_epochs,
            ga_pop_size = self.ga_pop_size,
            n_scenarios = "from scenario_fn_map",
        )
        t0 = time.time()
        _prev_best_solution = None   # warm-start: passes previous window's best to GA
        for w in range(n_windows):
            h_start       = w * window_hours
            T             = min(window_hours, total_hours - h_start)
            wr, carryover = self.solve_window(w, h_start, T, carryover,
                                              prev_best=_prev_best_solution)
            results.add(wr)
            # Store best solution for warm-start in next window
            _prev_best_solution = getattr(wr, "_best_solution", None)

        print(f"\n  Total: {time.time()-t0:.1f}s")
        # Store link names for PTDF flow visualisation
        if hasattr(self, "network") and self.network is not None:
            results._link_names = [
                f"{l.region_a}-{l.region_b}"
                for l in self.network.links()
            ]
        results._auto_report(report_path)
        return results


# ── Shared header printer ─────────────────────────────────────────────────────

def _print_header(**kw) -> None:
    sep = "=" * 76
    print(f"\n{sep}")
    print(f"  SUCCES — {kw['label']} Solver")
    print(f"  Regions: {kw['regions']}  |  Horizon: {kw['total_hours']}h  |  "
          f"{kw['n_windows']} × {kw['window_hours']}h windows")
    for r in kw['regions']:
        nf = len(kw['fleet'].free_units(r))
        nx = len(kw['fleet'].fixed_units(r))
        ns = len(kw['fleet'].storages(r))
        print(f"  [{r}]  free={nf}  fixed={nx}  storage={ns}")
    print(f"  GA: epochs={kw['ga_epochs']}  pop={kw['ga_pop_size']}  "
          f"scenarios={kw['n_scenarios']}")
    print(sep)
