"""
succes/simulator.py
-------------------
Stochastic dispatch simulator — fully vectorised, S-scenarios in parallel.

Key design decisions
--------------------
- The T-hour loop cannot be eliminated (causality: dispatch_t depends on
  prev_dispatch_t-1 via ramp constraints and SOC updates).
- Everything INSIDE each hour is fully vectorised over S via numpy broadcasting.
- No Python loops over S.  The inner loop is: for t in range(T): <numpy ops>
- MCP price: marginal unit found via numpy argmax, not a Python loop.
- Fuel cost matrix is pre-built once per call.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .assets import Fleet, ThermalPlant, HydroPlant, HeatPlant, StorageAsset
from .scenarios import ScenarioBank
from .network import Network, TransmissionLink
from . import config


# ── CarryoverState ────────────────────────────────────────────────────────────

@dataclass
class CarryoverState:
    """
    State carried between rolling-horizon windows.

    Attributes
    ----------
    last_dispatch  : {plant_name: float}  last MW output (ramping)
    plant_on       : {plant_name: bool}   was plant on at end of prev window
    hours_on       : {plant_name: int}    consecutive hours on at window end
    storage_soc    : {asset_name: ndarray(S,)} SOC per scenario
    hydro_reservoir: {asset_name: ndarray(S,)} reservoir per scenario
    """
    last_dispatch:    dict = field(default_factory=dict)
    plant_on:         dict = field(default_factory=dict)
    hours_on:         dict = field(default_factory=dict)
    storage_soc:      dict = field(default_factory=dict)
    hydro_reservoir:  dict = field(default_factory=dict)

    def get_last_dispatch(self, name: str, default: float = 0.0) -> float:
        return self.last_dispatch.get(name, default)

    def get_plant_on(self, name: str, default: bool = True) -> bool:
        return self.plant_on.get(name, default)

    def get_hours_on(self, name: str, default: int = 0) -> int:
        return self.hours_on.get(name, default)

    def get_soc(self, name: str, S: int, initial: float) -> np.ndarray:
        v = self.storage_soc.get(name)
        if v is None or len(v) != S:
            return np.full(S, float(initial))
        return v.copy()

    def get_reservoir(self, name: str, S: int, initial: float) -> np.ndarray:
        v = self.hydro_reservoir.get(name)
        if v is None or len(v) != S:
            return np.full(S, float(initial))
        return v.copy()


# ── Fuel cost matrix ──────────────────────────────────────────────────────────

def _build_fuel_cost_matrix(
    units: list, scenarios: ScenarioBank, S: int,
    co2_price: float = 82.0,   # EUR/tCO2 — ETS baseline, ~2023 level
) -> np.ndarray:
    """
    Build (S, N) marginal cost matrix including EU ETS CO2 cost.

    Marginal cost = fuel_cost + co2_price × co2_intensity

    CO2 price (€82/tCO2) produces the correct 2023-era CWE merit order:
      nuclear(~9) < hydro(28-38) < biomass(~57) < CCGT(~86) < coal(~90)
                  < lignite(~98) < OCGT(~127) EUR/MWh

    This is the "merit-order flip": gas CCGT is cheaper than coal and lignite
    on a full marginal cost basis including carbon — matching real post-2022
    European dispatch where lignite runs less and CCGT more.

    Hydro: water_value already represents opportunity cost; no CO2 adder.
    Storage/batteries: no combustion, no CO2.
    CO2 is deterministic here; add stochastic ETS variation via FuelPriceGenerator
    if scenario-to-scenario carbon price variation is desired.
    """
    N = len(units)
    fcm = np.zeros((S, N), dtype=np.float64)
    for i, u in enumerate(units):
        base = getattr(u, "base_fuel_cost", 0.0)
        hr   = getattr(u, "heat_rate",      1.0)
        ft   = getattr(u, "fuel_type",      None)
        wv   = getattr(u, "water_value",    None)
        co2i = float(getattr(u, "co2_intensity", 0.0))
        if wv is not None:
            # Hydro: water value is already an opportunity cost — no CO2 adder.
            # The water saved now could be dispatched when CO2 cost is priced in,
            # so the water_value implicitly reflects carbon premium.
            fcm[:, i] = float(wv)
        elif ft is not None:
            fname = ft.name.lower()
            fp      = scenarios.fuel_price_for(fname, base)    # (S_bank, T)
            fuel_mc = fp[:S].mean(axis=1) * hr                   # (S,) fuel cost only
            # CO2 cost: use stochastic ETS price if available in scenario bank,
            # otherwise use the fixed co2_price parameter.
            # This allows scenario-to-scenario carbon price variation (±15-20%)
            # which affects the merit order in high/low ETS price scenarios.
            if co2i > 1e-6:
                co2_fp = scenarios.fuel_price_for("co2", co2_price)  # (S_bank, T)
                # Align to the requested S (may differ from bank S)
                co2_mc = co2_fp[:S].mean(axis=1) * co2i   # (S,) per-scenario CO2 cost
            else:
                co2_mc = 0.0
            # Align fuel_mc length too (bank S may differ from requested S)
            fcm[:, i] = fuel_mc[:S] + co2_mc
        else:
            fcm[:, i] = base + co2_price * co2i
    return fcm


# ── Vectorised simulate_window ────────────────────────────────────────────────

def simulate_window(
    commitment:   np.ndarray,
    fleet:        Fleet,
    scenarios:    ScenarioBank,
    region:       str,
    carryover:    Optional[CarryoverState] = None,
    extra_supply: Optional[np.ndarray]    = None,
) -> dict:
    """
    Simulate one 24h window for one region, all S scenarios simultaneously.
    No Python loops over S — fully vectorised via numpy.

    Parameters
    ----------
    commitment   : (N_free, T) binary
    fleet        : Fleet
    scenarios    : ScenarioBank for this window
    region       : region key
    carryover    : end-state of previous window
    extra_supply : (S, T) imports [optional]

    Returns
    -------
    dict with keys: costs, fuel_costs, startup_costs, penalty_costs (S,),
        storage_soc, hydro_res (dicts), dispatch_exp (N_all, T),
        dispatch_mean (N_all, T) — mean over scenarios,
        storage_log (Ns, T), net_position (S, T),
        prices (S, T), mean_prices (T,)
    """
    S  = scenarios.S
    T  = scenarios.T
    co = carryover or CarryoverState()

    free_units  = fleet.free_units(region)
    fixed_units = fleet.fixed_units(region)
    storages    = fleet.storages(region)
    N_free      = len(free_units)
    N_fixed     = len(fixed_units)
    all_disp    = fixed_units + free_units
    N_all       = len(all_disp)
    Ns          = len(storages)

    # ── Full commitment (N_all, T) ─────────────────────────────────────────────
    fixed_com   = np.ones((N_fixed, T), dtype=np.float64)
    full_commit = np.vstack([fixed_com, commitment.astype(np.float64)]) \
                  if N_free > 0 else fixed_com
    avail       = np.array([u.availability(T) for u in all_disp], dtype=np.float64)
    full_commit = full_commit * avail           # (N_all, T)

    # ── Startup costs (deterministic) ─────────────────────────────────────────
    prev_on_vec   = np.array([1.0 if co.get_plant_on(u.name, True) else 0.0
                               for u in all_disp], dtype=np.float64)
    startup_costs = np.array([getattr(u, "startup_cost", 0.0) for u in all_disp],
                              dtype=np.float64)
    startup_total = 0.0
    for t in range(T):
        on_t   = full_commit[:, t]
        starts = (on_t > 0.5) & (prev_on_vec < 0.5)
        startup_total += (startup_costs * starts).sum()
        prev_on_vec    = on_t

    # ── Unit parameters ────────────────────────────────────────────────────────
    min_caps_v  = np.array([getattr(u, "min_cap",   0.0) for u in all_disp], dtype=np.float64)
    max_caps_v  = np.array([u.max_cap                    for u in all_disp], dtype=np.float64)
    ramp_rates_v= np.array([getattr(u, "ramp_rate", 0.0) for u in all_disp], dtype=np.float64)
    has_ramp    = ramp_rates_v > 0
    fcm         = _build_fuel_cost_matrix(all_disp, scenarios, S)  # (S, N_all)

    # ── Initial state ──────────────────────────────────────────────────────────
    # prev_on: whether each unit was ON at end of previous window
    prev_on_arr = np.array([1.0 if co.get_plant_on(u.name, True) else 0.0
                            for u in all_disp], dtype=np.float64)

    # prev_dispatch default:
    # - Units that were OFF: 0 (they ramp up from zero when turned on)
    # - Fast-ramping units (ramp >= 10% of max/h): 70% of max_cap
    #   (assumed to be near economic dispatch overnight)
    # - Slow-ramping baseload (ramp < 10% of max/h, e.g. nuclear, lignite):
    #   min_cap (these units run at minimum overnight, ramp up for morning peak)
    #   Using 70% max for these causes ramp-down curtailment in morning valley.
    prev_disp_def = np.zeros(N_all, dtype=np.float64)
    for i, u in enumerate(all_disp):
        if prev_on_arr[i] < 0.5:
            prev_disp_def[i] = 0.0   # was off
        else:
            rr = ramp_rates_v[i]
            mc = max_caps_v[i]
            if rr > 0 and rr / mc < 0.10:
                # Slow baseload: starts at min_cap (overnight minimum)
                prev_disp_def[i] = min_caps_v[i]
            else:
                # Fast/flexible: starts at 70% max (near economic dispatch)
                prev_disp_def[i] = mc * 0.7

    prev_dispatch_s = np.tile(np.array([
        co.get_last_dispatch(u.name, prev_disp_def[i])
        for i, u in enumerate(all_disp)
    ], dtype=np.float64), (S, 1))                                   # (S, N_all)

    prev_on_s = np.tile(prev_on_arr, (S, 1))                       # (S, N_all)

    hydro_soc   = {u.name: co.get_reservoir(u.name, S, u.initial_reservoir)
                   for u in fleet.hydros(region)}
    storage_soc = {st.name: co.get_soc(st.name, S, st.initial_soc)
                   for st in storages}

    # Hydro inflow arrays — precomputed
    hydro_inflow = {u.name: u.inflow_array(T) for u in fleet.hydros(region)}
    hydro_cap    = {u.name: u.reservoir_capacity for u in fleet.hydros(region)}

    # Hydro unit indices in all_disp
    hydro_idx    = {u.name: i for i, u in enumerate(all_disp)
                   if isinstance(u, HydroPlant)}

    # ── Merit order: fixed for window (use scenario-mean fuel cost) ────────────
    mean_fc = fcm.mean(axis=0)                    # (N_all,)
    order   = np.argsort(mean_fc)                 # cheapest first (index array)
    order_list = order.tolist()

    # ── Pre-build (N_all,) broadcast arrays ───────────────────────────────────
    # Hydro unit indices as boolean mask for fast per-step checks
    hydro_mask = np.zeros(N_all, dtype=bool)
    hydro_list = []   # [(unit_idx, name)]
    for hname, hidx_i in hydro_idx.items():
        hydro_mask[hidx_i] = True
        hydro_list.append((hidx_i, hname))

    # ── Output arrays ──────────────────────────────────────────────────────────
    demand_arr    = scenarios.demand_for_region(region)   # (S, T)
    fuel_costs    = np.zeros(S,         dtype=np.float64)
    penalty_costs = np.zeros(S,         dtype=np.float64)
    dispatch_log  = np.zeros((N_all, T), dtype=np.float64)  # scenario-1
    dispatch_sum  = np.zeros((N_all, T), dtype=np.float64)  # sum over S (for mean)
    storage_log   = np.zeros((Ns, T),    dtype=np.float64)
    net_position  = np.zeros((S, T),     dtype=np.float64)
    prices        = np.zeros((S, T),     dtype=np.float64)

    # ── Pre-allocate per-hour work arrays ──────────────────────────────────────
    lower_t     = np.empty((S, N_all), dtype=np.float64)
    upper_t     = np.empty((S, N_all), dtype=np.float64)
    dispatch_t  = np.empty((S, N_all), dtype=np.float64)
    remaining_s = np.empty(S,          dtype=np.float64)

    for t in range(T):
        on_t    = full_commit[:, t]                 # (N_all,) float
        on_mask = on_t > 0.5                        # (N_all,) bool
        on_cols = np.where(on_mask)[0]

        # Demand this hour
        if extra_supply is not None:
            remaining_s[:] = demand_arr[:, t] - extra_supply[:, t]
        else:
            remaining_s[:] = demand_arr[:, t]

        # ── Hydro inflow (cheap: one op per hydro unit) ───────────────────────
        for hidx_i, hname in hydro_list:
            soc = hydro_soc[hname]
            soc += hydro_inflow[hname][t]
            np.minimum(soc, hydro_cap[hname], out=soc)

        # ── Bounds: vectorised over S, loop only over ON units ────────────────
        # Broadcast shape: (S, N_on) all at once
        if len(on_cols):
            pd_on = prev_dispatch_s[:, on_cols]     # (S, N_on)
            lo_on = min_caps_v[on_cols]             # (N_on,)
            hi_on = max_caps_v[on_cols]             # (N_on,)
            rr_on = ramp_rates_v[on_cols]           # (N_on,)
            hr_on = has_ramp[on_cols]               # (N_on,) bool

            # Ramp-limited units: broadcast over S all at once
            lo_mat = np.where(hr_on, np.maximum(pd_on - rr_on, lo_on), lo_on)  # (S, N_on)
            hi_mat = np.where(hr_on, np.minimum(pd_on + rr_on, hi_on), hi_on)  # (S, N_on)

            # Hydro cap: per-unit update (only a few units)
            for ci, gi in enumerate(on_cols):
                if hydro_mask[gi]:
                    hname = next(k for k, v in hydro_idx.items() if v == gi)
                    soc_h = hydro_soc[hname]
                    hi_mat[:, ci] = np.minimum(hi_mat[:, ci], np.maximum(soc_h, 0.0))
                    lo_mat[:, ci] = np.minimum(lo_mat[:, ci], hi_mat[:, ci])

            lower_t[:, on_cols] = lo_mat
            upper_t[:, on_cols] = hi_mat
            # Off units stay at 0
            off_cols = np.where(~on_mask)[0]
            if len(off_cols):
                lower_t[:, off_cols] = 0.0
                upper_t[:, off_cols] = 0.0

            lb_total = lo_mat.sum(axis=1)           # (S,)
            ub_total = hi_mat.sum(axis=1)           # (S,)
        else:
            lower_t[:] = 0.0; upper_t[:] = 0.0
            lb_total = np.zeros(S); ub_total = np.zeros(S)

        # ── Structural penalty ────────────────────────────────────────────────
        # Only penalise if lower bounds EXCEED demand (unavoidable over-generation).
        # Do NOT pre-penalise demand > ub_total here — storage may cover that gap.
        # The post-dispatch check (after storage) handles true scarcity correctly.
        penalty_costs += np.maximum(0.0, lb_total - remaining_s) * config.MUST_RUN_PENALTY

        # ── Baseline + merit-order top-up ─────────────────────────────────────
        dispatch_t[:] = lower_t
        remaining_s  -= lb_total

        for i in order_list:
            if not on_mask[i]:
                continue
            headroom = upper_t[:, i] - lower_t[:, i]
            added    = np.minimum(headroom, np.maximum(0.0, remaining_s))
            dispatch_t[:, i] += added
            remaining_s      -= added

        # ── Storage dispatch: vectorised over S ───────────────────────────────
        for si, st in enumerate(storages):
            soc = storage_soc[st.name]             # (S,) mutable
            de  = st.discharge_efficiency
            ce  = st.charge_efficiency
            dr  = st.discharge_rate
            cr  = st.charge_rate
            ec  = st.energy_capacity

            # Discharge into deficit
            discharge = np.minimum(np.minimum(dr, soc * de),
                                   np.maximum(0.0, remaining_s))
            # Charge from surplus
            charge    = np.minimum(np.minimum(cr, (ec - soc) / max(ce, 1e-9)),
                                   np.maximum(0.0, -remaining_s))

            storage_soc[st.name] = np.clip(
                soc - discharge / max(de, 1e-9) + charge * ce, 0.0, ec
            )
            remaining_s          -= discharge - charge
            fuel_costs           += (discharge + charge) * st.marginal_cost
            storage_log[si, t]    = float(discharge[0] - charge[0])

        # ── Final balance ──────────────────────────────────────────────────────
        penalty_costs     += np.maximum(0.0,  remaining_s) * config.SCARCITY_PENALTY
        penalty_costs     += np.maximum(0.0, -remaining_s) * config.CURTAILMENT_PENALTY
        net_position[:, t] = -remaining_s

        # ── Fuel cost: variable dispatch cost ─────────────────────────────────
        # Variable cost = (dispatch - min_cap) * marginal_cost for ON units.
        # No-load cost  = min_cap * marginal_cost for every committed unit-hour.
        # Together: dispatch * marginal_cost (since dispatch >= min_cap for ON units).
        # The no-load cost is thus already implicit: when a unit is ON even at
        # min_cap it pays min_cap * fuel_cost per hour. This creates the incentive
        # to turn it OFF overnight when neighbours can supply cheaply via ATC.
        fuel_costs += (dispatch_t * fcm).sum(axis=1)

        # ── MCP (vectorised over S) ────────────────────────────────────────────
        # Approach: for each scenario, find the highest-merit-order unit with
        # dispatch > lower_bound (i.e. it was marginal).
        # Vectorised: build (S, N_on) matrix, find last non-zero headroom-used column.
        if len(on_cols):
            used     = dispatch_t[:, on_cols] - lower_t[:, on_cols]   # (S, N_on)
            # Reorder on_cols by merit order
            on_ordered = [i for i in order if on_mask[i]]
            used_ord   = dispatch_t[:, on_ordered] - lower_t[:, on_ordered]  # (S, N_on)
            # For each scenario: last column with used > 1e-3 -> its fuel cost
            fc_on = fcm[:, on_ordered]                                 # (S, N_on)
            marginal_mask = used_ord > 1e-3                            # (S, N_on)
            # Use argmax on reversed axis to find last marginal unit per scenario
            rev = marginal_mask[:, ::-1]
            idx = rev.argmax(axis=1)                                   # (S,)
            has_marginal = rev.any(axis=1)
            N_on = len(on_ordered)
            mcp_idx  = (N_on - 1) - idx                               # (S,)
            mcp_vals = fc_on[np.arange(S), mcp_idx]                   # (S,)
            mcp_vals = np.where(has_marginal, mcp_vals, 0.0)
        else:
            mcp_vals = np.zeros(S)

        prices[:, t] = np.where(net_position[:, t] < -1.0, config.SCARCITY_PENALTY, mcp_vals)
        prices[net_position[:, t] > 1.0, t] = 0.0

        # ── Hydro water consumption ────────────────────────────────────────────
        for hname, hidx_i in hydro_idx.items():
            if on_mask[hidx_i]:
                np.maximum(0.0, hydro_soc[hname] - dispatch_t[:, hidx_i],
                           out=hydro_soc[hname])

        # ── Logs (scenario-1 for dispatch_log, sum for dispatch_mean) ─────────
        dispatch_log[:, t] = dispatch_t[0]
        dispatch_sum[:, t] = dispatch_t.sum(axis=0)

        # ── Advance state ──────────────────────────────────────────────────────
        # Units OFF this hour → prev_dispatch = 0 so ramp starts from 0 when ON.
        # Units ON → record actual dispatch for ramp continuity.
        np.copyto(prev_dispatch_s, dispatch_t)
        off_this_hour = on_t < 0.5
        if off_this_hour.any():
            prev_dispatch_s[:, off_this_hour] = 0.0
        prev_on_s[:, :] = on_t[np.newaxis, :]

    mean_prices = prices.mean(axis=0)

    return {
        "costs":         fuel_costs + penalty_costs + startup_total,
        "fuel_costs":    fuel_costs,
        "startup_costs": np.full(S, startup_total),
        "penalty_costs": penalty_costs,
        "storage_soc":   storage_soc,
        "hydro_res":     hydro_soc,
        "commitment":    commitment,
        "dispatch_exp":  dispatch_log,              # scenario-1 (backward compat)
        "dispatch_mean": dispatch_sum / max(S, 1),  # mean over scenarios ← NEW
        "storage_log":   storage_log,
        "storage_names": [st.name for st in storages],
        "net_position":  net_position,
        "prices":        prices,
        "mean_prices":   mean_prices,
    }


# ── Coupled multi-region simulation with price-driven market coupling ─────────

def simulate_coupled(
    commitments:  dict[str, np.ndarray],
    fleet:        Fleet,
    scenario_map: dict[str, ScenarioBank],
    network:      Network,
    carryover:    Optional[CarryoverState] = None,
    flow_passes:  int = 2,
) -> dict:
    """
    Simulate all regions jointly with simultaneous price-driven market coupling.

    Algorithm (per hour, per scenario)
    ------------------------------------
    Phase 1 — Local lower-bound dispatch:
        Every committed unit dispatches to its lower bound (min_cap or
        ramp-limited minimum). This is non-negotiable regardless of trade.

    Phase 2 — Price-driven cross-border arbitrage (iterative):
        Compute each region's current marginal offer price.
        For connected pairs where price_A > price_B: route power B→A up to
        the ATC limit. This rebalances regional residual demands.
        Iterate coupling_passes times (3 typically converges).

    Phase 3 — Local stack top-up:
        Each region dispatches its remaining headroom (from cheapest to most
        expensive unit) to cover its post-trade residual demand.

    Phase 4 — Storage and final balance.

    Why simultaneous?
        Sequential (dispatch locally then flow) produces wrong prices. If DE
        dispatches its expensive CCGT first, then FR tries to export cheap
        nuclear, the DE CCGT was already committed — we paid for it twice.
        Simultaneous clearing lets cheap FR nuclear displace DE CCGT before
        DE decides how far up its stack to go.

    No-load cost (implicit):
        A committed unit always pays min_cap * marginal_cost per hour (the
        lower-bound dispatch). Turning it off saves that cost. Combined with
        startup_cost, this creates genuine overnight cycling incentives once
        cheap neighbouring capacity can cover the gap via ATC.
    """
    S       = next(iter(scenario_map.values())).S
    T       = next(iter(scenario_map.values())).T
    regions = sorted(fleet.regions())
    R       = len(regions)
    links   = network.links()

    # ── Step 1: Per-region setup (do NOT dispatch yet) ────────────────────────
    # Build fuel cost matrices, bounds, merit orders once per window.
    reg_data: dict = {}
    for r in regions:
        co       = carryover or CarryoverState()
        free_u   = fleet.free_units(r)
        fixed_u  = fleet.fixed_units(r)
        all_u    = fixed_u + free_u
        storages = fleet.storages(r)
        hydros   = fleet.hydros(r)
        N_all    = len(all_u)
        Ns       = len(storages)

        com = commitments.get(r, np.zeros((len(free_u), T)))
        fixed_com = np.ones((len(fixed_u), T), dtype=np.float64)
        if len(fixed_u) and len(free_u):
            full_com = np.vstack([fixed_com, com.astype(np.float64)])
        elif len(fixed_u):
            full_com = fixed_com
        elif len(free_u):
            full_com = com.astype(np.float64)
        else:
            full_com = np.zeros((0, T), dtype=np.float64)

        avail = np.array([u.availability(T) for u in all_u])
        if full_com.shape[0]:
            full_com = full_com * avail

        fcm      = _build_fuel_cost_matrix(all_u, scenario_map[r], S)
        mean_fc  = fcm.mean(axis=0)
        order    = np.argsort(mean_fc).tolist()

        min_caps_v  = np.array([getattr(u, "min_cap",   0.0) for u in all_u])
        max_caps_v  = np.array([u.max_cap                    for u in all_u])
        ramp_v      = np.array([getattr(u, "ramp_rate", 0.0) for u in all_u])
        has_ramp    = ramp_v > 0
        sc_vec      = np.array([getattr(u, "startup_cost", 0.0) for u in all_u])

        prev_on_arr = np.array([1.0 if co.get_plant_on(u.name, True) else 0.0
                                for u in all_u])
        prev_disp_def = np.zeros(N_all)
        for i, u in enumerate(all_u):
            if prev_on_arr[i] < 0.5:
                prev_disp_def[i] = 0.0
            else:
                rr_i = ramp_v[i]; mc_i = max_caps_v[i]
                prev_disp_def[i] = min_caps_v[i] if (rr_i > 0 and rr_i / max(mc_i, 1) < 0.10) \
                                   else mc_i * 0.7
        prev_dispatch_s = np.tile(np.array([
            co.get_last_dispatch(u.name, prev_disp_def[i])
            for i, u in enumerate(all_u)
        ]), (S, 1))
        prev_on_s = np.tile(prev_on_arr, (S, 1))

        hydro_soc   = {u.name: co.get_reservoir(u.name, S, u.initial_reservoir)
                       for u in hydros}
        hydro_inflow = {u.name: u.inflow_array(T) for u in hydros}
        hydro_cap    = {u.name: u.reservoir_capacity for u in hydros}
        hydro_idx    = {u.name: i for i, u in enumerate(all_u)
                       if isinstance(u, HydroPlant)}
        hydro_list   = [(v, k) for k, v in hydro_idx.items()]

        storage_soc  = {st.name: co.get_soc(st.name, S, st.initial_soc)
                       for st in storages}

        # Startup costs (deterministic)
        startup_total = 0.0
        prev_on_vec   = prev_on_arr.copy()
        for t in range(T):
            on_t = full_com[:, t] if full_com.shape[0] else np.array([])
            starts = (on_t > 0.5) & (prev_on_vec < 0.5)
            startup_total += (sc_vec * starts).sum()
            prev_on_vec = on_t.copy() if len(on_t) else prev_on_vec

        reg_data[r] = {
            "all_u": all_u, "storages": storages, "hydros": hydros,
            "N_all": N_all, "Ns": Ns,
            "full_com": full_com, "fcm": fcm, "mean_fc": mean_fc, "order": order,
            "min_caps_v": min_caps_v, "max_caps_v": max_caps_v,
            "ramp_v": ramp_v, "has_ramp": has_ramp,
            "prev_dispatch_s": prev_dispatch_s, "prev_on_s": prev_on_s,
            "hydro_soc": hydro_soc, "hydro_inflow": hydro_inflow,
            "hydro_cap": hydro_cap, "hydro_idx": hydro_idx, "hydro_list": hydro_list,
            "storage_soc": storage_soc,
            "startup_total": startup_total,
        }

    # ── Step 2: Hour-by-hour coupled dispatch ─────────────────────────────────
    fuel_costs_r    = {r: np.zeros(S) for r in regions}
    penalty_costs_r = {r: np.zeros(S) for r in regions}
    net_position_r  = {r: np.zeros((S, T)) for r in regions}
    prices_r        = {r: np.zeros((S, T)) for r in regions}
    dispatch_log_r  = {r: np.zeros((reg_data[r]["N_all"], T)) for r in regions}
    dispatch_sum_r  = {r: np.zeros((reg_data[r]["N_all"], T)) for r in regions}
    storage_log_r   = {r: np.zeros((reg_data[r]["Ns"], T)) for r in regions}
    flows_log       = np.zeros((S, T, len(links)), dtype=np.float64)

    # Pre-allocate per-region work arrays
    lower_t_r  = {r: np.zeros((S, reg_data[r]["N_all"])) for r in regions}
    upper_t_r  = {r: np.zeros((S, reg_data[r]["N_all"])) for r in regions}
    disp_t_r   = {r: np.zeros((S, reg_data[r]["N_all"])) for r in regions}
    remain_r   = {r: np.zeros(S) for r in regions}

    reg_idx = {r: i for i, r in enumerate(regions)}

    for t in range(T):
        # ── Phase 1: Compute per-region bounds and lower-bound baseline ────────
        marginal_price_r = {r: np.zeros(S) for r in regions}  # current offer price

        for r in regions:
            rd  = reg_data[r]
            N   = rd["N_all"]
            on_t    = rd["full_com"][:, t] if rd["full_com"].shape[0] > 0 else np.array([])
            on_mask = on_t > 0.5 if len(on_t) else np.zeros(N, dtype=bool)
            on_cols = np.where(on_mask)[0]
            demand_t = scenario_map[r].demand_for_region(r)[:, t]  # (S,)

            # Hydro inflow
            for hname, soc in rd["hydro_soc"].items():
                soc += rd["hydro_inflow"][hname][t]
                np.minimum(soc, rd["hydro_cap"][hname], out=soc)

            # Bounds
            lo = lower_t_r[r]; hi = upper_t_r[r]
            lo[:] = 0.0; hi[:] = 0.0
            if len(on_cols):
                pd_on = rd["prev_dispatch_s"][:, on_cols]
                lo_on = rd["min_caps_v"][on_cols]
                hi_on = rd["max_caps_v"][on_cols]
                rr_on = rd["ramp_v"][on_cols]
                hr_on = rd["has_ramp"][on_cols]
                lo_mat = np.where(hr_on, np.maximum(pd_on - rr_on, lo_on), lo_on)
                hi_mat = np.where(hr_on, np.minimum(pd_on + rr_on, hi_on), hi_on)
                for ci, gi in enumerate(on_cols):
                    if rd["hydro_mask"][gi] if "hydro_mask" in rd else gi in rd["hydro_idx"].values():
                        hname = next(k for k, v in rd["hydro_idx"].items() if v == gi)
                        soc_h = rd["hydro_soc"][hname]
                        hi_mat[:, ci] = np.minimum(hi_mat[:, ci], np.maximum(soc_h, 0.0))
                        lo_mat[:, ci] = np.minimum(lo_mat[:, ci], hi_mat[:, ci])
                lo[:, on_cols] = lo_mat; hi[:, on_cols] = hi_mat
                off_cols = np.where(~on_mask)[0]
                if len(off_cols): lo[:, off_cols] = 0.0; hi[:, off_cols] = 0.0
                lb_total = lo_mat.sum(axis=1)
            else:
                lb_total = np.zeros(S)

            # Must-run penalty for lb > demand
            penalty_costs_r[r] += np.maximum(0.0, lb_total - demand_t) * config.MUST_RUN_PENALTY

            # Baseline dispatch at lower bound
            disp_t_r[r][:] = lo
            remain_r[r][:] = np.maximum(demand_t - lb_total, 0.0)

            # Store on_mask for later phases
            rd["_on_mask_t"] = on_mask
            rd["_demand_t"]  = demand_t
            rd["_lb_total_t"] = lb_total

            # Cheapest available marginal price = first unit in merit order with headroom
            for i in rd["order"]:
                if on_mask[i] and (hi[:, i] - lo[:, i]).max() > 1e-3:
                    marginal_price_r[r] = rd["fcm"][:, i]  # (S,)
                    break

        # ── Phase 2: Price-driven cross-border arbitrage ───────────────────────
        # Track available export headroom per region (prevents overcommitment
        # when one region connects to many others simultaneously).
        export_avail = {r: (upper_t_r[r] - disp_t_r[r]).sum(axis=1).copy()
                        for r in regions}  # (S,) per region

        for _ in range(3):  # 3 iterations typically converges
            for l_idx, link in enumerate(links):
                ia = reg_idx[link.region_a]
                ib = reg_idx[link.region_b]
                ra = regions[ia]
                rb = regions[ib]
                pa = marginal_price_r[ra]  # (S,) cheapest offer price in A
                pb = marginal_price_r[rb]  # (S,) cheapest offer price in B

                # B→A: B is cheaper, A wants to import
                flow_ba = np.where(
                    pb * (1.0 + link.loss_factor) < pa,
                    np.clip(
                        np.minimum(np.minimum(export_avail[rb], remain_r[ra]),
                                   link.max_mw_ba),
                        0.0, link.max_mw_ba),
                    0.0
                )
                # A→B: A is cheaper, B wants to import
                flow_ab = np.where(
                    pa * (1.0 + link.loss_factor) < pb,
                    np.clip(
                        np.minimum(np.minimum(export_avail[ra], remain_r[rb]),
                                   link.max_mw_ab),
                        0.0, link.max_mw_ab),
                    0.0
                )

                # Apply flows — update remaining demand AND available export capacity
                recv_a = flow_ba * (1.0 - link.loss_factor)
                recv_b = flow_ab * (1.0 - link.loss_factor)
                remain_r[ra]      = np.maximum(remain_r[ra] - recv_a, 0.0)
                remain_r[rb]      = np.maximum(remain_r[rb] - recv_b, 0.0)
                export_avail[rb]  = np.maximum(export_avail[rb] - flow_ba, 0.0)
                export_avail[ra]  = np.maximum(export_avail[ra] - flow_ab, 0.0)
                flows_log[:, t, l_idx] += flow_ba - flow_ab

        # ── Phase 3: Local stack top-up after trade ────────────────────────────
        for r in regions:
            rd       = reg_data[r]
            on_mask  = rd["_on_mask_t"]
            for i in rd["order"]:
                if not on_mask[i]:
                    continue
                headroom = upper_t_r[r][:, i] - lower_t_r[r][:, i]
                added    = np.minimum(headroom, np.maximum(0.0, remain_r[r]))
                disp_t_r[r][:, i] += added
                remain_r[r]       -= added

        # ── Phase 4: Storage and final balance ─────────────────────────────────
        for r in regions:
            rd = reg_data[r]
            remaining_s = remain_r[r].copy()

            for si, st in enumerate(rd["storages"]):
                soc = rd["storage_soc"][st.name]
                dc  = np.minimum(np.minimum(st.discharge_rate, soc * st.discharge_efficiency),
                                 np.maximum(0.0, remaining_s))
                ch  = np.minimum(np.minimum(st.charge_rate,
                                            (st.energy_capacity - soc) / max(st.charge_efficiency, 1e-9)),
                                 np.maximum(0.0, -remaining_s))
                rd["storage_soc"][st.name] = np.clip(
                    soc - dc / max(st.discharge_efficiency, 1e-9) + ch * st.charge_efficiency,
                    0.0, st.energy_capacity
                )
                remaining_s         -= dc - ch
                fuel_costs_r[r]     += (dc + ch) * st.marginal_cost
                storage_log_r[r][si, t] = float(dc[0] - ch[0])

            penalty_costs_r[r] += np.maximum(0.0,  remaining_s) * config.SCARCITY_PENALTY
            penalty_costs_r[r] += np.maximum(0.0, -remaining_s) * config.CURTAILMENT_PENALTY
            net_position_r[r][:, t] = -remaining_s

            # Fuel costs for this hour
            fuel_costs_r[r] += (disp_t_r[r] * rd["fcm"]).sum(axis=1)

            # MCP: marginal unit in local stack after trade
            on_mask  = rd["_on_mask_t"]
            on_ordered = [i for i in rd["order"] if on_mask[i]]
            if on_ordered:
                used_ord = disp_t_r[r][:, on_ordered] - lower_t_r[r][:, on_ordered]
                fc_on    = rd["fcm"][:, on_ordered]
                mm       = used_ord > 1e-3
                rev      = mm[:, ::-1]
                idx      = rev.argmax(axis=1)
                has_m    = rev.any(axis=1)
                mcp_vals = fc_on[np.arange(S), (len(on_ordered) - 1) - idx]
                mcp_vals = np.where(has_m, mcp_vals, 0.0)
            else:
                mcp_vals = np.zeros(S)

            prices_r[r][:, t] = np.where(net_position_r[r][:, t] < -1.0,
                                          config.SCARCITY_PENALTY, mcp_vals)
            prices_r[r][net_position_r[r][:, t] > 1.0, t] = 0.0

            # Logs
            dispatch_log_r[r][:, t] = disp_t_r[r][0]
            dispatch_sum_r[r][:, t] = disp_t_r[r].sum(axis=0)

            # Hydro water consumption
            for hname, hidx_i in rd["hydro_idx"].items():
                if on_mask[hidx_i]:
                    np.maximum(0.0, rd["hydro_soc"][hname] - disp_t_r[r][:, hidx_i],
                               out=rd["hydro_soc"][hname])

            # Advance state
            on_t = rd["full_com"][:, t] if rd["full_com"].shape[0] else np.array([])
            np.copyto(rd["prev_dispatch_s"], disp_t_r[r])
            off_this = (on_t < 0.5) if len(on_t) else np.ones(rd["N_all"], dtype=bool)
            if off_this.any():
                rd["prev_dispatch_s"][:, off_this] = 0.0
            rd["prev_on_s"][:, :] = (on_t > 0.5)[np.newaxis, :] if len(on_t) \
                                    else np.zeros((S, rd["N_all"]))

    # ── Step 3: Assemble results ───────────────────────────────────────────────
    regional: dict = {}
    for r in regions:
        rd = reg_data[r]
        regional[r] = {
            "costs":         fuel_costs_r[r] + penalty_costs_r[r] + rd["startup_total"],
            "fuel_costs":    fuel_costs_r[r],
            "startup_costs": np.full(S, rd["startup_total"]),
            "penalty_costs": penalty_costs_r[r],
            "storage_soc":   rd["storage_soc"],
            "hydro_res":     rd["hydro_soc"],
            "commitment":    commitments.get(r, np.zeros((0, T))),
            "dispatch_exp":  dispatch_log_r[r],
            "dispatch_mean": dispatch_sum_r[r] / max(S, 1),
            "storage_log":   storage_log_r[r],
            "storage_names": [st.name for st in rd["storages"]],
            "net_position":  net_position_r[r],
            "prices":        prices_r[r],
            "mean_prices":   prices_r[r].mean(axis=0),
            "net_pos_mean":  net_position_r[r].mean(axis=0),
        }

    total = sum(regional[r]["costs"] for r in regions)

    return {
        "regions":     regional,
        "total_costs": total,
        "flows":       flows_log,
        "net_pos":     np.stack([net_position_r[r] for r in regions], axis=2),
    }


# ── CVaR ──────────────────────────────────────────────────────────────────────

def compute_cvar(costs: np.ndarray, alpha: float = 0.05) -> float:
    """Mean of the worst (1-alpha) fraction of cost scenarios."""
    k = max(1, int(np.ceil(len(costs) * alpha)))
    return float(np.partition(costs, -k)[-k:].mean())


# ── Objective builders ────────────────────────────────────────────────────────

def build_objective(
    fleet:        Fleet,
    scenarios:    ScenarioBank,
    region:       str,
    carryover:    Optional[CarryoverState] = None,
    T:            int  = config.WINDOW_HOURS,
    lambda_risk:  float = config.LAMBDA_RISK,
    alpha:        float = config.CVAR_ALPHA,
):
    N_free = len(fleet.free_units(region))

    def objective(solution: np.ndarray) -> float:
        commitment = solution.astype(float).reshape(N_free, T)
        soft_pen   = must_run_soft_penalty(commitment, fleet.free_units(region),
                                           carryover or CarryoverState(), T)
        result = simulate_window(commitment, fleet, scenarios, region,
                                 carryover=carryover)
        costs  = result["costs"]
        return float(costs.mean() + lambda_risk * compute_cvar(costs, alpha) + soft_pen)

    return objective


def build_coupled_objective(
    fleet:        Fleet,
    scenario_map: dict[str, ScenarioBank],
    network:      Network,
    carryover:    Optional[CarryoverState],
    region_sizes: list[tuple[str, int]],
    T:            int,
    lambda_risk:  float = config.LAMBDA_RISK,
    alpha:        float = config.CVAR_ALPHA,
):
    def objective(solution: np.ndarray) -> float:
        commitments    = {}
        soft_pen_total = 0.0
        offset         = 0
        for region, N_free in region_sizes:
            chunk = solution[offset: offset + N_free * T]
            offset += N_free * T
            com    = chunk.astype(float).reshape(N_free, T) if N_free > 0 \
                     else np.zeros((0, T))
            soft_pen_total += must_run_soft_penalty(
                com, fleet.free_units(region), carryover or CarryoverState(), T
            )
            commitments[region] = com

        result = simulate_coupled(commitments, fleet, scenario_map, network,
                                  carryover=carryover)
        costs  = result["total_costs"]
        return float(costs.mean() + lambda_risk * compute_cvar(costs, alpha)
                     + soft_pen_total)

    return objective


# ── Must-run soft penalty ──────────────────────────────────────────────────────

def must_run_soft_penalty(
    commitment:     np.ndarray,
    free_units:     list,
    carryover:      CarryoverState,
    T:              int,
    penalty_per_mw: float = None,
) -> float:
    if penalty_per_mw is None:
        penalty_per_mw = 2.0 * config.MUST_RUN_PENALTY

    total = 0.0
    for i, unit in enumerate(free_units):
        row     = commitment[i]
        min_run = getattr(unit, "min_run_hours", 1)

        hours_on = carryover.get_hours_on(unit.name, 0)
        prev     = 1 if carryover.get_plant_on(unit.name, False) else 0

        for t in range(T):
            curr = int(row[t] > 0.5)
            if curr == 1 and prev == 0:
                hours_on = 1
            elif curr == 1:
                hours_on += 1
            elif curr == 0 and prev == 1:
                if hours_on < min_run:
                    total += (min_run - hours_on) * unit.max_cap * penalty_per_mw
                hours_on = 0
            prev = curr

        if hasattr(unit, "must_run_heat") and unit.must_run_heat is not None:
            obligation = unit.heat_obligation(T)
            for t in range(T):
                if obligation[t] > 0 and row[t] < 0.5:
                    elec = obligation[t] / max(
                        getattr(unit, "power_to_heat_ratio", 1.0), 1e-6)
                    total += elec * penalty_per_mw

    return total


# ── Deprecated (kept for backward compat) ────────────────────────────────────

def _repair_min_run(commitment: np.ndarray, units: list) -> np.ndarray:
    """Deprecated: use must_run_soft_penalty instead."""
    T        = commitment.shape[1]
    repaired = commitment.copy()
    for i, unit in enumerate(units):
        min_run = getattr(unit, "min_run_hours", 1)
        prev_on = 1
        t = 0
        while t < T:
            curr_on = int(repaired[i, t])
            if curr_on == 1 and prev_on == 0:
                repaired[i, t:min(T, t + min_run)] = 1.0
                prev_on = 1
                t += min_run
            else:
                prev_on = curr_on
                t += 1
    return repaired
