"""
succes/julia_ga.py
------------------
Julia-accelerated GA for the unit commitment problem.

Architecture
------------
Python owns:  selection, crossover, mutation, population state
Julia owns:   fitness evaluation of the ENTIRE population each epoch

Instead of calling simulate_coupled() 1800 times per window (30 pop × 60 epochs),
we call Julia ONCE per epoch with all 30 individuals. Julia evaluates them in
parallel using Threads.@threads across the population dimension.

Expected speedup
----------------
Python GA (1800 × 46ms):  ~83s/window
Julia batch (60 × ~5ms):   ~3s/window  (assuming 4+ Julia threads)

The speedup scales with thread count. Set JULIA_NUM_THREADS=8 (or more)
before launching Python to exploit all cores.

Warm initialisation
-------------------
The commitment landscape is nearly monotone (more on = cheaper).
Random 50% initialisation wastes ~20 epochs climbing out of the hole.
We initialise 80% of the population at "all free units on" + random
perturbations, and 20% fully random. This gives near-optimal results
in 20–30 epochs instead of 60.

Soft-penalty must-run is applied in Python before sending to Julia
(it's a deterministic function of the binary string, not a simulation).
"""

from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Optional, Callable

import numpy as np

from .simulator import CarryoverState, compute_cvar, simulate_coupled, must_run_soft_penalty
from . import config
from .julia_bridge import _jl_to_python


def _to_jl_dict(payload: dict):
    """
    Convert a Python payload dict to a juliacall-compatible Dict{Any,Any}.
    juliacall handles plain Python dicts natively on modern versions;
    this wrapper adds the explicit convert() call for robustness.
    """
    try:
        import juliacall
        return juliacall.convert(
            juliacall.Main.seval("Dict{Any,Any}"), payload
        )
    except Exception:
        return payload  # juliacall handles it natively

log = logging.getLogger(__name__)

_BATCH_ENGINE_PATH = Path(__file__).parent / "engine_batch.jl"
_GA_ENGINE_PATH    = Path(__file__).parent / "engine_ga.jl"

# ── juliacall batch engine singleton ─────────────────────────────────────────

_batch_engine = None
_batch_engine_tried = False

_ga_engine = None
_ga_engine_tried = False


def _get_batch_engine():
    """Load engine_batch.jl into Julia once, cache for reuse."""
    global _batch_engine, _batch_engine_tried
    if _batch_engine is not None:
        return _batch_engine
    if _batch_engine_tried:
        return None
    _batch_engine_tried = True

    try:
        import juliacall
        jl = juliacall.newmodule("SuccesBatchHost")
        path_str = str(_BATCH_ENGINE_PATH).replace("\\", "/")
        jl.seval(f'include("{path_str}")')
        jl.seval("import .EngineBatch")
        _batch_engine = jl.EngineBatch
        n_threads = jl.seval("Threads.nthreads()")
        log.info("EngineBatch loaded, %d Julia threads", n_threads)
        return _batch_engine
    except Exception as e:
        log.warning("Could not load EngineBatch: %s", e)
        return None


def _get_ga_engine():
    """Load engine_ga.jl into Julia once, cache for reuse."""
    global _ga_engine, _ga_engine_tried
    if _ga_engine is not None:
        return _ga_engine
    if _ga_engine_tried:
        return None
    _ga_engine_tried = True

    # Ensure batch engine is loaded first (engine_ga.jl includes engine_batch.jl)
    if _get_batch_engine() is None:
        return None

    try:
        import juliacall
        jl = juliacall.newmodule("SuccesGAHost")
        path_str = str(_GA_ENGINE_PATH).replace("\\", "/")
        jl.seval(f'include("{path_str}")')
        jl.seval("import .EngineGA")
        _ga_engine = jl.EngineGA
        log.info("EngineGA loaded")
        return _ga_engine
    except Exception as e:
        log.warning("Could not load EngineGA: %s", e)
        return None


def julia_threads() -> int:
    """Return number of Julia threads available."""
    try:
        engine = _get_batch_engine()
        if engine is None:
            return 1
        import juliacall
        return int(juliacall.Main.seval("Threads.nthreads()"))
    except Exception:
        return 1


# ── Population builder ────────────────────────────────────────────────────────

def _build_seeds(
    region_sizes:      list[tuple[str, int]],
    T:                 int,
    engine,
    payload:           dict,
    rng:               np.random.Generator,
    cyclable_offsets:  list,   # flat bit offsets of ramp-feasible units only
    verbose:           bool = True,
) -> tuple[np.ndarray, float]:
    """
    Build a structured seed population centred on known-good solutions.

    Only units in cyclable_offsets (ramp fast enough to restart before morning
    peak) are seeded for overnight cycling. Slow-ramp lignite/coal that cannot
    restart within the overnight window are excluded — seeding them would cause
    morning scarcity on restart, inflating penalties rather than saving costs.

    Returns
    -------
    seeds     : list of binary arrays
    all_on_fit: fitness of the all-on solution
    """
    n_vars = sum(n * T for _, n in region_sizes)

    # ── All-on baseline ───────────────────────────────────────────────────────
    all_on = np.ones(n_vars, dtype=np.float64)
    payload["population"] = [all_on.tolist()]
    raw = np.array(_jl_to_python(engine.evaluate_population(_to_jl_dict(payload))),
                   dtype=np.float64)
    all_on_fit = float(raw[0])

    seeds = [all_on]   # slot 0 always = all-on

    # ── Single-unit overnight candidates ─────────────────────────────────────
    # Evaluate N_free candidates: all-on except unit i off h0-h9.
    # h0-h9 covers the full overnight valley (profile minimum at h6,
    # morning ramp h6-h9, CCGTs restart in <1h so h9 is safe).
    overnight_start = 1          # h0 always ON — clean carryover at window boundaries
    overnight_end   = min(8, T)   # h1-h7 inclusive (7 hours) — safe for fast coal
    # Cap at T-4: never cycle into last 4 hours — units must be warm at window end
    overnight_end   = min(overnight_end, T - 4)

    # Only cycle units that can physically restart before morning peak.
    # cyclable_offsets filters out slow-ramp lignite/coal (see run_julia_batch_ga).
    single_candidates = []
    for u_off in cyclable_offsets:
        cand = all_on.copy()
        cand[u_off + overnight_start: u_off + overnight_end] = 0.0
        single_candidates.append(cand)

    single_fits = np.full(len(single_candidates), np.inf)
    if single_candidates:
        payload["population"] = [c.tolist() for c in single_candidates]
        raw_c = np.array(_jl_to_python(engine.evaluate_population(_to_jl_dict(payload))),
                         dtype=np.float64)
        single_fits = raw_c

    # Keep beneficial single-unit seeds, sorted by improvement
    single_seeds_ranked = []   # (fitness, candidate, unit_offset)
    for fit, cand, u_off in zip(single_fits, single_candidates, cyclable_offsets):
        if fit < all_on_fit:
            seeds.append(cand)
            single_seeds_ranked.append((fit, cand, u_off))
    single_seeds_ranked.sort(key=lambda x: x[0])   # best first
    n_single_kept = len(single_seeds_ranked)

    # ── Pair-combination seeds ─────────────────────────────────────────────────
    # Take top-K single seeds and evaluate all pairs in one Julia batch call.
    # Pair seed = both units off h0-h9 simultaneously.
    # This directly gives the GA multi-unit cycling patterns to start from.
    MAX_PAIR_UNITS = 15   # top-15 → up to 105 pairs (one Julia call)
    top_single = single_seeds_ranked[:MAX_PAIR_UNITS]
    pair_candidates = []
    pair_offsets = []
    for ia in range(len(top_single)):
        for ib in range(ia + 1, len(top_single)):
            _, cand_a, u_off_a = top_single[ia]
            _, cand_b, u_off_b = top_single[ib]
            # Combine: start from all-on, zero out both overnight blocks
            pair = all_on.copy()
            pair[u_off_a + overnight_start: u_off_a + overnight_end] = 0.0
            pair[u_off_b + overnight_start: u_off_b + overnight_end] = 0.0
            pair_candidates.append(pair)
            pair_offsets.append((u_off_a, u_off_b))

    n_pair_kept = 0
    if pair_candidates:
        best_single_fit = single_seeds_ranked[0][0] if single_seeds_ranked else all_on_fit
        payload["population"] = [c.tolist() for c in pair_candidates]
        raw_p = np.array(_jl_to_python(engine.evaluate_population(_to_jl_dict(payload))),
                         dtype=np.float64)
        for fit, cand in zip(raw_p, pair_candidates):
            if fit < all_on_fit:
                seeds.append(cand)
                n_pair_kept += 1

    if verbose:
        n_all = sum(n for _, n in region_sizes)
        print(f"  [seeds] all-on=€{all_on_fit:,.0f}  "
              f"cyclable={len(cyclable_offsets)}/{n_all}  "
              f"single={n_single_kept}/{len(single_candidates)} kept  "
              f"pairs={n_pair_kept}/{len(pair_candidates)} kept", flush=True)

    return seeds, all_on_fit


def _warm_population(
    n_vars:       int,
    pop_size:     int,
    rng:          np.random.Generator,
    seeds:        list,
    region_sizes: list[tuple[str, int]],
    T:            int,
) -> np.ndarray:
    """
    Build initial population from seeds + structured warm + cold individuals.

    Layout:
      [0]          : all-on (seeds[0])
      [1…len(seeds)]: beneficial single-unit cycling seeds
      [next 30%]   : all-on + one random unit off for a random 4–12h block
      [remainder]  : random 50%
    """
    n_vars_check = sum(n * T for _, n in region_sizes)
    assert n_vars == n_vars_check

    pop = np.zeros((pop_size, n_vars), dtype=np.float64)

    # Insert seeds
    n_seeds = min(len(seeds), pop_size)
    for i in range(n_seeds):
        pop[i] = seeds[i]

    # Structured warm: all-on with one random unit off for random 4–12h block
    n_structured = max(0, int(pop_size * 0.30) - n_seeds)
    # Build list of unit start offsets for random selection
    unit_offsets = []
    offset = 0
    for r, n_free in region_sizes:
        for ui in range(n_free):
            unit_offsets.append(offset + ui * T)
        offset += n_free * T

    all_on = np.ones(n_vars, dtype=np.float64)
    for i in range(n_seeds, n_seeds + n_structured):
        ind = all_on.copy()
        u_off = int(unit_offsets[rng.integers(0, len(unit_offsets))])
        block_len = int(rng.integers(4, 13))           # 4–12 hours
        block_start = int(rng.integers(0, max(1, T - block_len)))
        ind[u_off + block_start: u_off + block_start + block_len] = 0.0
        pop[i] = ind

    # Cold: random 50%
    for i in range(n_seeds + n_structured, pop_size):
        pop[i] = (rng.random(n_vars) > 0.5).astype(np.float64)

    return pop


# ── GA operators ─────────────────────────────────────────────────────────────

def _crossover_block(
    p1: np.ndarray, p2: np.ndarray,
    pc: float, rng: np.random.Generator,
    region_sizes: list, T: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Block crossover: swap whole unit schedules (T-bit blocks) between parents.

    Unlike uniform crossover which shreds overnight-off patterns into noise,
    block crossover preserves each unit's full temporal pattern. Two seeds
    with different single units cycling overnight combine cleanly into a child
    where BOTH units cycle — the multi-unit pattern the GA needs to discover.
    """
    if rng.random() > pc:
        return p1.copy(), p2.copy()
    c1 = p1.copy(); c2 = p2.copy()
    offset = 0
    for _, n_free in region_sizes:
        for ui in range(n_free):
            if rng.random() < 0.5:   # swap this unit's T-bit block
                start = offset + ui * T
                tmp = c1[start:start+T].copy()
                c1[start:start+T] = c2[start:start+T]
                c2[start:start+T] = tmp
        offset += n_free * T
    return c1, c2


def _mutate(
    ind: np.ndarray, pm: float,
    rng: np.random.Generator,
    region_sizes: list = None, T: int = 24,
    p_block: float = 0.20, overnight_end: int = 7,
    cyclable_offsets: list = None,
) -> np.ndarray:
    """
    Two-operator mutation:

    1. Overnight block flip (p_block per individual):
       Randomly picks from cyclable_offsets — units whose ramp rate allows
       restart before morning peak. Slow lignite/coal are excluded.
       Flips the unit's entire h0-overnight_end block (on→off or off→on).

    2. Random bit-flip (pm per bit, ~0.005):
       Fine-grained local exploration without destroying overnight structure.
    """
    out = ind.copy()

    # Operator 1: overnight block flip — only on ramp-feasible units
    if cyclable_offsets and rng.random() < p_block:
        u_start = int(cyclable_offsets[rng.integers(0, len(cyclable_offsets))])
        block_end = min(overnight_end, T)
        # h0 always kept ON (index 0 of unit block) for clean window-boundary carryover
        out[u_start + 1: u_start + block_end] = 1.0 - out[u_start + 1: u_start + block_end]

    # Operator 2: random bit-flip (reduced rate)
    mut = rng.random(len(out)) < pm
    out = np.where(mut, 1.0 - out, out)
    return out


def _tournament_select(pop: np.ndarray, fitness: np.ndarray,
                       k: int, rng: np.random.Generator) -> np.ndarray:
    """Tournament selection, k-way."""
    idx  = rng.integers(0, len(pop), k)
    best = idx[np.argmin(fitness[idx])]
    return pop[best].copy()


# ── Payload builder (reused from JuliaBridge but adapted for batch) ──────────

def _build_batch_payload(
    fleet,
    scenario_map: dict,
    network,
    carryover:    CarryoverState,
    regions:      list[str],
    region_sizes: list[tuple[str, int]],
    T:            int,
    lambda_risk:  float,
    cvar_alpha:   float,
) -> dict:
    """Build the static payload sent to Julia once per epoch."""
    from .julia_bridge import JuliaBridge
    # Reuse the bridge's payload builder (it knows how to serialise everything)
    # We build a fake bridge just for payload construction
    bridge = JuliaBridge(mode="python", verbose=False)
    dummy_commits = {r: np.zeros((n, T)) for r, n in region_sizes}
    payload = bridge._build_payload(
        dummy_commits, fleet, scenario_map, network, carryover, 2
    )

    # Add batch-specific fields
    payload["lambda_risk"]   = float(lambda_risk)
    payload["cvar_alpha"]    = float(cvar_alpha)
    payload["region_sizes"]  = [[n, T] for _, n in region_sizes]

    # Add fixed-unit commitment (always 1, same for all individuals)
    for r in regions:
        fixed_u = fleet.fixed_units(r)
        N_fixed = len(fixed_u)
        if N_fixed > 0:
            avail = np.array([u.availability(T) for u in fixed_u])
            fixed_com = np.ones((N_fixed, T)) * avail
            payload["region_data"][r]["fixed_units_commit"] = fixed_com.tolist()
        else:
            payload["region_data"][r]["fixed_units_commit"] = []

    return payload


# ── Main batch GA ─────────────────────────────────────────────────────────────

def run_julia_batch_ga(
    fleet,
    scenario_map:   dict,
    network,
    carryover:      CarryoverState,
    region_sizes:   list[tuple[str, int]],
    T:              int,
    epochs:         int   = config.GA_EPOCHS,
    pop_size:       int   = config.GA_POP_SIZE,
    pc:             float = config.GA_PC,
    pm:             float = config.GA_PM,
    lambda_risk:    float = config.LAMBDA_RISK,
    cvar_alpha:     float = config.CVAR_ALPHA,
    seed:           int   = 42,
    verbose:        bool  = True,
    prev_best:      "np.ndarray | None" = None,
) -> tuple[np.ndarray, list[float]]:
    """
    Run the GA with Julia batch fitness evaluation.

    Returns
    -------
    best_solution : (n_vars,) float64 binary array
    convergence   : list of best fitness per epoch
    """
    regions   = [r for r, _ in region_sizes]
    n_vars    = sum(n * T for _, n in region_sizes)
    rng       = np.random.default_rng(seed)

    engine = _get_batch_engine()
    if engine is None:
        raise RuntimeError("Julia EngineBatch unavailable — cannot run GA.")

    ga_engine = _get_ga_engine()
    if ga_engine is None:
        raise RuntimeError(
            "Julia GA engine not loaded. Ensure JULIA_NUM_THREADS is set and "
            "the Julia environment is correctly installed."
        )

    # ── Build static payload ──────────────────────────────────────────────────
    payload = _build_batch_payload(
        fleet, scenario_map, network, carryover,
        regions, region_sizes, T, lambda_risk, cvar_alpha,
    )

    # ── Compute cyclable unit offsets ─────────────────────────────────────────
    min_hours_for_restart = max(2, T // 12)
    cyclable_offsets = []
    n_all = 0; n_slow = 0; offset = 0
    from .assets import WindPlant, SolarPlant
    for r, n_free in region_sizes:
        for u in fleet.free_units(r):
            if isinstance(u, (WindPlant, SolarPlant)):
                continue   # RES excluded from GA chromosome
            rr = getattr(u, "ramp_rate", u.max_cap)
            mc = getattr(u, "min_cap", 0)
            if (rr * min_hours_for_restart) >= mc:
                cyclable_offsets.append(offset)
            else:
                n_slow += 1
            n_all += 1; offset += T
    if verbose:
        print(f"  [julia_ga] {len(cyclable_offsets)}/{n_all} units cyclable "
              f"({n_slow} slow-ramp excluded)", flush=True)

    # Add GA-specific fields to payload
    payload["epochs"]           = epochs
    payload["pop_size"]         = pop_size
    payload["pc"]               = float(pc)
    payload["pm"]               = float(pm)
    payload["p_block"]          = 0.20
    payload["migrate_freq"]     = 20
    payload["cyclable_offsets"] = cyclable_offsets
    payload["rng_seed"]         = seed
    payload["verbose"]          = verbose
    payload["overnight_end"]    = min(8, T - 4)
    payload["max_pair_units"]   = getattr(config, "MAX_PAIR_UNITS", 20)
    payload["warm_start_enabled"]  = False
    payload["warm_start_fraction"] = 0.25
    payload["prev_best_solution"]  = None

    # Adaptive epoch stopping: GA runs until improvement < threshold over patience epochs
    payload["adaptive_epochs_enabled"]   = True
    payload["adaptive_patience"]         = 80
    payload["adaptive_min_improvement"]  = 2e-4

    jl_result = ga_engine.run_ga(_to_jl_dict(payload))
    best_sol_raw, conv_raw = jl_result
    best_sol    = np.array(_jl_to_python(best_sol_raw),  dtype=np.float64)
    convergence = [float(x) for x in _jl_to_python(conv_raw)]
    return best_sol, convergence

