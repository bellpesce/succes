# SUCCES

**Stochastic Unit Commitment for Coupled European Scenarios** — a power market
simulator for 15 European bidding zones that uses **no MILP, no LP, no
commercial solver**. Dispatch is found by a genetic algorithm; prices come from
pure merit order: the last dispatched unit sets the market clearing price.

> **Status: research code / show-and-tell.** This is a working experiment, not
> a production tool. It runs, it converges, it produces plausible January-2024
> prices for several zones and implausible ones for others — and it is honest
> about which is which. See [Limitations](#limitations).

## The headline number

A full run of the January 2024 example —

- **15 regions** (AT, BE, CH, CZ, DE, DK, ES, FR, GB, HU, IT, NL, NO, PL, SE)
- **336 hours** (14 rolling 24h windows)
- **15 stochastic scenarios** per window (AR(1) demand noise, spatially
  correlated wind/solar forecast errors, stochastic hydro inflow, per-scenario
  Bernoulli forced outages)
- **351 GA epochs** per window, population 80, two-island model
- CVaR-weighted risk objective, PTDF/ATC cross-border coupling

— completes in **~17 minutes of solve time on a consumer desktop** (8 Julia
threads). Zero scarcity hours. Startup costs land at 2.9% of total
fuel+startup, inside the realistic 1–3% band for European markets.

## Why no MILP?

Not because MILP is bad — it's the industry standard for a reason, and it
gives you optimality bounds this approach cannot. The trade is:

A GA evaluates the **actual nonconvex problem directly**. CVaR objectives,
per-scenario Bernoulli outages, nonlinear penalty structures, and
merit-order price formation are simply *computed* in the fitness function —
nothing has to be linearised, no big-M constraints, no scenario decomposition
machinery. The cost is that you get a good solution with no certificate of
how good. For a *market simulator* validated against observed market
outcomes (rather than a cost-minimisation tool), that trade seemed worth
exploring. This repo is the exploration.

## Architecture

```
Python (orchestration)                 Julia (number crunching)
──────────────────────                 ────────────────────────
examples/europe_nordic_jan2024.py      engine_ga.jl     GA loop: islands,
  fleet, demand, scenario defs                          block crossover,
succes/                                                 warm-start, diversity
  scenarios.py   scenario banks                         restart, adaptive stop
  julia_bridge.py / julia_ga.py        engine_batch.jl  fitness: 4-phase
  solver.py      rolling horizon                        dispatch x S scenarios
  network.py     ATC + PTDF                             x P individuals
  reporter.py    HTML + JSON output    engine.jl        final dispatch, MCP,
  config.py      every tunable knob                     penalty breakdown
```

Wind and solar are handled as **net-load pre-subtraction** ("Approach A"):
their output is computed deterministically from ERA5 capacity factors and
subtracted from gross demand, so the GA chromosome covers thermal + hydro
only. The chromosome is one fraction f ∈ [0,1] per free unit per hour;
f > 0.10 means committed, f × max_cap is the dispatch target. Ramp
constraints are enforced hour-by-hour within each window.

## Quick start

```bash
pip install juliacall numpy
# Julia 1.10 is bootstrapped automatically by juliacall on first run

export JULIA_NUM_THREADS=8          # Windows: set JULIA_NUM_THREADS=8
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes

python examples/europe_nordic_jan2024.py
```

Outputs land next to the example: an interactive HTML report (prices,
dispatch, flows, penalty breakdown per window) and a JSON with every number
in it.

The repo ships the derived ERA5 capacity-factor profiles
(`data/era5_202401_profiles.npz`, ~440 KB) so the example runs out of the
box. The raw ERA5 NetCDF files are not committed; `data/era5_fetch.py`
re-downloads them with a free Copernicus CDS account if you want to build
profiles for other months.

## How well does it match reality?

January 2024 mean day-ahead prices, model vs. real (EUR/MWh):

| Zone | Model | Real | Verdict |
|------|------:|-----:|---------|
| AT   | 82.5  | 84   | good |
| ES   | 47.7  | 52   | good |
| SE   | 62.5  | 52   | acceptable |
| NO   | 76.9  | 48   | too high (hydro spikes) |
| IT   | 56.0  | 128  | too low (ATC too generous on northern borders) |
| DE   | 32.6  | 78   | too low |
| GB   | 36.1  | 90   | too low |
| PL   | 37.1  | 83   | too low |
| FR   | 22.5  | 62   | too low |
| BE   | −1.5  | 71   | broken (nuclear surplus) |

The cluster of too-low zones shares one root cause: French nuclear is
modelled at ~39 GW where real January 2024 availability was 35–38 GW, and
the surplus exports suppress every CWE neighbour. It's a known, documented
calibration issue (one constant in the example file), deliberately left
as-is for this release — the point of the repo is the method, not a
calibrated price forecast.

## Limitations

Read these before quoting any number from this repo.

- **No optimality guarantee.** It's a GA. There is no bound, no gap, no
  certificate. Validation is against observed market outcomes, not a solver.
- **Calibration is half done.** See the table above. FR nuclear capacity,
  BE nuclear min-load, and Italian border ATCs are the known offenders.
- **Cross-border price co-movement is too weak.** The GA chromosome has no
  trade variables; each region balances itself and ATC trades the residual.
  Hourly price correlations between coupled zones come out at 0.1–0.45 where
  reality is 0.7–0.95. Fixing this needs export-fraction genes in the
  chromosome — designed, not built.
- **Storage is dispatched greedily post-GA**, not co-optimised with thermal
  commitment.
- **No strategic bidding.** All capacity is offered at marginal cost, so
  prices are slightly too low and too smooth versus real markets.
- **No minimum down-time**, single zonal price per region, no startup energy
  consumption.

## Citing

There's a `CITATION.cff` in the repo root — GitHub's "Cite this repository"
button gives you BibTeX. If this code or its ideas end up in your work, a
reference is all I ask.

## License

Apache 2.0 — see `LICENSE` and `NOTICE`. Commercial use welcome; the NOTICE
file must travel with redistributions.
