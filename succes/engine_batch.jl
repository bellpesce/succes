"""
succes/engine_batch.jl
----------------------
Batch fitness evaluator for the continuous dispatch GA.

FORMULATION (v3 — Direct Dispatch):
─────────────────────────────────────
The chromosome x[i,t] ∈ [0,1] is the dispatch FRACTION for unit i at hour t.
Actual dispatch MW = x[i,t] × max_cap[i].

The GA directly controls dispatch output — no separate merit-order engine inside
the evaluator. This is a joint UC + ED formulation.

FITNESS = E[total_cost] + λ × CVaR_α(total_cost)

total_cost per scenario s = Σ_t Σ_i (
    fuel_cost[i,t,s]      # x[i,t] × max_cap[i] × mc[i,s]
  + startup_cost[i,t]     # fires when x crosses START_THRESHOLD upward
  + demand_penalty[t,s]   # scarcity when sum(dispatch) < demand
  + curtailment[t,s]      # surplus when sum(dispatch) > demand
  + ramp_penalty[i,t]     # ramp constraint violation
  + reserve_penalty[t,s]  # insufficient spinning reserve
)

WHY THIS FIXES THE COMMITMENT PROBLEM:
  - Before: GA controlled availability (lo/hi bounds), merit order did dispatch.
    c=1.0 always dominant: maximum flexibility for merit order.
  - Now: GA controls actual dispatch MW. Setting x=1.0 for an unneeded unit
    directly adds fuel cost. GA learns: cheap units get high x, expensive units
    get x≈0 except when needed to cover demand.

CROSS-BORDER COUPLING:
  After computing single-region dispatch costs, Phase 2 does price-arbitrage
  ATC/PTDF flows exactly as before. The "price" signal is now the marginal
  cost of the last dispatched unit (from the GA's own dispatch schedule).

STARTUP THRESHOLD: 0.10 (unit is "on" if x > 0.10, regardless of MW level)
RAMP: penalises |Δx × max_cap| > ramp_rate per hour
"""

module EngineBatch

using Statistics: mean, partialsort

# ── Constants ──────────────────────────────────────────────────────────────────
const SCARCITY_PENALTY    = 10_000.0  # EUR/MWh in GA fitness — strongly penalises under-commitment.
                                       # NOTE: only in GA fitness. Reported MCP stays at 3,000 EUR/MWh.
const CURTAILMENT_PENALTY =   200.0   # EUR/MWh surplus after coupling.
                                       # 200 EUR/MWh is the correct value for the GA fitness landscape:
                                       #
                                       # (a) HIGHER than any fuel rate (coal ~80, CCGT ~70 EUR/MWh)
                                       #     -> GA genuinely prefers decommitting over running at min_cap
                                       #     -> overcomes the old 50 EUR/MWh (nuclear offer price) path
                                       #        which made curtailment almost free and caused 95%+ commitment
                                       #
                                       # (b) LOW enough that penalties don't dominate the fitness landscape
                                       #     -> 2000 EUR/MWh caused penalty >> fuel cost every window
                                       #     -> GA couldn't find gradient, all 14 windows hit epoch ceiling
                                       #     -> 200 keeps penalty at ~10-15% of total window cost
                                       #
                                       # Scarcity:curtailment ratio = 10,000 / 200 = 50:1
                                       # GA commits when demand requires it; decommits when it doesn't.
const MUST_RUN_PENALTY    = 2_000.0   # EUR/MWh must-run floor violation.
                                       # Kept at 2000: must-run violations are hard physical constraints
                                       # (lb > demand), not soft economic signals. Making these expensive
                                       # is correct and doesn't cause landscape issues because they're
                                       # rare (only fire when sum(min_caps) > demand).
const MIN_STABLE_PENALTY  = 2_000.0   # EUR/MWh operating below minimum stable generation
                                       # Real generators cannot run between 0 and min_cap.
                                       # This penalty snaps the GA away from the ambiguous
                                       # 10-30% zone that caused 16 GW to sit uncommitted
                                       # in PL/CZ while demand went unserved.
const RAMP_PENALTY        = 2_000.0   # EUR/MWh ramp violation
const RESERVE_PENALTY     =   300.0   # EUR/MWh insufficient spinning reserve
const INERTIA_PENALTY     =   150.0   # EUR/MWh insufficient inertia
const START_THRESHOLD     =     0.10  # x > threshold = unit is "on"

# ── Negative price constants ──────────────────────────────────────────────────
# Negative prices occur when must-run supply (nuclear, wind) exceeds demand.
# NEG_PRICE_SLOPE: EUR/MWh per GW of surplus.  At 5 EUR/GW:
#   10 GW surplus → -50 EUR/MWh  (typical FR overnight nuclear surplus)
#   20 GW surplus → -100 EUR/MWh (deep negative, DK high-wind)
# NEG_PRICE_FLOOR: hard lower bound.  Real markets: EEX cap -500 EUR/MWh.
const NEG_PRICE_SLOPE     =     5.0   # EUR/MWh per GW surplus
const NEG_PRICE_FLOOR     =  -500.0   # EUR/MWh hard floor

# ── PTDF ───────────────────────────────────────────────────────────────────────
struct PTDFData
    L_ac::Int
    R::Int
    matrix::Matrix{Float64}
    ram::Vector{Float64}
end
PTDFData() = PTDFData(0, 0, Matrix{Float64}(undef,0,0), Vector{Float64}())

# ── CVaR ───────────────────────────────────────────────────────────────────────
function cvar(costs::Vector{Float64}, alpha::Float64)::Float64
    k = max(1, round(Int, alpha * length(costs)))
    return mean(partialsort(costs, 1:k; rev=true))
end

# ── RegionData ─────────────────────────────────────────────────────────────────
struct RegionData
    S::Int; T::Int; N::Int; Ns::Int; Nh::Int
    demand::Matrix{Float64}         # (S, T)
    fc_mat::Matrix{Float64}         # (S, N) fuel cost EUR/MWh per scenario
    min_caps::Vector{Float64}       # (N,) MW — minimum dispatch when on
    max_caps::Vector{Float64}       # (N,) MW — maximum dispatch
    ramp_rates::Vector{Float64}     # (N,) MW/h
    startup_costs::Vector{Float64}  # (N,) EUR per cold start
    prev_on_startup::Vector{Float64}# (N,) previous window end dispatch fraction
    must_run_mask::Vector{Bool}     # (N,)
    provides_inertia::Vector{Bool}  # (N,)
    for_rates::Vector{Float64}      # (N,) forced outage rate
    for_stochastic::Bool
    hy_inflow_mult::Matrix{Float64} # (S, Nh) per-scenario inflow multipliers
    offer_prices::Vector{Float64}   # (N,) DA market bid per unit (EUR/MWh)
    inertia_constants::Vector{Float64} # (N,) H seconds per unit (0 for inverter-based)
    h_min_seconds::Float64          # minimum system inertia requirement (seconds)
                                    # wind/solar=0, hydro=water_value, nuclear=-30..-50
    # Storage
    st_cr::Vector{Float64}; st_dr::Vector{Float64}
    st_ce::Vector{Float64}; st_de::Vector{Float64}
    st_cap::Vector{Float64}; st_mc::Vector{Float64}
    st_soc0::Matrix{Float64}        # (S, Ns)
    # Hydro
    hy_unit_idx::Vector{Int}        # (Nh,) which unit index this hydro maps to
    hy_inflow::Matrix{Float64}      # (Nh, T)
    hy_cap::Vector{Float64}         # (Nh,) reservoir capacity MWh
    hy_soc0::Matrix{Float64}        # (S, Nh)
    prev_disp::Matrix{Float64}      # (S, N) last-hour dispatch from carryover
    order::Vector{Int}              # merit order (cheapest first, by offer_price)
end

# ── Link data ──────────────────────────────────────────────────────────────────
struct LinkData
    ia::Int; ib::Int
    atc_ab::Float64; atc_ba::Float64
    loss::Float64
end

# ── Helpers ────────────────────────────────────────────────────────────────────
function to_vec(v)
    isempty(v) && return Float64[]
    return Float64[Float64(x) for x in v]
end
function to_bvec(v)
    isempty(v) && return Bool[]
    return Bool[Bool(x) for x in v]
end
function to_mat(v, rows, cols)
    m = Matrix{Float64}(undef, rows, cols)
    for i in 1:rows, j in 1:cols
        m[i,j] = Float64(v[i][j])
    end
    return m
end
function to_fvec_or_zeros(d, key, n)
    haskey(d, key) && !isempty(d[key]) && return Float64[Float64(x) for x in d[key]]
    return zeros(Float64, n)
end

# ── RegionData constructor from payload dict ───────────────────────────────────
function build_region_data(rd::Dict, S::Int, T::Int)::RegionData
    # N: total units = length of min_caps (always sent by julia_bridge.py)
    N = length(rd["min_caps"])

    # N_fixed: from fixed_units_commit rows
    fc_raw_fix = get(rd, "fixed_units_commit", [])
    N_fixed    = length(fc_raw_fix)

    # Storage names → Ns
    st_names = String[String(n) for n in get(rd, "storage_names", [])]
    Ns = length(st_names)

    # Hydro names → Nh  (key "hydro_names" from julia_bridge)
    hy_names = String[String(n) for n in get(rd, "hydro_names", [])]
    Nh = length(hy_names)

    # Demand (S, T)
    demand = to_mat(rd["demand"], S, T)

    # Fuel cost matrix — julia_bridge sends "fuel_cost_matrix" (S×N list-of-lists)
    fc_raw = rd["fuel_cost_matrix"]
    fc_mat = Matrix{Float64}(undef, S, N)
    for s in 1:S, i in 1:N
        fc_mat[s,i] = Float64(fc_raw[s][i])
    end

    # Storage dicts (julia_bridge keys)
    st_cr_d  = Ns > 0 ? Dict{String,Float64}(k => Float64(v) for (k,v) in rd["storage_charge_rate"]) : Dict{String,Float64}()
    st_dr_d  = Ns > 0 ? Dict{String,Float64}(k => Float64(v) for (k,v) in rd["storage_discharge_rate"]) : Dict{String,Float64}()
    st_ce_d  = Ns > 0 ? Dict{String,Float64}(k => Float64(v) for (k,v) in rd["storage_charge_eff"]) : Dict{String,Float64}()
    st_de_d  = Ns > 0 ? Dict{String,Float64}(k => Float64(v) for (k,v) in rd["storage_discharge_eff"]) : Dict{String,Float64}()
    st_cap_d = Ns > 0 ? Dict{String,Float64}(k => Float64(v) for (k,v) in rd["storage_capacity"]) : Dict{String,Float64}()
    st_mc_d  = Ns > 0 ? Dict{String,Float64}(k => Float64(v) for (k,v) in rd["storage_marginal"]) : Dict{String,Float64}()

    # Storage SOC: dict {name → [S floats]} → (S, Ns) matrix
    soc0 = if Ns > 0
        d = rd["storage_soc"]
        m = Matrix{Float64}(undef, S, Ns)
        for (ni, nm) in enumerate(st_names)
            sv = d[nm]; for s in 1:S; m[s,ni] = Float64(sv[s]); end
        end
        m
    else
        Matrix{Float64}(undef, S, 0)
    end

    # Hydro inflow: dict {name → [T floats]} → (Nh, T) matrix
    hy_inflow_mat = if Nh > 0
        d = rd["hydro_inflow"]
        m = Matrix{Float64}(undef, Nh, T)
        for (hi, hn) in enumerate(hy_names)
            v = d[hn]; for t in 1:T; m[hi,t] = Float64(v[t]); end
        end
        m
    else
        Matrix{Float64}(undef, 0, T)
    end

    # Hydro capacity: dict {name → float} → Vector
    hy_cap_v = Nh > 0 ?
        Float64[Float64(rd["hydro_capacity"][hn]) for hn in hy_names] : Float64[]

    # hydro_unit_idx: dict {name → int (1-based)} → Vector{Int}
    hy_unit_idx = Nh > 0 ?
        Int[Int(rd["hydro_unit_idx"][hn]) for hn in hy_names] : Int[]

    # Hydro SOC: dict {name → [S floats]} → (S, Nh) matrix
    hy_soc0 = if Nh > 0
        d = rd["hydro_soc"]
        m = Matrix{Float64}(undef, S, Nh)
        for (hi, hn) in enumerate(hy_names)
            sv = d[hn]; for s in 1:S; m[s,hi] = Float64(sv[s]); end
        end
        m
    else
        Matrix{Float64}(undef, S, 0)
    end

    # prev_dispatch (S×N list-of-lists) — key "prev_dispatch"
    prev_d_raw = get(rd, "prev_dispatch", nothing)
    prev_disp  = if prev_d_raw !== nothing
        m = Matrix{Float64}(undef, S, N)
        for s in 1:S, i in 1:N; m[s,i] = Float64(prev_d_raw[s][i]); end
        m
    else
        zeros(Float64, S, N)
    end

    # Per-scenario hydro inflow multipliers
    hm = get(rd, "hydro_inflow_mult", [])
    hy_inflow_mult = (!isempty(hm) && !isempty(hm[1])) ?
        let Nh_h = length(hm); S_h = length(hm[1])
            mat = Matrix{Float64}(undef, S_h, Nh_h)
            for hi in 1:Nh_h, si in 1:S_h; mat[si,hi] = Float64(hm[hi][si]); end
            mat
        end : Matrix{Float64}(undef, 0, 0)

    max_caps_v   = to_fvec_or_zeros(rd, "max_caps",         N)
    min_caps_v   = to_fvec_or_zeros(rd, "min_caps",         N)
    ramp_rates_v = to_fvec_or_zeros(rd, "ramp_rates",       N)
    startup_v    = to_fvec_or_zeros(rd, "startup_cost_vec", N)  # julia_bridge key
    prev_on_v    = to_fvec_or_zeros(rd, "prev_on_startup",  N)
    offer_p_v    = to_fvec_or_zeros(rd, "offer_prices",       N)
    inertia_v    = to_fvec_or_zeros(rd, "inertia_constants",  N)
    h_min        = Float64(get(rd, "h_min_seconds", 3.5))

    # Merit order sorted by offer_price (cheapest first).
    # With Approach A (RES as net-load), this list is thermal + hydro only.
    # Nuclear:          offer_price = -30..-50 → sorted first (most negative).
    # Hydro:            offer_price = water_value (30-155) → mid-merit.
    # CCGT/OCGT:        offer_price = fuel MC (89-150) → expensive end.
    merit_order = sortperm(offer_p_v)   # cheapest (most negative) first

    RegionData(
        S, T, N, Ns, Nh,
        demand, fc_mat,
        min_caps_v, max_caps_v, ramp_rates_v, startup_v,
        prev_on_v,
        to_bvec(get(rd, "must_run_mask",    fill(false, N))),
        to_bvec(get(rd, "provides_inertia", fill(false, N))),
        to_fvec_or_zeros(rd, "for_rates", N),
        Bool(get(rd, "for_stochastic", false)),
        hy_inflow_mult,
        offer_p_v,
        inertia_v,
        h_min,
        Ns > 0 ? Float64[Float64(st_cr_d[n]) for n in st_names] : Float64[],
        Ns > 0 ? Float64[Float64(st_dr_d[n]) for n in st_names] : Float64[],
        Ns > 0 ? Float64[Float64(st_ce_d[n]) for n in st_names] : Float64[],
        Ns > 0 ? Float64[Float64(st_de_d[n]) for n in st_names] : Float64[],
        Ns > 0 ? Float64[Float64(st_cap_d[n]) for n in st_names] : Float64[],
        Ns > 0 ? Float64[Float64(st_mc_d[n])  for n in st_names] : Float64[],
        soc0,
        hy_unit_idx,
        hy_inflow_mat, hy_cap_v, hy_soc0,
        prev_disp,
        merit_order,
    )
end


# ── ATC cross-border flow phase ────────────────────────────────────────────────
# Same bilateral ATC logic as before, used after single-region dispatch cost.
function cross_border_price_arbitrage!(
    net_pos::Matrix{Float64},        # (S, R) net position (+ = surplus)
    marginal_p::Matrix{Float64},     # (S, R) marginal price per region
    export_avail::Matrix{Float64},   # (S, R) export headroom
    links::Vector{LinkData},
    S::Int, n_passes::Int,
    ptdf::PTDFData,
    net_export_buf::Vector{Float64},
)
    n_links = length(links)
    n_links == 0 && return

    for _ in 1:n_passes
        for lk in links
            ia = lk.ia; ib = lk.ib; loss = lk.loss
            @inbounds for s in 1:S
                pa = marginal_p[s,ia]; pb = marginal_p[s,ib]
                if pb*(1.0+loss) < pa
                    flow = clamp(min(export_avail[s,ib], max(0.0,-net_pos[s,ia])),
                                 0.0, lk.atc_ba)
                    net_pos[s,ia]        += flow*(1.0-loss)
                    net_pos[s,ib]        -= flow
                    export_avail[s,ib]   -= flow
                elseif pa*(1.0+loss) < pb
                    flow = clamp(min(export_avail[s,ia], max(0.0,-net_pos[s,ib])),
                                 0.0, lk.atc_ab)
                    net_pos[s,ib]        += flow*(1.0-loss)
                    net_pos[s,ia]        -= flow
                    export_avail[s,ia]   -= flow
                end
            end
        end

        # PTDF line-loading cap
        if ptdf.L_ac > 0
            R_ptdf = ptdf.R
            @inbounds for s in 1:S
                for r in 1:R_ptdf; net_export_buf[r] = -net_pos[s,r]; end
                scale = 1.0
                for l_ac in 1:ptdf.L_ac
                    loading = 0.0
                    for r in 1:R_ptdf; loading += ptdf.matrix[l_ac,r]*net_export_buf[r]; end
                    abs_load = abs(loading)
                    if abs_load > ptdf.ram[l_ac] + 1e-3
                        s_l = ptdf.ram[l_ac]/abs_load
                        if s_l < scale; scale = s_l; end
                    end
                end
                if scale < 1.0 - 1e-6
                    for r in 1:R_ptdf
                        net_pos[s,r] *= scale
                        export_avail[s,r] /= max(scale, 1e-9)
                    end
                end
            end
        end
    end
end

# ── Core fitness function ──────────────────────────────────────────────────────
"""
evaluate_individual(regions, region_data, fixed_commits, region_sizes, links,
                    x, S, T, R, lambda_r, alpha, flow_passes,
                    costs_buf, marginal_p_buf, remain_2d, export_avail_buf,
                    ptdf, net_export_buf) → Float64

x is the continuous chromosome: values in [0,1] representing dispatch fraction.
x[offset + (unit_idx-1)*T + (t-1)] = dispatch fraction for free unit unit_idx at hour t.

Fixed units (nuclear, biomass etc.) are always dispatched at 1.0.
"""
function evaluate_individual(
    regions, region_data, fixed_commits, region_sizes, links,
    x, S, T, R, lambda_r, alpha, flow_passes,
    costs_buf,
    marginal_p_buf,
    remain_2d,
    export_avail_buf,
    ptdf::PTDFData,
    net_export_buf::Vector{Float64},
)::Float64

    # ── Build dispatch fraction matrices from chromosome ───────────────────────
    # dispatch_frac[ri][i,t] ∈ [0,1]: fraction of max_cap dispatched
    dispatch_frac = Vector{Matrix{Float64}}(undef, R)
    offset = 1
    for (ri, r) in enumerate(regions)
        N_free, _ = region_sizes[ri]
        rd = region_data[r]
        N  = rd.N
        N_fixed = size(fixed_commits[r], 1)
        df = Matrix{Float64}(undef, N, T)
        # Fixed units: always 1.0
        for i in 1:N_fixed, t in 1:T; df[i,t] = 1.0; end
        # Free units: from chromosome
        for j in 1:N_free*T
            ui = N_fixed + div(j-1, T) + 1
            tt = mod(j-1, T) + 1
            df[ui,tt] = clamp(Float64(x[offset+j-1]), 0.0, 1.0)
        end
        offset += N_free * T
        dispatch_frac[ri] = df
    end

    fill!(costs_buf, 0.0)

    # ── Market coupling pre-pass ───────────────────────────────────────────────
    # Compute cross-border flows using PTDF/ATC based on merit-order prices.
    # Adjust each region's effective demand by net imports before the cost loop.
    # This gives the GA the correct incentive: cheap regions must dispatch more
    # to cover their export obligations; expensive regions dispatch less.
    # The PTDF correctly constrains physical line loading — no new GA variables.

    # demand_adj[ri, s, t] = net import into region ri in scenario s at hour t
    # Positive = ri received imports → effective demand decreases
    demand_adj_arr = zeros(Float64, R, S, T)

    if length(links) > 0
        # Build export headroom and merit-order prices per region per scenario per hour
        est_price_arr    = zeros(Float64, R, S, T)
        avail_export_arr = zeros(Float64, R, S, T)

        for (ri, r) in enumerate(regions)
            rd = region_data[r]
            N  = rd.N
            df = dispatch_frac[ri]   # (N, T) from chromosome — already built above
            @inbounds for t in 1:T
                for s in 1:S
                    d = rd.demand[s, t]
                    # Merit-order estimated price for ATC routing.
                    # Walk the merit order cheapest first; the last unit whose
                    # capacity is needed to cover demand sets the price.
                    # RES units (offer_price=0) are sorted first — if wind alone
                    # covers demand, est_price = 0 EUR/MWh.
                    # If demand exceeds all available capacity → est_price stays
                    # at the most expensive unit's offer price (scarcity signal).
                    # If generation exceeds demand (surplus): the cheapest unit
                    # is being curtailed; its offer_price is the marginal price
                    # (typically 0 for wind, -50 for nuclear). This correctly
                    # signals that the surplus region is cheap → ATC routes
                    # exports to positive-price deficit regions.
                    max_d    = 0.0
                    est_p    = 0.0
                    for idx in rd.order   # cheapest first
                        f = df[idx, t]
                        f > 0.001 || continue
                        unit_cap = f * rd.max_caps[idx]
                        if unit_cap < 1e-3; continue; end
                        est_p = rd.offer_prices[idx]   # this unit's bid
                        max_d += unit_cap
                        if max_d >= d; break; end
                    end
                    # In surplus: est_p is already the cheapest unit's offer price
                    # (wind=0, nuclear=-50) — correct market signal, no slope needed.
                    est_price_arr[ri, s, t] = est_p
                    avail_export_arr[ri, s, t] = max(0.0, max_d - d)
                    avail_export_arr[ri, s, t] = min(avail_export_arr[ri, s, t], d * 0.40)
                end
            end
        end

        net_export_pre = zeros(Float64, max(ptdf.R, 1))
        @inbounds for t in 1:T
            net_pos_pre   = zeros(Float64, S, R)
            marg_p_pre    = zeros(Float64, S, R)
            exp_avail_pre = zeros(Float64, S, R)
            for ri in 1:R, s in 1:S
                marg_p_pre[s,ri]    = est_price_arr[ri, s, t]
                exp_avail_pre[s,ri] = avail_export_arr[ri, s, t]
            end

            for _ in 1:flow_passes
                for lk in links
                    ia = lk.ia; ib = lk.ib; loss = lk.loss
                    @inbounds for s in 1:S
                        pa = marg_p_pre[s,ia]; pb = marg_p_pre[s,ib]
                        if pb*(1.0+loss) < pa
                            flow = max(0.0, min(exp_avail_pre[s,ib], lk.atc_ba))
                            net_pos_pre[s,ia]   += flow*(1.0-loss)
                            net_pos_pre[s,ib]   -= flow
                            exp_avail_pre[s,ib] -= flow
                        elseif pa*(1.0+loss) < pb
                            flow = max(0.0, min(exp_avail_pre[s,ia], lk.atc_ab))
                            net_pos_pre[s,ib]   += flow*(1.0-loss)
                            net_pos_pre[s,ia]   -= flow
                            exp_avail_pre[s,ia] -= flow
                        end
                    end
                end
                # PTDF line-loading cap
                if ptdf.L_ac > 0
                    Rp = ptdf.R
                    @inbounds for s in 1:S
                        for ri in 1:Rp; net_export_pre[ri] = -net_pos_pre[s,ri]; end
                        scale = 1.0
                        for l_ac in 1:ptdf.L_ac
                            loading = 0.0
                            for ri in 1:Rp
                                loading += ptdf.matrix[l_ac,ri]*net_export_pre[ri]
                            end
                            abs_load = abs(loading)
                            if abs_load > ptdf.ram[l_ac] + 1e-3
                                s_l = ptdf.ram[l_ac]/abs_load
                                if s_l < scale; scale = s_l; end
                            end
                        end
                        if scale < 1.0 - 1e-6
                            for ri in 1:R
                                net_pos_pre[s,ri] *= scale
                                exp_avail_pre[s,ri] /= max(scale, 1e-9)
                            end
                        end
                    end
                end
            end

            for ri in 1:R, s in 1:S
                demand_adj_arr[ri, s, t] = net_pos_pre[s, ri]
            end
        end
    end

    # ── Startup costs (proportional |Δx| smoothing) ────────────────────────────
    # Old behaviour: binary — full startup cost fires whenever x crosses 0.10 upward.
    # Problem: Gaussian mutation (σ=0.15) randomly crosses the threshold each epoch,
    # charging full 120,000–180,000 EUR per spurious crossing → startup = 14% of total.
    #
    # Fix: startup cost is proportional to the size of the upward crossing:
    #   cost = startup_cost × clamp(Δx / (1 - START_THRESHOLD), 0, 1)
    # where Δx = f_now - f_prev when crossing from below threshold to above.
    #
    # Effect:
    #   • A mutation that nudges x from 0.09 → 0.11 (Δx=0.02) costs only 2.2% of
    #     the full startup — trivial, so the GA stops avoiding threshold crossings.
    #   • Committing a unit from cold (x=0 → x=0.8) costs 78% of startup — realistic.
    #   • Committing to full output (x=0 → x=1.0) costs 100% — unchanged.
    #   • The continuous cost gradient removes the sharp cliff that Gaussian mutation
    #     was bouncing off, so the GA explores dispatch fractions freely.
    #
    # The previous window's final dispatch fraction (prev_on_startup) is the
    # correct boundary condition — unchanged.
    for (ri, r) in enumerate(regions)
        rd = region_data[r]
        startup_total = 0.0
        prev_f = copy(rd.prev_on_startup)
        @inbounds for t in 1:T
            for i in 1:rd.N
                f_now  = dispatch_frac[ri][i,t]
                f_prev = prev_f[i]
                if f_now > START_THRESHOLD && f_prev <= START_THRESHOLD
                    # Proportional crossing: how far above threshold did we jump?
                    # Δx relative to the range [START_THRESHOLD, 1.0]
                    crossing_fraction = clamp(
                        (f_now - START_THRESHOLD) / max(1.0 - START_THRESHOLD, 1e-6),
                        0.0, 1.0
                    )
                    startup_total += rd.startup_costs[i] * crossing_fraction
                end
                prev_f[i] = f_now
            end
        end
        @inbounds for s in 1:S; costs_buf[s,ri] += startup_total; end
    end

    # ── Per-hour costs and cross-border coupling ───────────────────────────────
    # prev_hour_disp tracks actual dispatch MW per unit per scenario for the
    # PREVIOUS HOUR within this window. This is what the ramp constraint must
    # compare against. rd.prev_disp is only valid at t=1 (window boundary carryover).
    # Initialise from carryover; update each hour after dispatch is computed.
    prev_hour_disp = Vector{Matrix{Float64}}(undef, R)
    for (ri, r) in enumerate(regions)
        rd = region_data[r]
        # Copy carryover: (S, N) — used at t=1
        prev_hour_disp[ri] = copy(rd.prev_disp)   # (S, N)
    end

    @inbounds for t in 1:T

        # ── Per-region dispatch cost ───────────────────────────────────────────
        for (ri, r) in enumerate(regions)
            rd  = region_data[r]
            N   = rd.N; Nh = rd.Nh
            df_t = view(dispatch_frac[ri], :, t)

            # Hydro inflow
            @inbounds for h in 1:Nh
                base_inflow = rd.hy_inflow[h, t]
                if size(rd.hy_inflow_mult, 1) > 0 && h <= size(rd.hy_inflow_mult, 2)
                    @inbounds for s in 1:S
                        remain_2d[s,ri] = 0.0   # temp: reuse as hy_s scratch below
                    end
                end
            end

            # Compute actual dispatch MW per unit per scenario
            @inbounds for s in 1:S
                total_dispatch = 0.0
                marginal_cost  = 0.0
                max_dispatch   = 0.0  # for export headroom

                for i in 1:N
                    f = df_t[i]
                    f > 0.001 || continue

                    # FOR outage: same deterministic hash as before
                    if rd.for_stochastic && rd.for_rates[i] > 0.0
                        for_seed = UInt32((i * 1000003 + t * 999983) % typemax(UInt32))
                        if ((for_seed % 100000) / 100000.0) < rd.for_rates[i]
                            continue
                        end
                    end

                    # Actual dispatch MW
                    disp_mw = f * rd.max_caps[i]

                    # Ramp constraint: compare against PREVIOUS HOUR dispatch.
                    # At t=1: prev_hour_disp == rd.prev_disp (window carryover).
                    # At t>1: prev_hour_disp was updated after the prior iteration.
                    # This correctly enforces ramp limits across ALL hours, not just t=1.
                    rr = rd.ramp_rates[i]
                    if rr > 0.0
                        prev_mw = prev_hour_disp[ri][s, i]
                        ramp_viol = max(0.0, abs(disp_mw - prev_mw) - rr)
                        if ramp_viol > 1e-3
                            costs_buf[s,ri] += ramp_viol * RAMP_PENALTY
                        end
                    end

                    # Min-cap penalty: if on but below min_cap
                    mc_i = rd.min_caps[i]
                    if f > START_THRESHOLD && disp_mw < mc_i - 1e-3
                        costs_buf[s,ri] += (mc_i - disp_mw) * MUST_RUN_PENALTY
                    end

                    # Minimum stable generation penalty.
                    # A unit operating between START_THRESHOLD and min_cap is in
                    # a physically infeasible zone — real generators either run at
                    # or above minimum stable generation, or stay off entirely.
                    # This penalty pushes the GA to make clean commit decisions:
                    # either dispatch at ≥ min_cap (committed) or at 0 (off).
                    # Without this, units hover at 10-20% dispatch fractions in
                    # peak hours, contributing little capacity while suppressing
                    # full commitment of the unit — the root cause of PL/CZ scarcity.
                    # The penalty zone is START_THRESHOLD < f < min_cap/max_cap.
                    min_frac = mc_i / max(rd.max_caps[i], 1e-3)
                    if f > START_THRESHOLD && f < min_frac - 1e-3 && mc_i > 1e-3
                        # Unit is nominally "on" but below minimum stable generation
                        # Cost: proportional to distance from min_stable point
                        below_min = (min_frac - f) * rd.max_caps[i]
                        costs_buf[s,ri] += below_min * MIN_STABLE_PENALTY
                    end

                    # Fuel cost
                    fuel_eur = disp_mw * rd.fc_mat[s,i]
                    costs_buf[s,ri] += fuel_eur
                    total_dispatch  += disp_mw
                    max_dispatch    += rd.max_caps[i]  # headroom for export

                    # Track marginal cost (highest-cost dispatching unit)
                    if f > START_THRESHOLD && rd.fc_mat[s,i] > marginal_cost
                        marginal_cost = rd.fc_mat[s,i]
                    end
                end

                # Effective demand = raw demand MINUS market coupling import
                eff_demand = rd.demand[s,t] - demand_adj_arr[ri, s, t]
                remain_2d[s,ri]        = total_dispatch - eff_demand
                marginal_p_buf[s,ri]   = marginal_cost
                export_avail_buf[s,ri] = max(0.0, max_dispatch - total_dispatch)

                # ── Inertia constraint ────────────────────────────────────────
                # System inertia = sum(H_i * dispatch_MW_i) for all synchronous
                # units currently dispatched above START_THRESHOLD.
                # Required: inertia >= h_min * demand
                # When violated: GA is penalised to keep synchronous plant online
                # even when it appears uneconomic from an energy-only perspective.
                # This prevents the all-wind dispatch that leaves no thermal
                # prewarmed for the next peak hour.
                if rd.h_min_seconds > 0.0
                    sys_inertia = 0.0
                    @inbounds for i in 1:N
                        f = df_t[i]
                        f > START_THRESHOLD || continue
                        H_i = rd.inertia_constants[i]
                        H_i > 0.0 || continue
                        sys_inertia += H_i * f * rd.max_caps[i]
                    end
                    required_inertia = rd.h_min_seconds * eff_demand
                    if sys_inertia < required_inertia - 1e-3
                        inertia_shortfall = required_inertia - sys_inertia
                        costs_buf[s,ri] += inertia_shortfall * INERTIA_PENALTY
                    end
                end
            end
        end

        # ── Update prev_hour_disp for ramp tracking ───────────────────────────
        # After processing all regions and scenarios at hour t, record each unit's
        # actual dispatch so the next hour's ramp check uses the correct baseline.
        for (ri, r) in enumerate(regions)
            rd  = region_data[r]
            N   = rd.N
            df_t_update = view(dispatch_frac[ri], :, t)
            for s in 1:S
                for i in 1:N
                    f = df_t_update[i]
                    prev_hour_disp[ri][s, i] = f > 0.001 ? f * rd.max_caps[i] : 0.0
                end
            end
        end

        # ── ATC / PTDF cross-border price arbitrage ────────────────────────────
        # remain_2d[s,r] = net position (+ = surplus available to export)
        # After arbitrage: imbalances are reduced by cross-border flows.
        #
        # COUPLING REWARD: snapshot remain_2d before ATC, then after ATC
        # compute Δremain = flow received. Reward = flow × price_differential.
        # This gives the GA an incentive to dispatch extra in cheap regions
        # and export the surplus to expensive neighbours.
        # Without this reward, the GA dispatches ≈ demand everywhere and
        # the ATC has nothing to trade.
        if length(links) > 0 && flow_passes > 0
            cross_border_price_arbitrage!(
                remain_2d, marginal_p_buf, export_avail_buf,
                links, S, flow_passes, ptdf, net_export_buf,
            )
        end

        # ── Demand balance penalties after coupling ───────────────────────────
        @inbounds for (ri, r) in enumerate(regions)
            rd = region_data[r]
            @inbounds for s in 1:S
                imbal = remain_2d[s,ri]
                d = rd.demand[s,t]
                if imbal < -2.0
                    # Unserved demand exceeds 2 MW dead-band → scarcity penalty.
                    # Matches engine.jl MCP threshold: sub-2 MW gaps are numerical
                    # noise and should not trigger a full 3000 EUR/MWh signal in
                    # the GA fitness function either.
                    costs_buf[s,ri] += (-imbal) * SCARCITY_PENALTY
                elseif imbal > 1e-3
                    # Surplus: thermal dispatch exceeds net demand after coupling.
                    # Cost the GA sees for over-committing.
                    #
                    # PREVIOUSLY: curtail_cost = abs(cheapest_offer_price) * imbal
                    # For nuclear at -50 EUR/MWh this meant surplus cost only 50 EUR/MWh.
                    # Against SCARCITY_PENALTY=10,000 the ratio was 200:1 → GA always
                    # over-committed, running all units at 95%+ regardless of demand.
                    #
                    # NOW: use CURTAILMENT_PENALTY = 2,000 EUR/MWh directly.
                    # This gives a 5:1 ratio (10,000 scarcity : 2,000 curtailment)
                    # which is a realistic market signal: curtailment is genuinely costly
                    # (wasted fuel, wear on equipment) but cheaper than load shedding.
                    # The GA now has real incentive to decommit cheap units in low-demand
                    # hours rather than running everything and dumping the surplus.
                    costs_buf[s,ri] += imbal * CURTAILMENT_PENALTY
                end

                # Spinning reserve: 10% of demand (unchanged)
                min_reserve = 0.10 * d
                if export_avail_buf[s,ri] < min_reserve - 1e-3
                    costs_buf[s,ri] += (min_reserve - export_avail_buf[s,ri]) * RESERVE_PENALTY
                end
            end
        end

    end  # hour loop

    # ── Aggregate across scenarios → CVaR objective ───────────────────────────
    total_costs = vec(sum(costs_buf, dims=2))
    return mean(total_costs) + lambda_r * cvar(total_costs, alpha)
end


# ── Batch evaluator (called from Python via evaluate_population) ───────────────
function evaluate_population(payload)::Vector{Float64}
    payload  = Dict{Any,Any}(payload)
    S        = Int(payload["S"])
    T        = Int(payload["T"])
    regions  = String[String(r) for r in payload["regions"]]
    R        = length(regions)
    lambda_r = Float64(get(payload, "lambda_risk", 0.15))
    alpha    = Float64(get(payload, "cvar_alpha",  0.05))
    passes   = Int(get(payload, "flow_passes",     2))

    rd_outer    = Dict{Any,Any}(payload["region_data"])
    region_data = Dict{String, RegionData}(
        r => build_region_data(Dict{Any,Any}(rd_outer[r]), S, T)
        for r in regions
    )

    fixed_commits = Dict{String, Matrix{Float64}}()
    for r in regions
        fc_raw  = Dict{Any,Any}(rd_outer[r])["fixed_units_commit"]
        N_fixed = length(fc_raw)
        fixed_commits[r] = N_fixed > 0 ?
            to_mat(fc_raw, N_fixed, T) : Matrix{Float64}(undef, 0, T)
    end

    links = LinkData[]
    reg_idx = Dict{String,Int}(r => i for (i,r) in enumerate(regions))
    if haskey(payload, "links")
        for lk in payload["links"]
            lk_d = Dict{Any,Any}(lk)
            push!(links, LinkData(
                reg_idx[String(lk_d["region_a"])], reg_idx[String(lk_d["region_b"])],
                Float64(lk_d["atc_ab"]), Float64(lk_d["atc_ba"]), Float64(lk_d["loss_factor"]),
            ))
        end
    end

    raw_rsizes = payload["region_sizes"]
    rsizes_ga  = Tuple{Int,Int}[(Int(rs[1]), Int(rs[2])) for rs in raw_rsizes]

    pop_raw = payload["population"]
    P       = length(pop_raw)
    n_vars  = sum(n * T for (n, _) in rsizes_ga)
    pop_jl  = Matrix{Float64}(undef, P, n_vars)
    for p in 1:P, j in 1:n_vars
        pop_jl[p,j] = Float64(pop_raw[p][j])
    end

    ptdf = if haskey(payload, "ptdf_matrix") && length(payload["ptdf_matrix"]) > 0
        pm   = payload["ptdf_matrix"]
        L_ac = length(pm); Rp = L_ac > 0 ? length(pm[1]) : 0
        mat  = Matrix{Float64}(undef, L_ac, Rp)
        for l in 1:L_ac, r in 1:Rp; mat[l,r] = Float64(pm[l][r]); end
        ram  = Float64[Float64(v) for v in payload["ptdf_ram"]]
        PTDFData(L_ac, Rp, mat, ram)
    else
        PTDFData()
    end

    n_threads = Threads.nthreads()
    costs_bufs        = [zeros(Float64, S, R) for _ in 1:n_threads]
    marginal_p_bufs   = [zeros(Float64, S, R) for _ in 1:n_threads]
    remain_2d_bufs    = [zeros(Float64, S, R) for _ in 1:n_threads]
    export_avail_bufs = [zeros(Float64, S, R) for _ in 1:n_threads]
    net_export_bufs   = [zeros(Float64, max(ptdf.R, 1)) for _ in 1:n_threads]

    fitness = Vector{Float64}(undef, P)
    Threads.@threads for p in 1:P
        tid = Threads.threadid()
        fitness[p] = evaluate_individual(
            regions, region_data, fixed_commits, rsizes_ga, links,
            @view(pop_jl[p, :]), S, T, R, lambda_r, alpha, passes,
            costs_bufs[tid], marginal_p_bufs[tid],
            remain_2d_bufs[tid], export_avail_bufs[tid],
            ptdf, net_export_bufs[tid],
        )
    end
    return fitness
end

end  # module EngineBatch
