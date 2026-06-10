"""
succes/engine.jl
----------------
Julia simulation engine for SUCCES.
Owns the stochastic dispatch inner loop. Called in-process via juliacall.

Performance notes
-----------------
- All matrices arrive as Julia-native Float64 arrays from juliacall (zero-copy).
- Work arrays are pre-allocated once per dispatch_region call, reused per hour.
- Merit order is sorted once per window (fuel costs are scenario-1 proxy).
- Storage and hydro dispatch are fully vectorised over S.
- Inner loops are written column-major (Julia is column-major like Fortran).
"""

module SuccesEngine

using LinearAlgebra
using Statistics: mean

# JSON3 lazy-loaded only for the subprocess CLI path (main()).
const _JSON3_AVAILABLE = try
    Base.require(Base.PkgId(Base.UUID("0f8b85d8-7e73-4b5b-8d18-14e9ebf4c3a8"), "JSON3"))
    true
catch
    false
end

# ── Constants ──────────────────────────────────────────────────────────────────
const SCARCITY_PENALTY    =  3_000.0  # EUR/MWh (DA market cap — sets the reported MCP)
const CURTAILMENT_PENALTY =    200.0  # EUR/MWh surplus — matches engine_batch.jl
const MUST_RUN_PENALTY    =  2_000.0  # EUR/MWh must-run floor violation — hard physical constraint

# Negative price constants — must match engine_batch.jl exactly
# so GA cost function and final dispatch produce consistent price signals.
const NEG_PRICE_SLOPE     = 5.0     # EUR/MWh per GW surplus
const NEG_PRICE_FLOOR     = -500.0  # EUR/MWh hard floor (EEX cap)

# ── Fast matrix conversion from juliacall Any-typed payloads ──────────────────

"""Convert a list-of-lists (rows) to a Float64 matrix (S, T) efficiently."""
function to_matrix(data::AbstractVector, S::Int, T::Int)::Matrix{Float64}
    out = Matrix{Float64}(undef, S, T)
    @inbounds for s in 1:S
        row = data[s]
        for t in 1:T
            out[s,t] = Float64(row[t])
        end
    end
    return out
end

function to_vec(data::AbstractVector)::Vector{Float64}
    return Float64[Float64(x) for x in data]
end

function to_bool_vec(data::AbstractVector)::BitVector
    return BitVector(Bool(x) for x in data)
end

# ── Storage state struct (avoids repeated Dict lookups in hot loop) ────────────

struct StorageState
    soc::Vector{Float64}   # (S,) mutable
    cr::Float64
    dr::Float64
    ce::Float64
    de::Float64
    cap::Float64
    mc::Float64
end

# ── PTDF data struct ──────────────────────────────────────────────────────────
struct PTDFData
    L_ac::Int
    R::Int
    matrix::Matrix{Float64}
    ram::Vector{Float64}
end
PTDFData() = PTDFData(0, 0, Matrix{Float64}(undef,0,0), Vector{Float64}())

# ── Main dispatch ──────────────────────────────────────────────────────────────

function dispatch_region(
    S::Int, T::Int,
    demand::Matrix{Float64},           # (S, T) — column-major, inner dim = S
    fc_mat::Matrix{Float64},           # (S, N)
    min_caps::Vector{Float64},
    max_caps::Vector{Float64},
    ramp_rates::Vector{Float64},
    commitment::Matrix{Float64},       # (N, T)
    prev_dispatch::Matrix{Float64},    # (S, N)
    prev_on::Matrix{Float64},          # (S, N)
    storages::Vector{StorageState},
    hydro_names::Vector{String},
    hydro_soc::Dict{String,Vector{Float64}},
    hydro_inflow::Dict{String,Vector{Float64}},
    hydro_capacity::Dict{String,Float64},
    hydro_unit_idx::Dict{String,Int},
    startup_cost_vec::Vector{Float64},
    prev_on_startup::Vector{Float64},
    must_run_mask::BitVector,
    co2_intensity::Vector{Float64},    # tCO2/MWh_e per unit
    storage_log::Matrix{Float64},      # (n_storages, T) for reporting
    offer_prices::Vector{Float64},     # EUR/MWh offer price per unit (negative for must-run nuclear)
)
    N  = length(min_caps)
    Ns = length(storages)

    fuel_costs    = zeros(Float64, S)
    penalty_costs    = zeros(Float64, S)
    # Per-type penalty breakdown (summed over T, averaged over S at return)
    pen_scarcity     = zeros(Float64, S)   # unserved demand
    pen_curtailment  = zeros(Float64, S)   # surplus that can't be absorbed
    pen_must_run     = zeros(Float64, S)   # must-run units not at minimum
    pen_inertia      = zeros(Float64, S)   # inertia constraint violations
    pen_ramp         = zeros(Float64, S)   # ramp rate violations (post-dispatch)
    co2_emissions = zeros(Float64, S)
    dispatch_log  = zeros(Float64, N, T)   # scenario-1 dispatch
    dispatch_sum  = zeros(Float64, N, T)   # sum over S for mean
    net_position  = Matrix{Float64}(undef, S, T)
    prices        = zeros(Float64, S, T)

    # ── Startup costs (deterministic — same across all scenarios) ──────────────
    startup_total = 0.0
    prev_col = copy(prev_on_startup)
    @inbounds for t in 1:T, i in 1:N
        if commitment[i,t] > 0.5 && prev_col[i] < 0.5
            startup_total += startup_cost_vec[i]
        end
        prev_col[i] = commitment[i,t]
    end

    # ── Merit order: sort by s=1 fuel cost once per window ────────────────────
    order = sortperm(view(fc_mat, 1, :))   # cheapest first, fixed for window

    # ── Working arrays (allocated once) ───────────────────────────────────────
    pd_s    = copy(prev_dispatch)      # (S, N)
    po_s    = copy(prev_on)            # (S, N)
    lo      = zeros(Float64, S, N)
    hi      = zeros(Float64, S, N)
    disp_t  = zeros(Float64, S, N)
    remain  = Vector{Float64}(undef, S)

    # Pre-build array forms of hydro lookups for the hot loop (avoid Dict per hour)
    hydro_soc_arr      = Vector{Float64}[hydro_soc[n] for n in hydro_names]   # [soc_vec...]
    hydro_unit_idx_arr = zeros(Int, N)  # unit_index → soc_arr_index (0 = not hydro)
    hydro_unit_idx_rev = zeros(Int, length(hydro_names))  # soc_arr_index → unit_index
    for (ki, hname) in enumerate(hydro_names)
        if haskey(hydro_unit_idx, hname)
            ui = hydro_unit_idx[hname]
            if ui <= N
                hydro_unit_idx_arr[ui] = ki
                hydro_unit_idx_rev[ki] = ui
            end
        end
    end

    @inbounds for t in 1:T
        on_t = view(commitment, :, t)  # (N,)

        # Demand this hour
        @inbounds for s in 1:S; remain[s] = demand[s, t]; end

        # ── Hydro inflow ───────────────────────────────────────────────────────
        for hname in hydro_names
            inflow = hydro_inflow[hname][t]
            cap    = hydro_capacity[hname]
            soc    = hydro_soc[hname]
            @inbounds for s in 1:S
                soc[s] = min(soc[s] + inflow, cap)
            end
        end

        # ── Per-unit bounds ────────────────────────────────────────────────────
        fill!(lo, 0.0)
        fill!(hi, 0.0)
        for i in 1:N
            c_level = on_t[i]
            c_level > 0.01 || continue
            # hi scales with commitment fraction (GA controls dispatch level)
            # lo is the physical minimum stable generation — does NOT scale.
            # A unit either runs at ≥ min_cap or stays off. There is no
            # fractional minimum. Scaling lo by c_level was neutralising the
            # minimum stable generation constraint and allowing units to run
            # at 5-15% of capacity, which is physically infeasible and was
            # the root cause of persistent scarcity: units appeared committed
            # but contributed negligible capacity.
            lo_i = min_caps[i]           # physical minimum — constant
            hi_i = max_caps[i] * c_level # GA-controlled upper bound
            # Ensure hi >= lo; if c_level is very low (near START_THRESHOLD)
            # and min_cap > 0, then hi < lo. Resolve by clamping hi to lo:
            # the unit runs at exactly min_cap regardless of the low fraction.
            # The MIN_STABLE_PENALTY in engine_batch.jl discourages this zone.
            if hi_i < lo_i; hi_i = lo_i; end
            rr   = ramp_rates[i]   # physical ramp rate — does not scale with c_level

            if rr > 0.0
                @inbounds for s in 1:S
                    if po_s[s,i] > 0.5
                        # Unit was running: apply ramp limits around previous dispatch
                        lo[s,i] = max(lo_i, pd_s[s,i] - rr)
                        hi[s,i] = min(hi_i, pd_s[s,i] + rr)
                    else
                        # Fix 6: cold start / startup trajectory.
                        # Unit just started: it can only produce rr MW in first hour
                        # (ramping from zero). Lower bound is 0 (can't enforce min_cap
                        # in the very first hour of startup — ramp constraint takes priority).
                        # This creates a realistic startup ramp rather than an instant jump.
                        # The unit will reach min_cap after ceil(min_cap/ramp_rate) hours.
                        startup_dispatch = rr   # first-hour output = ramp_rate
                        if startup_dispatch >= lo_i
                            lo[s,i] = lo_i      # can already meet min_cap in one step
                            hi[s,i] = hi_i
                        else
                            lo[s,i] = 0.0       # during ramp-up: lower bound = 0
                            hi[s,i] = startup_dispatch  # upper bound = ramp_rate
                        end
                    end
                end
            else
                @inbounds for s in 1:S
                    lo[s,i] = lo_i
                    hi[s,i] = hi_i
                end
            end

            # Hydro: cap by reservoir (hydro_unit_idx_arr maps unit-index→soc-index)
            if i <= length(hydro_unit_idx_arr) && hydro_unit_idx_arr[i] > 0
                soc = hydro_soc_arr[hydro_unit_idx_arr[i]]
                @inbounds for s in 1:S
                    hi[s,i] = min(hi[s,i], max(0.0, soc[s]))
                    lo[s,i] = min(lo[s,i], hi[s,i])
                end
            end
        end

        # ── Must-run soft penalty (fixed units not committed) ──────────────────
        @inbounds for i in 1:N
            if must_run_mask[i] && on_t[i] < 0.5
                pen = min_caps[i] * MUST_RUN_PENALTY
                @inbounds for s in 1:S
                    penalty_costs[s] += pen
                    pen_must_run[s]  += pen
                end
            end
        end

        # ── Baseline: dispatch every unit to its lower bound ──────────────────
        @. disp_t = lo
        @inbounds for s in 1:S
            lb = 0.0; ub = 0.0
            for i in 1:N
                lb += lo[s,i]; ub += hi[s,i]
            end
            if lb > remain[s]
                penalty_costs[s] += (lb - remain[s]) * MUST_RUN_PENALTY
                pen_must_run[s]  += (lb - remain[s]) * MUST_RUN_PENALTY
            elseif remain[s] > ub
                penalty_costs[s] += (remain[s] - ub) * SCARCITY_PENALTY
                pen_scarcity[s]  += (remain[s] - ub) * SCARCITY_PENALTY
            end
            remain[s] -= lb
        end

        # ── Storage pre-dispatch + merit-order top-up + storage charge ───────────
        # Correct sequence: (1) storage discharge, (2) thermal top-up on reduced
        # remain, (3) storage charge from surplus. This lets storage discharge
        # at peak BEFORE the most expensive thermal units are called — the correct
        # price-arbitrage behaviour. Storage acts as a low-MC dispatchable asset.
        @inbounds for si in 1:Ns
            st = storages[si]
            soc = st.soc
            eff_mc = st.mc / max(st.de, 1e-9)   # effective discharge MC
            log_dc = 0.0; log_ch = 0.0
            for s in 1:S
                r = remain[s]
                if r > 1e-6
                    # Discharge if we have stored energy: replace thermal top-up
                    avail_dc = min(st.dr, soc[s] * st.de)
                    if avail_dc > 1e-3
                        dc = min(avail_dc, r)
                        soc[s]        -= dc / max(st.de, 1e-9)
                        remain[s]     -= dc
                        fuel_costs[s] += dc * st.mc
                        if s == 1; log_dc = dc; end
                    end
                end
            end
            storage_log[si, t] = log_dc   # discharge contribution (charge added below)
        end

        # ── Merit-order thermal top-up (on remain after storage discharge) ─────
        for i in order
            on_t[i] > 0.01 || continue
            @inbounds for s in 1:S
                remain[s] <= 1e-6 && continue
                room = hi[s,i] - lo[s,i]
                add  = min(room, remain[s])
                disp_t[s,i] += add
                remain[s]   -= add
            end
        end

        # ── Storage charge pass: absorb overnight surplus ──────────────────────
        @inbounds for si in 1:Ns
            st = storages[si]
            soc = st.soc
            log_ch = 0.0
            for s in 1:S
                r = remain[s]
                if r < -1e-6
                    # Surplus: charge storage
                    avail_ch = min(st.cr, (st.cap - soc[s]) / max(st.ce, 1e-9))
                    ch = min(avail_ch, -r)
                    soc[s]        += ch * st.ce
                    remain[s]     += ch
                    fuel_costs[s] += ch * st.mc
                    if s == 1; log_ch = ch; end
                end
            end
            storage_log[si, t] -= log_ch   # net = discharge - charge (+ = net discharge)
        end

        # ── Curtailment / unserved after storage ──────────────────────────────
        @inbounds for s in 1:S
            r = remain[s]
            if r < -1e-6
                penalty_costs[s]   += (-r) * CURTAILMENT_PENALTY
                pen_curtailment[s] += (-r) * CURTAILMENT_PENALTY
            elseif r > 1e-6
                penalty_costs[s] += r * SCARCITY_PENALTY
                pen_scarcity[s]  += r * SCARCITY_PENALTY
            end
            net_position[s,t] = -r
        end

        # ── Fuel cost + CO2 emissions ────────────────────────────────────────────
        @inbounds for s in 1:S
            fc = 0.0; co2 = 0.0
            for i in 1:N
                fc  += disp_t[s,i] * fc_mat[s,i]
                co2 += disp_t[s,i] * co2_intensity[i]
            end
            fuel_costs[s]    += fc
            co2_emissions[s] += co2
        end

        # ── MCP: market clearing price ────────────────────────────────────────
        # The price is set by the LAST (most expensive) unit dispatched above
        # its minimum — the marginal unit in a uniform-price DA market.
        #
        # Algorithm: walk merit order MOST-EXPENSIVE-FIRST (Iterators.reverse).
        # The first unit found with dispatch > minimum is the marginal unit.
        #
        # This is the critical correction: walking cheapest-first and breaking
        # on the first match gives the CHEAPEST dispatched unit (nuclear at
        # -50 EUR/MWh) even when 20 GW of CCGT at 90 EUR/MWh is also running.
        # Walking reverse correctly identifies the most expensive unit needed.
        #
        # Surplus (remain < 0): the cheapest running unit is being curtailed.
        # Walk forward (cheapest-first) to find it — its offer_price is the MCP.
        # This correctly produces 0 EUR/MWh when wind is curtailed, and
        # -30 to -50 EUR/MWh when nuclear alone creates surplus.
        @inbounds for s in 1:S
            mcp            = 0.0
            found_marginal = false

            # Normal: walk expensive → cheap, find last unit that was needed
            for i in Iterators.reverse(order)
                on_t[i] > 0.01 || continue
                hi[s,i] > 1e-3 || continue
                if disp_t[s,i] > lo[s,i] + 1e-3
                    mcp = offer_prices[i]
                    found_marginal = true
                    break
                end
            end

            r2 = remain[s]
            if r2 > 2.0
                # Shortfall exceeds 2 MW dead-band → scarcity price.
                # Sub-2 MW gaps are numerical noise (rounding in dispatch fractions,
                # min-cap clamping, floating-point accumulation across many units).
                # Triggering 3000 EUR/MWh for a 0.3 MW residual would broadcast
                # a false scarcity signal; instead we let found_marginal set the price.
                mcp = SCARCITY_PENALTY
            elseif r2 < -1e-3
                # Surplus: find cheapest running unit (it is being curtailed)
                for i in order
                    on_t[i] > 0.01 || continue
                    hi[s,i] > 1e-3 || continue
                    mcp = offer_prices[i]
                    break
                end
                mcp = max(mcp, NEG_PRICE_FLOOR)
            elseif !found_marginal
                # All units at minimum — use cheapest running unit's offer
                for i in order
                    on_t[i] > 0.01 || continue
                    mcp = offer_prices[i]
                    break
                end
            end

            prices[s, t] = mcp
        end

        # ── Hydro water consumption (array lookup, no dict) ───────────────────
        for (ki, soc) in enumerate(hydro_soc_arr)
            hidx = hydro_unit_idx_rev[ki]
            @inbounds for s in 1:S
                soc[s] = max(0.0, soc[s] - disp_t[s,hidx])
            end
        end

        # ── Dispatch log (scenario 1) and scenario-mean ──────────────────────
        @inbounds for i in 1:N
            dispatch_log[i,t] = disp_t[1,i]
            s_sum = 0.0
            for s in 1:S; s_sum += disp_t[s,i]; end
            dispatch_sum[i,t] = s_sum
        end

        # ── Advance state ──────────────────────────────────────────────────────
        @inbounds for s in 1:S, i in 1:N
            pd_s[s,i] = disp_t[s,i]
            po_s[s,i] = on_t[i]
        end
    end  # hour loop

    return (
        costs         = fuel_costs .+ penalty_costs .+ startup_total,
        fuel_costs    = fuel_costs,
        co2_emissions = co2_emissions,
        startup_costs = fill(startup_total, S),
        penalty_costs = penalty_costs,
        # Per-type penalty breakdown (mean over S, summed over T)
        pen_scarcity     = pen_scarcity,
        pen_curtailment  = pen_curtailment,
        pen_must_run     = pen_must_run,
        pen_inertia      = pen_inertia,
        pen_ramp         = pen_ramp,
        dispatch_log  = dispatch_log,
        dispatch_mean = dispatch_sum ./ S,
        storage_log   = storage_log,
        net_position  = net_position,
        prices        = prices,
    )
end


# ── Price-driven cross-border market coupling (in-place) ──────────────────────
# Mirrors the 4-phase algorithm in engine_batch.jl::evaluate_individual.
# Called after all regions have completed their lower-bound baseline dispatch.
# net_pos[s,t,r] on entry  : surplus(>0) / deficit(<0) after local lb dispatch
# net_pos[s,t,r] on exit   : residual after trade (used for post-flow penalties)
# marginal_p[s,t,r]         : cheapest offer price per region per hour per scenario
#
# Trade logic (per link per scenario per hour):
#   if price_B * (1+loss) < price_A → B exports to A up to min(B headroom, A deficit, ATC)
#   if price_A * (1+loss) < price_B → A exports to B
# Iterates flow_passes times. export_avail prevents double-commitment across links.

function cross_border_flows!(
    net_pos::Array{Float64,3},            # (S, T, R) — modified in place
    marginal_p::Array{Float64,3},         # (S, T, R) — cheapest offer price
    export_avail::Array{Float64,3},       # (S, T, R) — available headroom, modified
    la::Vector{Int}, lb::Vector{Int},
    atc_ab::Vector{Float64}, atc_ba::Vector{Float64},
    lf::Vector{Float64},
    S::Int, T::Int, n_passes::Int,
    ptdf::PTDFData,                       # PTDF constraints (empty = ATC only)
)
    n_links   = length(la)
    flows_log = zeros(Float64, S, T, n_links)

    net_export_buf = zeros(Float64, max(ptdf.R, 1))

    @inbounds for t in 1:T
        for _ in 1:n_passes
            for l in 1:n_links
                ia = la[l]; ib = lb[l]; loss = lf[l]
                cap_ab = atc_ab[l]; cap_ba = atc_ba[l]
                for s in 1:S
                    pa = marginal_p[s,t,ia]
                    pb = marginal_p[s,t,ib]
                    if pb*(1.0+loss) < pa
                        deficit_a = max(-net_pos[s,t,ia], 0.0)
                        flow = clamp(min(export_avail[s,t,ib], deficit_a), 0.0, cap_ba)
                        net_pos[s,t,ia]       += flow * (1.0 - loss)
                        net_pos[s,t,ib]       -= flow
                        export_avail[s,t,ib]  -= flow
                        flows_log[s,t,l]      += flow
                    elseif pa*(1.0+loss) < pb
                        deficit_b = max(-net_pos[s,t,ib], 0.0)
                        flow = clamp(min(export_avail[s,t,ia], deficit_b), 0.0, cap_ab)
                        net_pos[s,t,ib]       += flow * (1.0 - loss)
                        net_pos[s,t,ia]       -= flow
                        export_avail[s,t,ia]  -= flow
                        flows_log[s,t,l]      -= flow
                    end
                end
            end

            # ── PTDF line-loading cap (AC lines only) ────────────────────────
            if ptdf.L_ac > 0
                R_ptdf = ptdf.R
                for s in 1:S
                    # Net export[r] = -net_pos (positive = exporting)
                    for r in 1:R_ptdf
                        net_export_buf[r] = -net_pos[s, t, r]
                    end
                    # Find most binding AC line constraint
                    scale = 1.0
                    for l_ac in 1:ptdf.L_ac
                        loading = 0.0
                        for r in 1:R_ptdf
                            loading += ptdf.matrix[l_ac, r] * net_export_buf[r]
                        end
                        abs_load = abs(loading)
                        if abs_load > ptdf.ram[l_ac] + 1e-3
                            s_l = ptdf.ram[l_ac] / abs_load
                            if s_l < scale; scale = s_l; end
                        end
                    end
                    # Scale all net positions toward zero by binding factor
                    if scale < 1.0 - 1e-6
                        for r in 1:R_ptdf
                            net_pos[s, t, r] *= scale
                            export_avail[s, t, r] /= max(scale, 1e-9)
                        end
                    end
                end
            end  # PTDF check
        end  # passes
    end  # time
    return flows_log
end


# ── Top-level entry (called from juliacall) ────────────────────────────────────

function run_simulation(payload::Dict)
    # juliacall passes Python dicts/lists; coerce everything up front
    S       = Int(payload["S"])
    T       = Int(payload["T"])
    regions = String[String(r) for r in payload["regions"]]
    passes  = Int(get(payload, "flow_passes", 2))
    R       = length(regions)

    regional_results = Dict{String,Any}()
    storage_socs_out = Dict{String,Dict{String,Vector{Float64}}}()
    hydro_socs_out   = Dict{String,Dict{String,Vector{Float64}}}()

    # ── Market coupling pre-pass ──────────────────────────────────────────────
    # Before dispatching each region, estimate cross-border flows using PTDF.
    # This adjusts each region's effective demand so that cheap regions serve
    # expensive neighbours, subject to ATC and PTDF line limits.
    #
    # Algorithm:
    #   1. For each region, estimate the marginal price at demand coverage:
    #      = fc_mat[1, last_needed_unit] (cheapest scenario-1 merit order).
    #   2. Run the ATC/PTDF arbitrage to find optimal import/export per region.
    #   3. Store the resulting demand adjustment per (s, t, region).
    #
    # This keeps PTDF as a pure DC load-flow constraint — no new GA variables.
    # The GA dispatches against coupled (adjusted) demand, giving it the
    # correct incentive to dispatch more in cheap regions.

    demand_adj = Dict{String, Matrix{Float64}}()   # (S, T) demand adjustment per region
    for r in regions
        demand_adj[r] = zeros(Float64, S, T)
    end

    # Bug-fix: pre-pass flows stored here so flows_out can report them.
    # flows_3d (post-dispatch net_pos) is always ≈0 because dispatch already
    # consumed the demand adjustment; the true cross-border flows are the
    # ones computed in the pre-pass below.
    # Initialised to nothing; allocated once L_pre is known inside the if-block.
    flows_pre_3d::Union{Array{Float64,3}, Nothing} = nothing
    L_pre_saved::Int = 0

    if haskey(payload, "links") && length(payload["links"]) > 0 && R > 1
        ridx_pre  = Dict{String,Int}(r => i for (i,r) in enumerate(regions))
        ldata_pre = payload["links"]
        L_pre     = length(ldata_pre)
        la_pre    = Int[ridx_pre[String(l["region_a"])] for l in ldata_pre]
        lb_pre    = Int[ridx_pre[String(l["region_b"])] for l in ldata_pre]
        atcab_pre = Float64[Float64(l["atc_ab"])      for l in ldata_pre]
        atcba_pre = Float64[Float64(l["atc_ba"])      for l in ldata_pre]
        # Allocate tensor to capture per-link flows from the pre-pass.
        # Indexed [s, t, lk_i]: net flow on link lk_i in scenario s at hour t
        # (positive = ia→ib direction).  Used below to build flows_out so that
        # reporter.py sees non-zero mean_flows instead of the post-dispatch ≈0 values.
        flows_pre_3d  = zeros(Float64, S, T, L_pre)
        L_pre_saved   = L_pre
        lfv_pre   = Float64[Float64(l["loss_factor"]) for l in ldata_pre]

        ptdf_pre = if haskey(payload, "ptdf_matrix") && length(payload["ptdf_matrix"]) > 0
            pm_pre = payload["ptdf_matrix"]
            L_ac   = length(pm_pre); Rp = L_ac > 0 ? length(pm_pre[1]) : 0
            mat    = Matrix{Float64}(undef, L_ac, Rp)
            for l in 1:L_ac, r in 1:Rp; mat[l,r] = Float64(pm_pre[l][r]); end
            ram    = Float64[Float64(v) for v in payload["ptdf_ram"]]
            PTDFData(L_ac, Rp, mat, ram)
        else
            PTDFData()
        end

        # Build merit-order estimated prices per region (for price signal)
        # and available capacity per region (for export headroom)
        est_price    = zeros(Float64, S, T, R)
        avail_export = zeros(Float64, S, T, R)

        for (ri, r) in enumerate(regions)
            rd_pre  = payload["region_data"][r]
            N_pre   = length(rd_pre["min_caps"])
            fc_pre  = to_matrix(rd_pre["fuel_cost_matrix"], S, N_pre)
            cap_pre = to_vec(rd_pre["max_caps"])
            com_pre = to_matrix(rd_pre["commitment"], N_pre, T)
            dem_pre = to_matrix(rd_pre["demand"], S, T)

            # Merit order: sorted by fc_mat[1,:] (use scenario-1 for ordering)
            order_pre = sortperm(fc_pre[1, :])

            @inbounds for t in 1:T
                for s in 1:S
                    d = dem_pre[s, t]
                    cumcap = 0.0
                    last_price = fc_pre[s, order_pre[end]]
                    total_cap = 0.0
                    for idx in order_pre
                        c = com_pre[idx, t]
                        c > 0.01 || continue
                        unit_cap  = cap_pre[idx] * c
                        total_cap += unit_cap
                        cumcap    += unit_cap
                        if cumcap >= d
                            last_price = fc_pre[s, idx]
                            break
                        end
                    end
                    est_price[s, t, ri]    = last_price
                    # Cap export at 40% of demand (raised from 25%).
                    # The 25% cap was blocking FR→DE and DE→NL flows that the ATC
                    # physically allows. At 40% the pre-pass correctly signals that
                    # cheap FR nuclear surplus should suppress DE CCGT dispatch.
                    avail_export[s, t, ri] = min(max(0.0, total_cap - d), d * 0.40)
                end
            end
        end

        # Run ATC per hour to compute coupled trade
        net_export_pre = zeros(Float64, max(ptdf_pre.R, 1))
        @inbounds for t in 1:T
            # net_pos_coupled[s,r]: positive = surplus available to export
            # Initialise at zero — regions are balanced before coupling
            net_pos_pre    = zeros(Float64, S, R)
            marg_p_pre     = zeros(Float64, S, R)
            exp_avail_pre  = zeros(Float64, S, R)
            for ri in 1:R
                for s in 1:S
                    marg_p_pre[s,ri]    = est_price[s, t, ri]
                    exp_avail_pre[s,ri] = avail_export[s, t, ri]
                end
            end

            # Run ATC: moves surplus from cheap to deficit in expensive regions
            for _ in 1:passes
                for lk_i in 1:L_pre
                    ia = la_pre[lk_i]; ib = lb_pre[lk_i]
                    loss = lfv_pre[lk_i]
                    for s in 1:S
                        pa = marg_p_pre[s,ia]; pb = marg_p_pre[s,ib]
                        if pb*(1.0+loss) < pa
                            # ib is cheaper: export ib→ia (negative ia→ib convention)
                            flow = min(exp_avail_pre[s,ib], atcba_pre[lk_i])
                            flow = max(0.0, flow)
                            net_pos_pre[s,ia]    += flow*(1.0-loss)
                            net_pos_pre[s,ib]    -= flow
                            exp_avail_pre[s,ib]  -= flow
                            flows_pre_3d[s, t, lk_i] -= flow   # ib→ia = negative ia→ib
                        elseif pa*(1.0+loss) < pb
                            # ia is cheaper: export ia→ib (positive ia→ib convention)
                            flow = min(exp_avail_pre[s,ia], atcab_pre[lk_i])
                            flow = max(0.0, flow)
                            net_pos_pre[s,ib]    += flow*(1.0-loss)
                            net_pos_pre[s,ia]    -= flow
                            exp_avail_pre[s,ia]  -= flow
                            flows_pre_3d[s, t, lk_i] += flow   # ia→ib = positive
                        end
                    end
                end

                # PTDF constraint check and scaling
                if ptdf_pre.L_ac > 0
                    Rp = ptdf_pre.R
                    for s in 1:S
                        for ri in 1:Rp; net_export_pre[ri] = -net_pos_pre[s,ri]; end
                        scale = 1.0
                        for l_ac in 1:ptdf_pre.L_ac
                            loading = 0.0
                            for ri in 1:Rp; loading += ptdf_pre.matrix[l_ac,ri]*net_export_pre[ri]; end
                            abs_load = abs(loading)
                            if abs_load > ptdf_pre.ram[l_ac] + 1e-3
                                s_l = ptdf_pre.ram[l_ac]/abs_load
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

            # net_pos_pre[s,r] = net import into r from market coupling
            # Apply as demand adjustment: demand_adj[r][s,t] = net_pos_pre[s,ri]
            # Positive net_pos = region received imports → effective demand decreases
            for (ri, r) in enumerate(regions)
                for s in 1:S
                    demand_adj[r][s, t] = net_pos_pre[s, ri]
                end
            end
        end
    end

    for r in regions
        rd  = payload["region_data"][r]
        N   = length(rd["min_caps"])

        # ── Fast matrix conversions ────────────────────────────────────────────
        # Apply market coupling demand adjustment: subtract net imports from demand
        demand_raw = to_matrix(rd["demand"],           S, T)
        demand     = demand_raw .- demand_adj[r]       # effective demand after coupling
        fc_mat   = to_matrix(rd["fuel_cost_matrix"], S, N)
        commit   = to_matrix(rd["commitment"],       N, T)  # (N,T)
        prev_d   = to_matrix(rd["prev_dispatch"],    S, N)
        prev_o   = to_matrix(rd["prev_on"],          S, N)

        min_caps  = to_vec(rd["min_caps"])
        max_caps  = to_vec(rd["max_caps"])
        ramp_r    = to_vec(rd["ramp_rates"])
        sc_vec    = to_vec(rd["startup_cost_vec"])
        prev_on_s = to_vec(rd["prev_on_startup"])
        must_run  = to_bool_vec(rd["must_run_mask"])

        # ── Storage structs ────────────────────────────────────────────────────
        st_names = String[String(n) for n in rd["storage_names"]]
        Ns = length(st_names)
        storages = StorageState[]
        for sname in st_names
            soc_raw = rd["storage_soc"][sname]
            push!(storages, StorageState(
                Float64[Float64(x) for x in soc_raw],
                Float64(rd["storage_charge_rate"][sname]),
                Float64(rd["storage_discharge_rate"][sname]),
                Float64(rd["storage_charge_eff"][sname]),
                Float64(rd["storage_discharge_eff"][sname]),
                Float64(rd["storage_capacity"][sname]),
                Float64(rd["storage_marginal"][sname]),
            ))
        end
        st_log = zeros(Float64, Ns, T)

        # ── Hydro ──────────────────────────────────────────────────────────────
        hy_names  = String[String(n) for n in rd["hydro_names"]]
        hy_soc    = Dict{String,Vector{Float64}}(
            String(k) => Float64[Float64(x) for x in v]
            for (k,v) in rd["hydro_soc"]
        )
        hy_inflow = Dict{String,Vector{Float64}}(
            String(k) => Float64[Float64(x) for x in v]
            for (k,v) in rd["hydro_inflow"]
        )
        hy_cap  = Dict{String,Float64}(String(k) => Float64(v) for (k,v) in rd["hydro_capacity"])
        hy_uidx = Dict{String,Int}(String(k) => Int(v) for (k,v) in rd["hydro_unit_idx"])

        co2_int   = haskey(rd, "co2_intensity") ? Float64[Float64(v) for v in rd["co2_intensity"]] : zeros(Float64, N)
        # Change 3: offer prices — negative for must-run nuclear, zero for thermal/hydro.
        # Loaded from payload key "offer_prices"; defaults to fc_mat row 1 if absent
        # (which preserves pre-change behaviour for units without an offer price set).
        offer_p   = haskey(rd, "offer_prices") ?
                        Float64[Float64(v) for v in rd["offer_prices"]] :
                        Float64[fc_mat[1,i] for i in 1:N]

        # ── Dispatch ───────────────────────────────────────────────────────────
        res = dispatch_region(
            S, T, demand, fc_mat,
            min_caps, max_caps, ramp_r,
            commit, prev_d, prev_o,
            storages,
            hy_names, hy_soc, hy_inflow, hy_cap, hy_uidx,
            sc_vec, prev_on_s, must_run,
            co2_int,
            st_log,
            offer_p,
        )

        # ── Pack results: return actual arrays, not wrapped Julia objects ──────
        regional_results[r] = Dict{String,Any}(
            "costs"          => Vector{Float64}(res.costs),
            "fuel_costs"     => Vector{Float64}(res.fuel_costs),
            "co2_emissions"  => Vector{Float64}(res.co2_emissions),
            "startup_costs"  => Vector{Float64}(res.startup_costs),
            "penalty_costs"  => Vector{Float64}(res.penalty_costs),
            # Per-type penalty breakdown (S-length vectors; mean over S gives EUR/window)
            "pen_scarcity"    => Vector{Float64}(res.pen_scarcity),
            "pen_curtailment" => Vector{Float64}(res.pen_curtailment),
            "pen_must_run"    => Vector{Float64}(res.pen_must_run),
            "pen_inertia"     => Vector{Float64}(res.pen_inertia),
            "pen_ramp"        => Vector{Float64}(res.pen_ramp),
            "dispatch_log"   => Matrix{Float64}(res.dispatch_log),    # (N, T) scenario-1
            "dispatch_mean"  => Matrix{Float64}(res.dispatch_mean),   # (N, T) mean over S
            "storage_log"    => Matrix{Float64}(res.storage_log),     # (Ns, T)
            "storage_names"  => st_names,
            "net_position"   => Matrix{Float64}(res.net_position),    # (S, T)
            "prices"         => Matrix{Float64}(res.prices),          # (S, T)
            "mean_prices"    => Vector{Float64}(vec(mean(res.prices, dims=1))),  # (T,)
        )
        storage_socs_out[r] = Dict{String,Vector{Float64}}(
            st_names[i] => copy(storages[i].soc) for i in 1:Ns
        )
        hydro_socs_out[r] = Dict{String,Vector{Float64}}(
            k => copy(v) for (k,v) in hy_soc
        )
    end

    # ── Cross-border flows ────────────────────────────────────────────────────
    total_costs = zeros(Float64, S)
    flows_out   = nothing

    if haskey(payload, "links") && length(payload["links"]) > 0 && R > 1
        ridx = Dict{String,Int}(r => i for (i,r) in enumerate(regions))
        ldata = payload["links"]
        L = length(ldata)
        la    = Int[ridx[String(l["region_a"])] for l in ldata]
        lb    = Int[ridx[String(l["region_b"])] for l in ldata]
        atcab = Float64[Float64(l["atc_ab"])     for l in ldata]
        atcba = Float64[Float64(l["atc_ba"])     for l in ldata]
        lfv   = Float64[Float64(l["loss_factor"]) for l in ldata]

        net_pos_3d = zeros(Float64, S, T, R)
        for (ri, r) in enumerate(regions)
            np = regional_results[r]["net_position"]   # (S, T)
            net_pos_3d[:,:,ri] .= np
        end

        # Build marginal_p (S, T, R) and export_avail (S, T, R) from dispatch results
        # marginal_p[s,t,r] = price recorded per region in dispatch_region
        # export_avail[s,t,r] = sum of (hi-lo) headroom — approximated from net_pos surplus
        marginal_p_3d  = zeros(Float64, S, T, R)
        export_avail_3d = zeros(Float64, S, T, R)
        for (ri, r) in enumerate(regions)
            pr = regional_results[r]["prices"]           # (S, T)
            marginal_p_3d[:,:,ri] .= pr
            np = regional_results[r]["net_position"]     # (S, T): surplus > 0
            @inbounds for t in 1:T, s in 1:S
                # Available export = local surplus (what we haven't consumed)
                export_avail_3d[s,t,ri] = max(net_pos_3d[s,t,ri], 0.0)
            end
        end

        # Build PTDFData from payload if present
        ptdf_eng = if haskey(payload, "ptdf_matrix") && length(payload["ptdf_matrix"]) > 0
            pm   = payload["ptdf_matrix"]
            L_ac = length(pm)
            Rp   = L_ac > 0 ? length(pm[1]) : 0
            mat  = Matrix{Float64}(undef, L_ac, Rp)
            for l in 1:L_ac, r in 1:Rp; mat[l,r] = Float64(pm[l][r]); end
            ram  = Float64[Float64(v) for v in payload["ptdf_ram"]]
            PTDFData(L_ac, Rp, mat, ram)
        else
            PTDFData()
        end

        flows_3d = cross_border_flows!(
            net_pos_3d, marginal_p_3d, export_avail_3d,
            la, lb, atcab, atcba, lfv, S, T, passes, ptdf_eng
        )

        # Post-flow penalties
        post_pen = zeros(Float64, S)
        @inbounds for ri in 1:R, s in 1:S, t in 1:T
            np = net_pos_3d[s,t,ri]
            if np < 0.0
                post_pen[s] += (-np) * SCARCITY_PENALTY
            elseif np > 0.0
                post_pen[s] += np * CURTAILMENT_PENALTY
            end
        end

        for r in regions
            rr = regional_results[r]
            total_costs .+= (rr["fuel_costs"] .+ rr["startup_costs"])
        end
        total_costs .+= post_pen

        # Post-coupling price update.
        # After cross-border trade, the MCP from dispatch_region is already correct
        # for balanced hours (last dispatched unit's offer price).
        # We only need to handle extreme residuals after trade:
        #   net_pos > 1 MW: surplus remains → keep dispatch_region MCP (already
        #                   reflects the cheapest unit's offer price = 0 for wind,
        #                   negative for nuclear)
        #   net_pos < -1 MW: deficit after trade → scarcity price
        # No slope adjustment needed — the offer_price mechanism handles negative
        # prices correctly without a formula.
        for (ri, r) in enumerate(regions)
            pr_old = regional_results[r]["prices"]   # (S, T)
            pr_new = copy(pr_old)
            @inbounds for s in 1:S, t in 1:T
                np = net_pos_3d[s,t,ri]
                if np < -1.0
                    pr_new[s,t] = SCARCITY_PENALTY
                elseif np > 1.0
                    # Surplus after trade: price is already set by dispatch_region
                    # to the cheapest running unit's offer price. Enforce floor only.
                    pr_new[s,t] = max(pr_old[s,t], NEG_PRICE_FLOOR)
                end
            end
            regional_results[r]["prices"]      = pr_new
            regional_results[r]["mean_prices"] = vec(mean(pr_new, dims=1))
        end

        flows_mean_TL   = zeros(Float64, T, L)
        net_pos_mean_TR = zeros(Float64, T, R)
        @inbounds for t in 1:T
            for l in 1:L
                # Bug-fix: use pre-pass flows (flows_pre_3d) which are the actual
                # cross-border trades.  flows_3d from cross_border_flows! is computed
                # from post-dispatch net_pos which is ≈0 because dispatch already
                # consumed demand_adj — so flows_3d was always near-zero in the output.
                # flows_pre_3d uses the same link ordering (L_pre == L when this
                # block executes because both draw from payload["links"]).
                s_sum = 0.0
                if flows_pre_3d !== nothing && l <= L_pre_saved
                    for s in 1:S; s_sum += flows_pre_3d[s,t,l]; end
                else
                    for s in 1:S; s_sum += flows_3d[s,t,l]; end
                end
                flows_mean_TL[t,l] = s_sum / S
            end
            for ri in 1:R
                s_sum = 0.0
                for s in 1:S; s_sum += net_pos_3d[s,t,ri]; end
                net_pos_mean_TR[t,ri] = s_sum / S
            end
        end
        flows_out = Dict{String,Any}(
            "flows_mean"   => flows_mean_TL,
            "net_pos_mean" => net_pos_mean_TR,
        )
    else
        for r in regions
            total_costs .+= regional_results[r]["costs"]
        end
    end

    # Attach end-of-window SOC/reservoir to regional results for carryover
    for r in regions
        regional_results[r]["storage_soc"] = storage_socs_out[r]
        regional_results[r]["hydro_soc"]   = hydro_socs_out[r]
    end

    return Dict{String,Any}(
        "regional"    => regional_results,
        "total_costs" => total_costs,
        "flows"       => flows_out,
        "regions"     => regions,
    )
end


# ── CLI entry (subprocess mode, requires JSON3) ───────────────────────────────

function main()
    input_path  = ARGS[1]
    output_path = ARGS[2]
    _JSON3_AVAILABLE || error(
        "JSON3 not available. Install: julia -e 'import Pkg; Pkg.add(\"JSON3\")'\n" *
        "Or use juliacall mode (recommended) which needs no extra packages."
    )
    JSON3 = Base.require(Base.PkgId(
        Base.UUID("0f8b85d8-7e73-4b5b-8d18-14e9ebf4c3a8"), "JSON3"
    ))
    payload = open(io -> JSON3.read(io, Dict{String,Any}), input_path)
    result  = run_simulation(payload)
    open(io -> JSON3.write(io, result), output_path, "w")
end

end  # module SuccesEngine

if abspath(PROGRAM_FILE) == @__FILE__
    SuccesEngine.main()
end
