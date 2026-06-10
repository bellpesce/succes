"""
succes/engine_ga.jl
-------------------
Continuous GA for direct dispatch optimisation.

CHROMOSOME: x[i,t] ∈ [0,1] = dispatch fraction for unit i at hour t.
  Actual MW = x[i,t] × max_cap[i]
  x = 0.0 → unit off (no fuel cost, no startup unless was on)
  x = 0.7 → unit at 70% of its max capacity
  x = 1.0 → unit at full capacity

OPERATORS:
  Crossover: unit-block blend (BLX) — arithmetic blend per unit's T-hour block
  Mutation:
    1. Overnight block-zero: set overnight hours of a cyclable unit to 0.0
    2. Gaussian perturbation: N(0, sigma) noise on each gene
    3. Random reset: U(0,1) for rare exploration
  Selection: k-way tournament

WHY THIS WORKS:
  - x=1.0 for an unneeded unit costs fuel directly → GA learns to set x=0
  - x=0.0 overnight avoids forced fuel cost → cycling is economically rational
  - Intermediate values (0.3, 0.7) represent ramp-up/ramp-down profiles
  - The GA directly optimises the dispatch schedule, not just availability
"""

module EngineGA

using Statistics: mean
using Random: MersenneTwister, rand!, shuffle!

const _ENGINE_BATCH_PATH = joinpath(@__DIR__, "engine_batch.jl")
include(_ENGINE_BATCH_PATH)

# ── Crossover: unit-block blend (BLX) ─────────────────────────────────────────
"""
For each unit's T-hour block: randomly swap (1/3), arithmetic blend (1/3), or keep (1/3).
Blend produces intermediate values naturally — key for continuous optimisation.
"""
function crossover_block!(
    c1::Vector{Float64}, c2::Vector{Float64},
    p1::AbstractVector{Float64}, p2::AbstractVector{Float64},
    rsizes::Vector{Tuple{Int,Int}},
    pc::Float64, rng::MersenneTwister,
)
    copyto!(c1, p1); copyto!(c2, p2)
    rand(rng) > pc && return
    offset = 1
    for (n_free, T) in rsizes
        for ui in 0:(n_free-1)
            r = rand(rng)
            s = offset + ui * T
            if r < 0.33
                # Full swap
                @inbounds for j in 0:(T-1)
                    c1[s+j], c2[s+j] = c2[s+j], c1[s+j]
                end
            elseif r < 0.66
                # Arithmetic blend: alpha ~ U(0,1)
                alpha = rand(rng)
                @inbounds for j in 0:(T-1)
                    v1 = p1[s+j]; v2 = p2[s+j]
                    c1[s+j] = clamp(alpha*v1 + (1.0-alpha)*v2, 0.0, 1.0)
                    c2[s+j] = clamp(alpha*v2 + (1.0-alpha)*v1, 0.0, 1.0)
                end
            end
            # else: keep parents unchanged for this unit block
        end
        offset += n_free * T
    end
end

# ── Mutation: three continuous operators ───────────────────────────────────────
"""
1. Overnight block-zero: sets overnight hours of one cyclable unit to 0.0.
   This is the primary cycling driver — unit produces nothing overnight.
2. Gaussian perturbation at rate pm: x += N(0, sigma), clamped to [0,1].
3. Random reset at rate pm/4: x ~ U(0,1) for exploration.
"""
function mutate!(
    ind::AbstractVector{Float64},
    pm::Float64,
    rng::MersenneTwister,
    cyclable_offsets::Vector{Int},
    T::Int,
    p_block::Float64 = 0.20,
    overnight_end::Int = 8,
    sigma::Float64 = 0.15,
)
    n = length(ind)

    # Operator 1: overnight block-zero (cycling)
    if !isempty(cyclable_offsets) && rand(rng) < p_block
        u_start   = cyclable_offsets[rand(rng, 1:length(cyclable_offsets))]
        block_end = min(u_start + overnight_end - 1, u_start + T - 1)
        @inbounds for j in (u_start+1):block_end
            ind[j] = 0.0
        end
    end

    # Operator 2: Gaussian perturbation
    @inbounds for i in 1:n
        if rand(rng) < pm
            ind[i] = clamp(ind[i] + sigma * randn(rng), 0.0, 1.0)
        end
    end

    # Operator 3: random reset
    @inbounds for i in 1:n
        if rand(rng) < pm * 0.25
            ind[i] = rand(rng)
        end
    end
end

# ── Tournament selection ───────────────────────────────────────────────────────
function tournament_idx(fitness::AbstractVector{Float64}, k::Int, rng::MersenneTwister)::Int
    n = length(fitness)
    best_idx = rand(rng, 1:n)
    best_fit = fitness[best_idx]
    for _ in 2:k
        idx = rand(rng, 1:n)
        if fitness[idx] < best_fit
            best_fit = fitness[idx]; best_idx = idx
        end
    end
    return best_idx
end

# ── eval_population! wrapper ───────────────────────────────────────────────────
function eval_population!(
    fitness::Vector{Float64},
    pop::Matrix{Float64},
    P::Int,
    regions::Vector{String},
    region_data::Dict{String, EngineBatch.RegionData},
    fixed_commits::Dict{String, Matrix{Float64}},
    rsizes_ga::Vector{Tuple{Int,Int}},
    links::Vector{EngineBatch.LinkData},
    S::Int, T::Int, R::Int,
    lambda_r::Float64, alpha::Float64, passes::Int,
    costs_bufs, marginal_p_bufs, remain_2d_bufs, export_avail_bufs,
    ptdf::EngineBatch.PTDFData,
    net_export_bufs::Vector{Vector{Float64}},
)
    Threads.@threads for p in 1:P
        tid = Threads.threadid()
        row = @view pop[p, :]
        fitness[p] = EngineBatch.evaluate_individual(
            regions, region_data, fixed_commits, rsizes_ga, links,
            row, S, T, R, lambda_r, alpha, passes,
            costs_bufs[tid], marginal_p_bufs[tid],
            remain_2d_bufs[tid], export_avail_bufs[tid],
            ptdf, net_export_bufs[tid],
        )
    end
end

# ── Population initialisation ──────────────────────────────────────────────────
"""
Initialise population with a mix of:
  - Seeds from build_seeds (heuristic good solutions)
  - Night-cycling (all-on except overnight=0 for 1-3 units)
  - Mid-range (U(0.3, 0.7) per gene)
  - All-on with small noise
  - Cold random U(0,1)
"""
function init_population!(
    pop::Matrix{Float64},
    P::Int, n_vars::Int, T::Int,
    seeds::Vector{Vector{Float64}},
    rsizes_ga::Vector{Tuple{Int,Int}},
    rng::MersenneTwister,
)
    all_on = ones(Float64, n_vars)

    # Insert seeds
    n_seeds = min(length(seeds), P)
    for i in 1:n_seeds
        pop[i, :] = seeds[i]
    end

    # Build unit start offsets
    unit_starts = Int[]
    offset = 1
    for (n_free, T_r) in rsizes_ga
        for ui in 0:(n_free-1)
            push!(unit_starts, offset + ui * T_r)
        end
        offset += n_free * T_r
    end

    # Structured: 45% of population
    n_structured = max(0, round(Int, P * 0.45) - n_seeds)
    n_cycling    = round(Int, n_structured / 3)
    n_midrange   = round(Int, n_structured / 3)

    # Tier 1: night-cycling seeds
    for i in (n_seeds+1):(n_seeds + n_cycling)
        pop[i, :] = all_on
        if !isempty(unit_starts)
            n_cyc  = rand(rng, 1:min(3, length(unit_starts)))
            chosen = shuffle!(rng, collect(1:length(unit_starts)))[1:n_cyc]
            for idx in chosen
                u_start = unit_starts[idx]
                for j in (u_start+1):min(u_start + T÷3, u_start + T - 1)
                    if j <= n_vars; pop[i, j] = 0.0; end
                end
            end
        end
    end

    # Tier 2: mid-range U(0.3, 0.7)
    for i in (n_seeds + n_cycling + 1):(n_seeds + n_cycling + n_midrange)
        for j in 1:n_vars
            pop[i, j] = 0.3 + rand(rng) * 0.4
        end
    end

    # Tier 3: all-on with small perturbation
    for i in (n_seeds + n_cycling + n_midrange + 1):(n_seeds + n_structured)
        pop[i, :] = all_on
        for j in 1:n_vars
            if rand(rng) < 0.1
                pop[i, j] = clamp(pop[i, j] + randn(rng) * 0.15, 0.0, 1.0)
            end
        end
    end

    # Cold: full random U(0,1)
    for i in (n_seeds + n_structured + 1):P
        for j in 1:n_vars
            pop[i, j] = rand(rng)
        end
    end
end

# ── Seed builder ───────────────────────────────────────────────────────────────
"""
Build heuristic seed solutions:
1. All-on (x=1.0 everywhere) — upper bound baseline
2. Merit-order seeds: cheap units at 1.0, expensive at 0.0
3. Night-cycling candidates: all-on but overnight=0 for each cyclable unit
4. Pair candidates: overnight=0 for pairs of cyclable units
"""
function build_seeds(
    n_vars, T, cyclable_offsets, max_pair_units,
    overnight_start, overnight_end,
    regions, region_data, fixed_commits, rsizes_ga, links,
    S, R, lambda_r, alpha, passes,
    costs_bufs, marginal_p_bufs, remain_2d_bufs, export_avail_bufs,
    ptdf, net_export_bufs,
    verbose,
)::Vector{Vector{Float64}}

    all_on   = ones(Float64, n_vars)

    # Thread-safe evaluation using per-thread buffers (same as GA epochs).
    function eval_seed(s::Vector{Float64})::Float64
        tid = Threads.threadid()
        return EngineBatch.evaluate_individual(
            regions, region_data, fixed_commits, rsizes_ga, links,
            s, S, T, R, lambda_r, alpha, passes,
            costs_bufs[tid], marginal_p_bufs[tid],
            remain_2d_bufs[tid], export_avail_bufs[tid],
            ptdf, net_export_bufs[tid],
        )
    end

    all_on_fit = eval_seed(all_on)
    seeds      = Vector{Vector{Float64}}([copy(all_on)])
    seed_fits  = Float64[all_on_fit]
    if verbose
        println("  [seeds] all-on=€$(round(Int, all_on_fit))  cyclable=$(length(cyclable_offsets))/$(n_vars÷T)")
        flush(stdout)
    end

    # ── Single-unit overnight candidates (parallelised) ───────────────────────
    # Build all candidate chromosomes first, then evaluate in parallel.
    n_cyc = length(cyclable_offsets)
    single_cands = Vector{Vector{Float64}}(undef, n_cyc)
    for (k, u_off) in enumerate(cyclable_offsets)
        cand = copy(all_on)
        for j in (u_off + overnight_start - 1):(u_off + overnight_end - 2)
            if j <= n_vars; cand[j] = 0.0; end
        end
        single_cands[k] = cand
    end

    single_fits = Vector{Float64}(undef, n_cyc)
    Threads.@threads for k in 1:n_cyc
        single_fits[k] = eval_seed(single_cands[k])
    end

    n_single_kept = 0
    for k in 1:n_cyc
        if single_fits[k] < all_on_fit
            push!(seeds, single_cands[k]); push!(seed_fits, single_fits[k])
            n_single_kept += 1
        end
    end

    # ── Pair candidates (parallelised) ───────────────────────────────────────
    n_pairs = min(max_pair_units, n_cyc)
    pair_list = Tuple{Int,Int}[
        (ia, ib)
        for ia in 1:min(n_pairs, n_cyc)
        for ib in (ia+1):min(n_pairs, n_cyc)
    ]
    n_pl = length(pair_list)
    pair_cands_a = Vector{Vector{Float64}}(undef, n_pl)
    pair_cands_b = Vector{Vector{Float64}}(undef, n_pl)
    for (k, (ia, ib)) in enumerate(pair_list)
        u_off_a = cyclable_offsets[ia]; u_off_b = cyclable_offsets[ib]
        ca = copy(all_on); cb = copy(all_on)
        for j in (u_off_a + overnight_start - 1):(u_off_a + overnight_end - 2)
            if j <= n_vars; ca[j] = 0.0; cb[j] = 0.0; end
        end
        for j in (u_off_b + overnight_start - 1):(u_off_b + overnight_end - 2)
            if j <= n_vars; cb[j] = 0.0; end
        end
        pair_cands_a[k] = ca; pair_cands_b[k] = cb
    end

    pair_fits_a = Vector{Float64}(undef, n_pl)
    pair_fits_b = Vector{Float64}(undef, n_pl)
    Threads.@threads for k in 1:n_pl
        pair_fits_a[k] = eval_seed(pair_cands_a[k])
        pair_fits_b[k] = eval_seed(pair_cands_b[k])
    end

    n_pair_kept = 0
    for k in 1:n_pl
        if pair_fits_a[k] < all_on_fit
            push!(seeds, pair_cands_a[k]); push!(seed_fits, pair_fits_a[k])
            n_pair_kept += 1
        end
        if pair_fits_b[k] < all_on_fit
            push!(seeds, pair_cands_b[k]); push!(seed_fits, pair_fits_b[k])
            n_pair_kept += 1
        end
    end

    if verbose
        println("  [seeds] single=$(n_single_kept)/$(n_cyc) kept  pairs=$(n_pair_kept)/$(n_pl*2) kept")
        flush(stdout)
    end
    return seeds
end

# ── Main GA entry point ────────────────────────────────────────────────────────
function run_ga(payload)::Tuple{Vector{Float64}, Vector{Float64}}
    payload  = Dict{Any,Any}(payload)

    epochs         = Int(get(payload, "epochs",        200))
    pop_size       = Int(get(payload, "pop_size",       50))
    pc             = Float64(get(payload, "pc",         0.85))
    pm             = Float64(get(payload, "pm",         0.005))
    p_block        = Float64(get(payload, "p_block",    0.20))
    migrate_freq   = Int(get(payload, "migrate_freq",   20))
    rng_seed       = Int(get(payload, "rng_seed",       42))
    verbose        = Bool(get(payload, "verbose",       true))
    overnight_end  = Int(get(payload, "overnight_end",  8))
    max_pair_units = Int(get(payload, "max_pair_units", 15))

    adaptive_epochs_enabled  = Bool(get(payload, "adaptive_epochs_enabled",   false))
    adaptive_patience        = Int(get(payload, "adaptive_patience",          80))
    adaptive_min_improvement = Float64(get(payload, "adaptive_min_improvement", 5e-4))
    warm_start_enabled       = Bool(get(payload, "warm_start_enabled",        false))
    warm_start_fraction      = Float64(get(payload, "warm_start_fraction",    0.25))
    prev_best_raw = get(payload, "prev_best_solution", nothing)

    rng = MersenneTwister(rng_seed)

    S        = Int(payload["S"])
    T        = Int(payload["T"])
    regions  = String[String(r) for r in payload["regions"]]
    lambda_r = Float64(get(payload, "lambda_risk", 0.15))
    alpha    = Float64(get(payload, "cvar_alpha",  0.05))
    passes   = Int(get(payload, "flow_passes",     2))
    R        = length(regions)

    overnight_end = min(overnight_end, T - 4)

    rd_outer    = Dict{Any,Any}(payload["region_data"])
    region_data = Dict{String, EngineBatch.RegionData}(
        r => EngineBatch.build_region_data(Dict{Any,Any}(rd_outer[r]), S, T)
        for r in regions
    )

    fixed_commits = Dict{String, Matrix{Float64}}()
    for r in regions
        fc_raw  = Dict{Any,Any}(rd_outer[r])["fixed_units_commit"]
        N_fixed = length(fc_raw)
        fixed_commits[r] = N_fixed > 0 ?
            EngineBatch.to_mat(fc_raw, N_fixed, T) :
            Matrix{Float64}(undef, 0, T)
    end

    links = EngineBatch.LinkData[]
    reg_idx = Dict{String,Int}(r => i for (i,r) in enumerate(regions))
    if haskey(payload, "links")
        for lk in payload["links"]
            lk_d = Dict{Any,Any}(lk)
            push!(links, EngineBatch.LinkData(
                reg_idx[String(lk_d["region_a"])], reg_idx[String(lk_d["region_b"])],
                Float64(lk_d["atc_ab"]), Float64(lk_d["atc_ba"]), Float64(lk_d["loss_factor"]),
            ))
        end
    end

    raw_rsizes = payload["region_sizes"]
    rsizes_ga  = Tuple{Int,Int}[(Int(rs[1]), Int(rs[2])) for rs in raw_rsizes]
    n_vars     = sum(n * T for (n, _) in rsizes_ga)

    raw_offsets      = get(payload, "cyclable_offsets", Int[])
    cyclable_offsets = Int[Int(x) + 1 for x in raw_offsets]

    n_threads = Threads.nthreads()
    costs_bufs        = [zeros(Float64, S, R) for _ in 1:n_threads]
    marginal_p_bufs   = [zeros(Float64, S, R) for _ in 1:n_threads]
    remain_2d_bufs    = [zeros(Float64, S, R) for _ in 1:n_threads]
    export_avail_bufs = [zeros(Float64, S, R) for _ in 1:n_threads]

    ptdf = if haskey(payload, "ptdf_matrix") && length(payload["ptdf_matrix"]) > 0
        pm_raw = payload["ptdf_matrix"]
        L_ac = length(pm_raw); Rp = L_ac > 0 ? length(pm_raw[1]) : 0
        mat  = Matrix{Float64}(undef, L_ac, Rp)
        for l in 1:L_ac, r in 1:Rp; mat[l,r] = Float64(pm_raw[l][r]); end
        ram  = Float64[Float64(v) for v in payload["ptdf_ram"]]
        EngineBatch.PTDFData(L_ac, Rp, mat, ram)
    else
        EngineBatch.PTDFData()
    end
    net_export_bufs = [zeros(Float64, max(ptdf.R, 1)) for _ in 1:n_threads]

    # ── Helper: evaluate one solution (thread 1 buffers) ─────────────────────
    function eval_one(s::Vector{Float64})::Float64
        return EngineBatch.evaluate_individual(
            regions, region_data, fixed_commits, rsizes_ga, links,
            s, S, T, R, lambda_r, alpha, passes,
            costs_bufs[1], marginal_p_bufs[1],
            remain_2d_bufs[1], export_avail_bufs[1],
            ptdf, net_export_bufs[1],
        )
    end

    # ── Build seeds ────────────────────────────────────────────────────────────
    seeds = build_seeds(
        n_vars, T, cyclable_offsets, max_pair_units,
        2, overnight_end,
        regions, region_data, fixed_commits, rsizes_ga, links,
        S, R, lambda_r, alpha, passes,
        costs_bufs, marginal_p_bufs, remain_2d_bufs, export_avail_bufs,
        ptdf, net_export_bufs,
        verbose,
    )

    # Add peak-commitment seed: all thermal at f=1.0 during peak hours.
    # The GA starts from all-on seeds and removes units to save fuel.
    # The peak seed lets it start from a fully-committed peak and REFINE,
    # rather than having to BUILD UP to full commitment from under-committed starts.
    peak_seed = peak_commitment_seed(n_vars, T, rsizes_ga, 15, 23)
    peak_fit  = eval_one(peak_seed)
    if verbose
        println("  [seeds] peak-commitment seed: €$(round(Int, peak_fit))")
        flush(stdout)
    end
    push!(seeds, peak_seed)

    if verbose
        println("  [julia_ga] GA: $(pop_size) individuals × $(epochs) epochs on $(n_threads) threads")
        flush(stdout)
    end

    # ── Initialise population ──────────────────────────────────────────────────
    pop = Matrix{Float64}(undef, pop_size, n_vars)
    init_population!(pop, pop_size, n_vars, T, seeds, rsizes_ga, rng)

    if warm_start_enabled && prev_best_raw !== nothing
        prev_best = Float64[Float64(x) for x in prev_best_raw]
        if length(prev_best) == n_vars
            n_warm = max(1, round(Int, pop_size * warm_start_fraction))
            pop[1, :] = prev_best
            for k in 2:n_warm
                pop[k, :] = copy(prev_best)
                mutate!(pop[k, :], 0.02, MersenneTwister(42+k), cyclable_offsets, T, p_block, overnight_end)
            end
        end
    end

    fitness = fill(Inf, pop_size)
    eval_population!(
        fitness, pop, pop_size,
        regions, region_data, fixed_commits, rsizes_ga, links,
        S, T, R, lambda_r, alpha, passes,
        costs_bufs, marginal_p_bufs, remain_2d_bufs, export_avail_bufs,
        ptdf, net_export_bufs,
    )

    best_idx = argmin(fitness)
    best_sol = copy(pop[best_idx, :])
    best_fit = fitness[best_idx]
    convergence = [float(best_fit)]
    t_eval = 0.0; t_ga = 0.0

    island_size = pop_size ÷ 2
    k_way       = max(3, island_size ÷ 5)
    c1_buf      = Vector{Float64}(undef, n_vars)
    c2_buf      = Vector{Float64}(undef, n_vars)
    new_pop     = Matrix{Float64}(undef, pop_size, n_vars)
    new_fitness = fill(Inf, pop_size)

    # ════════════════════════════════════════════════════════════════════════════
    # ── GA epoch loop ──────────────────────────────────────────────────────────
    # ════════════════════════════════════════════════════════════════════════════
    for epoch in 1:epochs
        t_ga_start = time()

        for island_start in (0, island_size)
            island_end = island_start + island_size
            isl_pop    = @view pop[(island_start+1):island_end, :]
            isl_fit    = @view fitness[(island_start+1):island_end]
            isl_best   = argmin(isl_fit)

            i = island_start + 1
            while i <= island_end - 1
                p1_idx = tournament_idx(isl_fit, k_way, rng)
                p2_idx = tournament_idx(isl_fit, k_way, rng)
                p1 = @view isl_pop[p1_idx, :]
                p2 = @view isl_pop[p2_idx, :]

                crossover_block!(c1_buf, c2_buf, p1, p2, rsizes_ga, pc, rng)
                mutate!(c1_buf, pm, rng, cyclable_offsets, T, p_block, overnight_end)
                new_pop[i, :] = c1_buf

                if i + 1 <= island_end
                    mutate!(c2_buf, pm, rng, cyclable_offsets, T, p_block, overnight_end)
                    new_pop[i+1, :] = c2_buf
                end
                i += 2
            end
            new_pop[island_start+1, :] = isl_pop[isl_best, :]
        end

        if epoch % migrate_freq == 0
            best0  = argmin(@view fitness[1:island_size])
            best1  = island_size + argmin(@view fitness[(island_size+1):pop_size])
            worst0 = argmax(@view fitness[1:island_size])
            worst1 = island_size + argmax(@view fitness[(island_size+1):pop_size])
            new_pop[worst1, :] = new_pop[best0, :]
            new_pop[worst0, :] = new_pop[best1, :]
        end
        new_pop[1, :] = best_sol
        t_ga += time() - t_ga_start

        t_eval_start = time()
        eval_population!(
            new_fitness, new_pop, pop_size,
            regions, region_data, fixed_commits, rsizes_ga, links,
            S, T, R, lambda_r, alpha, passes,
            costs_bufs, marginal_p_bufs, remain_2d_bufs, export_avail_bufs,
            ptdf, net_export_bufs,
        )
        t_eval += time() - t_eval_start

        copyto!(pop, new_pop); copyto!(fitness, new_fitness)
        epoch_best = argmin(fitness)
        if fitness[epoch_best] < best_fit
            best_fit = fitness[epoch_best]
            best_sol = copy(pop[epoch_best, :])
        end
        push!(convergence, float(best_fit))

        if adaptive_epochs_enabled && length(convergence) >= adaptive_patience + 1
            recent = convergence[end - adaptive_patience]
            if (recent - best_fit) / max(abs(recent), 1.0) < adaptive_min_improvement
                if verbose
                    println("    [adaptive] GA converged at epoch $(epoch)/$(epochs)")
                    flush(stdout)
                end
                break
            end
        end

        if verbose && epoch % 50 == 0
            println("    epoch $(lpad(epoch,3))/$(epochs)  best=€$(round(Int,best_fit))  eval=$(round(t_eval,digits=1))s  ga=$(round(t_ga,digits=1))s")
            flush(stdout)
        end
    end

    if verbose
        println("  [julia_ga] done: best=€$(round(Int,best_fit))")
        flush(stdout)
    end
    return best_sol, convergence
end

# ── Peak-commitment seed ───────────────────────────────────────────────────────
# Adds one chromosome with all thermal at f=1.0 during evening peak hours (15-23).
# Currently build_seeds only generates seeds that cycle units OFF overnight.
# The GA must work upward from under-committed to find full-commitment solutions.
# This seed gives the GA a fully-committed starting point to refine instead.
function peak_commitment_seed(
    n_vars::Int, T::Int, rsizes_ga::Vector{Tuple{Int,Int}},
    peak_start::Int, peak_end::Int,
)::Vector{Float64}
    s = zeros(Float64, n_vars)
    offset = 0
    for (N, _) in rsizes_ga
        for u in 1:N
            for t in 1:T
                # Full commitment during peak hours, 0.5 (half) otherwise
                # 0.5 during non-peak = unit nominally on but at half dispatch
                # The GA will refine this: push non-peak toward 0 (off) to save fuel
                h = t  # 1-indexed hour within window
                s[offset + (u-1)*T + t] = (h >= peak_start && h <= peak_end) ? 1.0 : 0.5
            end
        end
        offset += N * T
    end
    return s
end

end  # module EngineGA
