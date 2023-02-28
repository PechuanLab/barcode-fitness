using DataFrames, LinearAlgebra
using StatsBase, Statistics
using Distributions
using Distances
using KernelDensity, KernelDensityEstimate
using Optim
include("rejection_sampler.jl")
include("BC_fit_sample_fns.jl")
include("BC_fit_utils.jl")

function add_clones(num_cells::Vector{Int64}, new_times::Vector{Int64}, new_freqs::Vector{Float64}, end_time::Int64)
	total_to_add = 0.0
	cells_to_add = num_cells .* 0.0
	total_cells = sum(num_cells)
	for i in 1:length(new_times)
		if new_times[i] == end_time
			total_to_add += new_freqs[i]
			cells_to_add[i] = total_cells * new_freqs[i]
		end
	end
	if total_to_add == 0.0
		return num_cells
	else
		to_return = num_cells .* (1-total_to_add)
		to_return = round.(to_return + cells_to_add)
		return to_return
	end
end

## Growth functions
# Arguments: Population, start time, end time
# Returns: nothing, but mutates Population.N

function pure_birth_growth(pop::PureBirth, start_time::Int64, end_time::Int64)
	time_passed = end_time - start_time

	mean_val = pop.N.*exp.(time_passed*pop.birth_rates)
	stdev_val = ((pop.N.*exp.(2*time_passed*pop.birth_rates)).*(ones(length(pop.birth_rates))-exp.(-time_passed*pop.birth_rates))).^(1/2)

	pop.N = broadcast(sample_clipped_normal, mean_val, stdev_val)
	return nothing
end

function bd_approx_growth(pop::BirthDeathSample, start_time::Int64, end_time::Int64)
	time_passed = end_time - start_time
	sample_ratio = pop.samples[end_time]/pop.samples[start_time]

	pop.N = round.(broadcast(sample_bd_approx, pop.N, pop.growth_rates, pop.variances, sample_ratio))
	return nothing
end

function bd_evosim_growth(pop::BirthDeathEvoSim, start_time::Int64, end_time::Int64)
	time_passed = convert(Float64, end_time - start_time)

	pop.N = round.(broadcast(sample_evosim_nomut, pop.N, pop.birth_rates, pop.death_rates, time_passed))
	return nothing
end

function bd_gillespie_growth(pop::BirthDeathGillespie, start_time::Int64, end_time::Int64)
	time_passed = convert(Float64, end_time - start_time)

	pop.N = round.(broadcast(sample_gillespie_nomut, pop.N, pop.birth_rates, pop.death_rates, time_passed))
	return nothing
end

function bd_lazyhybrid_growth(pop::BirthDeathLazyHybrid, start_time::Int64, end_time::Int64)
	time_passed = convert(Float64, end_time - start_time)

	pop.N = round.(broadcast(sample_lazyhybrid_nomut, pop.N, pop.birth_rates, pop.death_rates, time_passed))
	return nothing
end

function bd_NegBin_growth(pop::BirthDeathNegBin, start_time::Int64, end_time::Int64)
	time_passed = convert(Float64, end_time - start_time)

	pop.N = round.(broadcast(sample_NegBinBD_nomut, pop.N, pop.birth_rates, pop.death_rates, time_passed))
	return nothing
end

function bd_insertclone_growth(pop::SubcloneCNA, start_time::Int64, end_time::Int64)
	time_passed = convert(Float64, end_time - start_time)
	pop.N = round.(broadcast(sample_NegBinBD_nomut, pop.N, pop.birth_rates, pop.death_rates, time_passed))
	pop.N = add_clones(pop.N, pop.first_times, pop.first_freqs, end_time)
	return nothing
end

function bd_onemut_growth(pop::BirthDeathOneMut, start_time::Int64, end_time::Int64)
	muts_to_resolve = Vector{Vector}()

	for i = 1:length(pop.mut_times)
		if (pop.mut_times[i] >= start_time) & (pop.mut_times[i] < end_time)
			push!(muts_to_resolve, [pop.mut_times[i], i])
		end
	end

	if length(muts_to_resolve) > 0
		sort_order = sortperm([x[1] for x in muts_to_resolve])
		sorted_times = muts_to_resolve[sort_order]
		curr_time = start_time
		for mut in sorted_times
			time_passed = convert(Float64, mut[1] - curr_time)
			pop.N_WT = round.(broadcast(sample_NegBinBD_nomut, pop.N_WT, pop.birth_rates_WT, pop.death_rates_WT, time_passed))
			pop.N_mut = round.(broadcast(sample_NegBinBD_nomut, pop.N_mut, pop.birth_rates_mut, pop.death_rates_mut, time_passed))
			if pop.N_WT[floor(Int, mut[2])] >= 1
				pop.N_mut[floor(Int, mut[2])] += 1
				pop.N_WT[floor(Int, mut[2])] -= 1
			end
			curr_time = mut[1]
		end
		time_passed = convert(Float64, end_time - curr_time)
		pop.N_WT = round.(broadcast(sample_NegBinBD_nomut, pop.N_WT, pop.birth_rates_WT, pop.death_rates_WT, time_passed))
		pop.N_mut = round.(broadcast(sample_NegBinBD_nomut, pop.N_mut, pop.birth_rates_mut, pop.death_rates_mut, time_passed))
	else
		time_passed = convert(Float64, end_time - start_time)
		pop.N_WT = round.(broadcast(sample_NegBinBD_nomut, pop.N_WT, pop.birth_rates_WT, pop.death_rates_WT, time_passed))
		pop.N_mut = round.(broadcast(sample_NegBinBD_nomut, pop.N_mut, pop.birth_rates_mut, pop.death_rates_mut, time_passed))
	end

	return nothing
end
