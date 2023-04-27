using DataFrames, LinearAlgebra
using StatsBase, Statistics
using Distributions
using Distances
using KernelDensity, KernelDensityEstimate
using Optim
include("rejection_sampler.jl")

## Cost functions
# Arguments: output matrix (sampled times x BCs), ground truth read matrix (sampled times x BCs)
# Returns: float cost value (larger is worse)

# HELPER FUNCTION TO CHECK VALIDITY OF COST FUNCTION INPUTS
function check_cost_inputs(sim_output::Matrix, ground_truth::Matrix)
	@assert size(sim_output) == size(ground_truth)
	return nothing
end

function euclidean_cost(sim_output::Matrix, ground_truth::Matrix)
	check_cost_inputs(sim_output, ground_truth)
	#print("\n")
 	#print(sim_output)
	#print((colwise(Euclidean(), ground_truth, sim_output)))
	#print("\n")
	#print(ground_truth)
	#print((sum(colwise(Euclidean(), ground_truth, sim_output)))^(1/2))
	return (sum(colwise(Euclidean(), ground_truth, sim_output)))^(1/2)
end

function weighted_euclidean_cost(sim_output::Matrix, ground_truth::Matrix)
	check_cost_inputs(sim_output, ground_truth)
	time_weights = ones(size(sim_output))
	#time_weights[1:3] .= 0.2
	#time_weights[length(time_weights)-4:length(time_weights)] .= 0.5
	#print("\n")
	#print((colwise(Euclidean(), ground_truth, sim_output)))
	#print((sum(colwise(Euclidean(), ground_truth, sim_output)))^(1/2))
	return (sum(colwise(Euclidean(), ground_truth .* time_weights, sim_output .* time_weights)))^(1/2)
end

## Getters/setters
# (!!!) reset_pop functions- required for simulation. The reset_pop function used is determined by the Population type and requires 3 inputs: the Population, init_N (initial population sizes), and all additional parameters in params to set. See example.

function get_total_pop_size(pop::Population)
	return sum(pop.N)
end

function get_num_BCs(pop::Population)
	return length(pop.N)
end

function get_num_outputs(pop::Population)
	return get_num_BCs(pop)
end

function get_num_outputs(pop::SubcloneCNA)
	return size(pop.CNA_clones)[2]
end

function get_freq_BCs(pop::Population, total_cells::Int64)
	return min.(pop.N/total_cells,1)
end

function get_total_pop_size(pop::BirthDeathOneMut)
	return sum(pop.N_WT) + sum(pop.N_mut)
end

function get_num_BCs(pop::BirthDeathOneMut)
	return length(pop.N_WT)
end

function get_freq_BCs(pop::BirthDeathOneMut, total_cells::Int64)
	return min.((pop.N_WT .+ pop.N_mut)/total_cells,1)
end

function get_freq_WT(pop::BirthDeathOneMut, total_cells::Int64)
	return min.((pop.N_WT)/total_cells,1)
end

function get_freq_mut(pop::BirthDeathOneMut, total_cells::Int64)
	return min.((pop.N_mut)/total_cells,1)
end

function reset_pop(pop::PureBirth, init_N::Vector{Int64}, params_to_set)
	pop.N = init_N
	pop.birth_rates = params_to_set.birth_rates
	@assert length(pop.N) == length(pop.birth_rates)
	return nothing
end

function reset_pop(pop::BirthDeathSample, init_N::Vector{Int64}, params_to_set)
	pop.N = init_N
	pop.growth_rates = params_to_set.growth_rates
	pop.variances = params_to_set.variances
	pop.samples = params_to_set.samples
	@assert length(pop.N) == length(pop.growth_rates)
	@assert length(pop.N) == length(pop.variances)
	return nothing
end

function reset_pop(pop::BirthDeathEvoSim, init_N::Vector{Int64}, params_to_set)
	pop.N = init_N
	pop.birth_rates = params_to_set.birth_rates
	pop.death_rates = params_to_set.death_rates
	@assert length(pop.N) == length(pop.birth_rates)
	@assert length(pop.N) == length(pop.death_rates)
	return nothing
end

function reset_pop(pop::BirthDeathGillespie, init_N::Vector{Int64}, params_to_set)
	pop.N = init_N
	pop.birth_rates = params_to_set.birth_rates
	pop.death_rates = params_to_set.death_rates
	@assert length(pop.N) == length(pop.birth_rates)
	@assert length(pop.N) == length(pop.death_rates)
	return nothing
end

function reset_pop(pop::BirthDeathLazyHybrid, init_N::Vector{Int64}, params_to_set)
	pop.N = init_N
	pop.birth_rates = params_to_set.birth_rates
	pop.death_rates = params_to_set.death_rates
	@assert length(pop.N) == length(pop.birth_rates)
	@assert length(pop.N) == length(pop.death_rates)
	return nothing
end

function reset_pop(pop::BirthDeathNegBin, init_N::Vector{Int64}, params_to_set)
	pop.N = init_N
	pop.birth_rates = params_to_set.birth_rates
	pop.death_rates = params_to_set.death_rates
	@assert length(pop.N) == length(pop.birth_rates)
	@assert length(pop.N) == length(pop.death_rates)
	return nothing
end

function reset_pop(pop::SubcloneCNA, init_N::Vector{Int64}, params_to_set)
	pop.N = init_N
	pop.birth_rates = params_to_set.birth_rates
	pop.death_rates = params_to_set.death_rates
	@assert length(pop.N) == length(pop.birth_rates)
	@assert length(pop.N) == length(pop.death_rates)

	pop.first_times = params_to_set.first_times
	pop.first_freqs = params_to_set.first_freqs
	@assert length(pop.N) == length(pop.first_times)
	@assert length(pop.N) == length(pop.first_freqs)

	pop.CNA_clones = params_to_set.CNA_clones
	@assert size(pop.CNA_clones)[1] == length(pop.N)
	return nothing
end

function reset_pop(pop::BirthDeathOneMut, init_N::Vector{Int64}, params_to_set)
	pop.N_WT = init_N
	pop.birth_rates_WT = params_to_set.birth_rates_WT
	pop.death_rates_WT = params_to_set.death_rates_WT
	pop.birth_rates_mut = params_to_set.birth_rates_mut
	pop.death_rates_mut = params_to_set.death_rates_mut
	pop.mut_times = params_to_set.mut_times
	pop.N_mut = round.(zeros(length(pop.N_WT)))

	@assert length(pop.N_WT) == length(pop.birth_rates_WT)
	@assert length(pop.N_WT) == length(pop.death_rates_WT)
	@assert length(pop.N_WT) == length(pop.mut_times)
	return nothing
end

## Other helper functions

function complete_posterior(ressmc,param_names)
	#Save result
	ABCResult = DataFrame()
	for i = 1:length(param_names)
		colname = param_names[i]
		ABCResult[!,colname]= ressmc[1][i].particles
	end
	return ABCResult
end

function compute_mode_CI(dist::Vector{Float64}, CI_width::Float64)
	CI_lower = (1-CI_width)/2.0
	CI = quantile(dist, [CI_lower, 1-CI_lower])
	smoothed = kde(dist)
	MAP_estimate = smoothed.x[findmax(smoothed.density)[2]]
	return (MAP_estimate, CI[1], CI[2])
end

function compute_mode_CI_ll(dist::Vector{Float64}, CI_width::Float64, prior::Distribution)
	CI_lower = (1-CI_width)/2.0
	CI = quantile(dist, [CI_lower, 1-CI_lower])
	smoothed = kde(dist)
	max_idx = findmax(smoothed.density)[2]
	MAP_estimate = smoothed.x[max_idx]
	log_like = log(smoothed.density[max_idx])
	log_like = log_like - logpdf(prior, MAP_estimate)
	return (MAP_estimate, CI[1], CI[2], log_like)
end

function compute_mode_CI_multi(dist::Matrix{Float64}, CI_width::Float64, prior::Distribution, prior_ranges)
	num_dim = size(dist, 1);
	#to_kde = [dist[:, i] for i in 1:size(dist, 2)]
	#dims = repeat([ContinuousDim()], num_dim);

	#bw = [(x[2]-x[1])/5 for x in prior_ranges]

	mv_kde = kde!(dist, [std(dist[i,:])/2. for i in 1:num_dim])

	lower = [x[1] for x in prior_ranges]
	upper = [x[2] for x in prior_ranges]
	initial_x = vec(mean(dist, dims=2))#[(x[2]-x[1])/2 for x in prior_ranges]

	function to_optim(input_vec::Vector{Float64})
		#print("\n")
		#print(input_vec)
		to_return = -mv_kde(input_vec[:,:])[1]
		#print("\n")
		#print(to_return)
		return to_return
	end

	inner_optimizer = LBFGS()
	MAP_estimate = optimize(to_optim, lower, upper, initial_x, Fminbox(inner_optimizer))
	print("\n")
	print(log(-MAP_estimate.minimum))
	print("\n")
	print(loglikelihood(prior, MAP_estimate.minimizer))
	log_like = log(-MAP_estimate.minimum) - loglikelihood(prior, MAP_estimate.minimizer)

	CI_lower = (1-CI_width)/2.0
	CIs = [quantile(dist[i,:], [CI_lower, 1-CI_lower]) for i in 1:num_dim]
	lower_CIs = [x[1] for x in CIs]
	upper_CIs = [x[2] for x in CIs]

	return (MAP_estimate.minimizer, lower_CIs, upper_CIs, log_like)
end
