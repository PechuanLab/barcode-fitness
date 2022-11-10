using DataFrames
using StatsBase
using Distributions
using Distances
using KernelDensity
include("rejection_sampler.jl")
#include("evosim_BP_sampler.jl")
include("gillespie_sampler.jl")

## Getters/setters
# (!!!) reset_pop functions- required for simulation. The reset_pop function used is determined by the Population type and requires 3 inputs: the Population, init_N (initial population sizes), and all additional parameters in params to set. See example.

function get_total_pop_size(pop::Population)
	return sum(pop.N)
end

function get_num_BCs(pop::Population)
	return length(pop.N)
end

function get_freq_BCs(pop::Population, total_cells::Int64)
	return pop.N/total_cells
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

function sample_normal(mu::Float64, sig::Float64)
	d = Normal(mu, sig)
	return round(rand(d, 1)[1])
end

function sample_clipped_normal(mu::Float64, sig::Float64)
	norm_res = sample_normal(mu, sig)
	if norm_res < 0
		norm_res = 0
	end
	return round(norm_res)
end

function sample_poisson(lambda::Float64)
	d = Poisson(lambda)
	return rand(d, 1)[1]
end

function multinomial_counts(pop::Population, sample_size::Int64)
	freqs = get_freq_BCs(pop, get_total_pop_size(pop))
	d = Categorical(freqs)
	R = rand(d, sample_size)
	return counts(R, 1:get_num_BCs(pop))
end

function poisson_counts(pop::Population, sample_size::Int64, growth_num::Int64)
	freqs = get_freq_BCs(pop, growth_num)
	lambda_val = freqs * sample_size
	return broadcast(sample_poisson, lambda_val)
end


function sample_bd_approx(n_init::Int64, growth_rate::Float64, variance::Float64, sample_ratio::Float64)
	mean_growth = growth_rate*sample_ratio*n_init
	function bd_approx_pdf(n_final::Float64)
		to_return = sqrt(sqrt(mean_growth)/(4*pi*variance*n_final^(3/2)))
		to_return = to_return*exp(-((sqrt(n_final)-sqrt(mean_growth))^2)/variance)
		return to_return
	end
	val_at_mean = bd_approx_pdf(mean_growth)

	function to_rejection(n_final::Float64)
		return (bd_approx_pdf(n_final) * exp(2)/val_at_mean)
	end

	support = (1.0, Inf)
	sample_params = get_sample_params(to_rejection)
	return sample_growth(to_rejection, sample_params, 1)[1]
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

## Passage functions
# Arguments: Population, number of cells to be passaged, total number of cells at end of growth period
# Returns: nothing, but mutates Population

function multinomial_passage(pop::Population, pass_num::Int64, growth_num::Int64)
	pop.N = multinomial_counts(pop, pass_num)
	return nothing
end

function poisson_passage(pop::Population, pass_num::Int64, growth_num::Int64)
	pop.N = poisson_counts(pop, pass_num, growth_num)
	return nothing
end

function total_passage(pop::Population, pass_num::Int64, growth_num::Int64)
	return nothing
end

## Sample functions
# Arguments: Population, number of reads to sample, total number of cells at end of growth period
# Returns: array of ints of sampled reads, DOES NOT mutate Population

function multinomial_sample(pop::Population, sample_num::Int64, growth_num::Int64)
	return multinomial_counts(pop, sample_num)
end

function poisson_sample(pop::Population, sample_num::Int64, growth_num::Int64)
	return poisson_counts(pop, sample_num, growth_num)
end

function total_sample(pop::Population, sample_num::Int64, growth_num::Int64)
	return copy(pop.N)
end

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
	return (sum(colwise(Euclidean(), ground_truth, sim_output)))^(1/2)
end

## Simulation function (DO NOT MODIFY)

function simulate(pop::Population, growth_fn::Function, pass_fn::Function, sample_fn::Function, pass_times::Vector{Int64}, pass_nums::Vector{Int64}, sample_nums::Vector{Int64}, growth_nums::Vector{Int64})

	@assert length(pass_times) == length(pass_nums)
	@assert length(pass_times) == length(sample_nums)
	@assert length(pass_times) == length(growth_nums)

	num_passages = length(pass_times)

	sampled_data = zeros(Int64, num_passages-1, get_num_BCs(pop))
	for i in 1:(num_passages-1)
		growth_fn(pop, pass_times[i], pass_times[i+1])
		if sample_nums[i+1] > 0
			sampled_data[i,:] = sample_fn(pop, sample_nums[i+1], growth_nums[i+1])
		end
		pass_fn(pop, pass_nums[i+1], growth_nums[i+1])
	end
	return sampled_data
end

## Simulate and compute cost from output (DO NOT MODIFY)
# You will probably need to wrap this function up to use it for SMC or other estimation procedures.
# Threadsafe (probably)- copies all inputs that will be mutated

function simulate_cost(pop_template::Population, init_N::Vector{Int64}, (growth_fn, pass_fn, sample_fn, cost_fn), (pass_times, pass_nums, sample_nums, growth_nums), true_data::Matrix, params_to_set)
	#=
	INPUTS
	pop_template: Population of the correct type that will be copied and used as a template. Object copy will be reset after copying using the reset_pop function defined for the type and the inputs init_N and params_to_set.
	init_N: initial sizes of each clonal population
	growth_fn: function describing the growth process for the population (see specs above). Must be compatible with pop_template.
	pass_fn: function describing the passaging process for the population (see specs above). Must be compatible with pop_template.
	sample_fn: function describing the sampling process for sequencing (see specs above). Must be compatible with pop_template.
	cost_fn: function comparing the simulation results with the experimental data (see specs above). Must be compatible with pop_template and the true_data matrix.
	pass_times: Vector{Int64} of times at which the population was passaged
	pass_nums: Vector{Int64} of cell numbers that were seeded at each passaging time
	sample_nums: Vector{Int64} of total read numbers were sequenced at each passaging time
	growth_nums: Vector{Int64} of cell numbers that were counted at the end of growth at each passaging time
	true_data: ground truth read number matrix for each barcode at each sampled timepoint (sampled times x BCs)
	params_to_set: inputs (likely a tuple) passed into the reset_pop function to load inputs into the population.

	NB: pass_times, pass_nums, sample_nums, growth_nums MUST have the same length. If no cells were sequenced in a particular passage, the sample_num entry for that time should be 0.

	RETURNS
	Number (likely float) value for the distance between true data and one run of the simulation.
	=#

	pop = deepcopy(pop_template)
	reset_pop(pop, copy(init_N), deepcopy(params_to_set))

	sampled_data = simulate(pop, growth_fn, pass_fn, sample_fn, pass_times, pass_nums, sample_nums, growth_nums)
	sample_filter = 0:(length(sample_nums)-1)
	times_to_use = sample_filter[sample_nums .> 0]
	sim_to_test = sampled_data[times_to_use,:]
	return cost_fn(sim_to_test, true_data)
end
