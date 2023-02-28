using DataFrames, LinearAlgebra
using StatsBase, Statistics
using Distributions
using Distances
using KernelDensity, KernelDensityEstimate
#using MultiKDE
using Optim
include("rejection_sampler.jl")
include("BC_fit_utils.jl")
include("gillespie_sampler.jl")

## Simulation function (DO NOT MODIFY)

function simulate(pop::Population, growth_fn::Function, pass_fn::Function, sample_fn::Function, pass_times::Vector{Int64}, pass_nums::Vector{Int64}, sample_nums::Vector{Int64}, growth_nums::Vector{Int64})

	@assert length(pass_times) == length(pass_nums)
	@assert length(pass_times) == length(sample_nums)
	@assert length(pass_times) == length(growth_nums)

	num_passages = length(pass_times)

	sampled_data = zeros(Float64, num_passages-1, get_num_outputs(pop))
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
	sim_to_test = sim_to_test ./ sample_nums[times_to_use.+1]
	true_data = true_data ./ sample_nums[times_to_use.+1]
	return cost_fn(sim_to_test, true_data)
end
