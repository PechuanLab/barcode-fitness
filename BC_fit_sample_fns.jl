using DataFrames, LinearAlgebra
using StatsBase, Statistics
using Distributions
using Distances
using KernelDensity, KernelDensityEstimate
using Optim
include("rejection_sampler.jl")
include("BC_fit_utils.jl")

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

function poisson_passage(pop::BirthDeathOneMut, pass_num::Int64, growth_num::Int64)
	freqs = get_freq_WT(pop, growth_num)
	lambda_val = freqs * pass_num
	pop.N_WT = broadcast(sample_poisson, lambda_val)

	freqs = get_freq_mut(pop, growth_num)
	lambda_val = freqs * pass_num
	pop.N_mut = broadcast(sample_poisson, lambda_val)
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

function CNA_sample(pop::SubcloneCNA, sample_num::Int64, growth_num::Int64)
	freqs = get_freq_BCs(pop, get_total_pop_size(pop))
	freqs = reshape(freqs, 1, length(freqs))
	#=
	print("\n")
	print(freqs)
	print("\n")
	print(vec(freqs * pop.CNA_clones))
	=#
	return vec(freqs * pop.CNA_clones)
end
