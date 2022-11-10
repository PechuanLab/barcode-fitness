using DataFrames
using StatsBase
using Distributions

function get_sample_params(growth_pdf::Function, tol::Float64=1e-6, min_cdf::Float64=0.2)
    # returns ([upper limit for sampling], [max of pdf])
    curr_val = 1.
    total_cdf = 0.
    max_val = 0.
    curr_pdf = 1.
    #while 1-total_cdf > tol:
    while (total_cdf < min_cdf) || (curr_pdf > tol)
        #print(total_cdf)
        #print(curr_pdf)
        curr_pdf = growth_pdf(curr_val)
        total_cdf += curr_pdf
        max_val = max(max_val, curr_pdf)
        curr_val += 1
    end
    return (curr_val, max_val)
end

function sample_growth(growth_pdf::Function, sample_params::Tuple{Float64, Float64}, n_samples::Int=1)
    sampled = Vector{Float64}()
    while length(sampled) < n_samples
        x_proposed_dist = Uniform(0, sample_params[1])
        x_proposed = rand(x_proposed_dist, 1)[1]
        y_proposed = growth_pdf(x_proposed)
        y_sampled_dist = Uniform(0, sample_params[2])
        y_sampled = rand(y_sampled_dist, 1)[1]
        if y_sampled < y_proposed
            push!(sampled, x_proposed)
        end
    end
    return sampled
end
