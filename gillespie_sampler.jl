using DataFrames
using Distributions

function step_gillespie_nomut(init_N::Int64, b::Float64, d::Float64)
    comb_rate = b + d
    dist = Exponential(1/(comb_rate*init_N))
    new_time = rand(dist, 1)[1]
    dist = Bernoulli(b/comb_rate)
    new_cell = rand(dist, 1)[1]
    if new_cell == 1
        return (init_N+1, new_time)
    else
        return (init_N-1, new_time)
    end
end

function deterministic_growth(init_N::Int64, b::Float64, d::Float64, end_time::Float64)
    new_pop = init_N*exp((b-d)*end_time)
    return convert(Int64, round(new_pop))
end

function sample_gillespie_nomut(init_N::Int64, b::Float64, d::Float64, end_time::Float64)
    t = 0.
    new_pop = init_N
    curr_pop = init_N
    while (t < end_time) & (curr_pop > 0)
        curr_pop = new_pop
        new_pop, new_time = step_gillespie_nomut(curr_pop, b, d)
        t += new_time
    end
    return curr_pop
end

function sample_lazyhybrid_nomut(init_N::Int64, b::Float64, d::Float64, end_time::Float64, threshold::Float64=1e3)
    t = 0.
    new_pop = init_N
    curr_pop = init_N
    while (t < end_time) & (curr_pop < threshold) & (curr_pop > 0)
        curr_pop = new_pop
        new_pop, new_time = step_gillespie_nomut(curr_pop, b, d)
        t += new_time
    end
    if (t > end_time) | (curr_pop == 0)
        return curr_pop
    else
        curr_pop = new_pop
        remaining_time = end_time - t
        return deterministic_growth(curr_pop, b, d, remaining_time)
    end
end

function sample_NegBinBD_nomut(init_N::Int64, b::Float64, d::Float64, end_time::Float64)
    if sum(init_N) == 0
        return 0
    end
    lambda = b - d
    alpha = (d*exp(lambda*end_time) - d)/(b*exp(lambda*end_time) - d)
    beta = (b*exp(lambda*end_time)-b)/(b*exp(lambda*end_time) - d)
    dist = Binomial(init_N, 1.0 - alpha)
    surviving = rand(dist, 1)[1]
    dist = NegativeBinomial(surviving, 1.0 - beta)
    return surviving + rand(dist, 1)[1]
end
