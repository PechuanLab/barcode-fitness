using CSV
using KissABC
using DataFrames

include("BC_fit_types.jl")
include("BC_fit_sim_fns.jl")

function run_purebirth_adjtimes(BC_file::String, sampling_file::String, out_file::String, verbose::Bool=False, prior_lim::Float64=0.0)
    BC_data = CSV.read(BC_file, DataFrame)
    sampling_data = CSV.read(sampling_file, DataFrame)

    results_table = DataFrame(BC=Vector{Int64}(), time=Vector{Int64}(), MAP=Vector{Float64}(), lower_CI=Vector{Float64}(), upper_CI=Vector{Float64}())
    for i in 1:length(BC_data[:, "start_num"])
        if verbose
            print("\n")
            print(i)
        end

        start_time = BC_data[i,"time"]
        end_time = BC_data[i,"time"] + 1

        if ! (start_time in sampling_data[!, "time"]) & (end_time in sampling_data[!, "time"])
            continue
        end

        sampling_row = subset(sampling_data, :time => t -> t .== end_time)
        init_row = subset(sampling_data, :time => t -> t .== start_time)
        start_cells = init_row[1, "passaged"]
        end_cells = sampling_row[1, "counted"]
        start_day = init_row[1, "day"]
        end_day = sampling_row[1, "day"]

        mean_growth = log(end_cells/start_cells)/(end_day-start_day)

        if prior_lim > 0.0
            prior = Uniform(0,prior_lim)
        else
            prior = Uniform(0,mean_growth*3)
        end
        pop_template = PureBirth([0], [0.0])

        end_sample = sampling_row[1, "sampled"]
        true_data = Matrix{Number}(undef, 1, 1)
        true_data[1,1] = BC_data[i,"end_num"]

        init_N = round(Int, start_cells*BC_data[i,"start_num"]/init_row[1, "sampled"])
        #print(true_data[1,1]/init_N)
        #print("\n")
        function cost_fn(birth_rate::Float64)
            params_to_set = (birth_rates=[birth_rate],)
            to_return = simulate_cost(pop_template, [init_N], (pure_birth_growth, poisson_passage, poisson_sample, euclidean_cost), ([start_day, end_day], [start_cells, start_cells], [0, end_sample], [start_cells, end_cells]), true_data, params_to_set)
            return(to_return)
        end
        result_smc = smc(prior, cost_fn, nparticles=1000, parallel=true)
        MAP_growth, lower_CI_growth, upper_CI_growth = compute_mode_CI(result_smc[1][1].particles, 0.95)
        push!(results_table, [BC_data[i,"BC"], start_time, MAP_growth, lower_CI_growth, upper_CI_growth])
    end

    CSV.write(out_file, results_table)
    return nothing
end

function run_sherlockBD_adjtimes(BC_file::String, sampling_file::String, out_file::String, verbose::Bool=False, prior_lim::Float64=0.0)
    BC_data = CSV.read(BC_file, DataFrame)
    sampling_data = CSV.read(sampling_file, DataFrame)

    results_table = DataFrame(BC=Vector{Int64}(), time=Vector{Int64}(), MAP=Vector{Float64}(), lower_CI=Vector{Float64}(), upper_CI=Vector{Float64}())
    sample_dict = Dict{Int64, Int64}(sampling_data[!, "day"] .=> sampling_data[!, "sampled"])
    for i in 1:length(BC_data[:, "start_num"])
        if verbose
            print("\n")
            print(i)
        end

        start_time = BC_data[i,"time"]
        end_time = BC_data[i,"time"] + 1

        if ! (start_time in sampling_data[!, "time"]) & (end_time in sampling_data[!, "time"])
            continue
        end

        sampling_row = subset(sampling_data, :time => t -> t .== end_time)
        init_row = subset(sampling_data, :time => t -> t .== start_time)
        start_cells = init_row[1, "passaged"]
        end_cells = sampling_row[1, "counted"]
        start_day = init_row[1, "day"]
        end_day = sampling_row[1, "day"]
        kappa = init_row[1, "kappa"]

        mean_growth = log(end_cells/start_cells)/(end_day-start_day)

        prior = Uniform(0,mean_growth*3)
        pop_template = BirthDeathSample([0], [0.0], [0.0], Dict{Int64, Int64}())

        end_sample = sampling_row[1, "sampled"]
        true_data = Matrix{Number}(undef, 1, 1)
        true_data[1,1] = BC_data[i,"end_num"]

        init_N = round(Int, start_cells*BC_data[i,"start_num"]/init_row[1, "sampled"])
        #print(true_data[1,1]/init_N)
        #print("\n")
        function cost_fn(birth_rate::Float64)
            params_to_set = (growth_rates=[birth_rate], variances=[kappa], samples=sample_dict)
            to_return = simulate_cost(pop_template, [init_N], (bd_approx_growth, poisson_passage, poisson_sample, euclidean_cost), ([start_day, end_day], [start_cells, start_cells], [0, end_sample], [start_cells, end_cells]), true_data, params_to_set)
            return(to_return)
        end
        result_smc = smc(prior, cost_fn, nparticles=1000, parallel=true)
        MAP_growth, lower_CI_growth, upper_CI_growth = compute_mode_CI(result_smc[1][1].particles, 0.95)
        push!(results_table, [BC_data[i,"BC"], start_time, MAP_growth, lower_CI_growth, upper_CI_growth])
    end

    CSV.write(out_file, results_table)
    return nothing
end

function run_evosim_adjtimes(BC_file::String, sampling_file::String, out_file::String, death_rate::Float64, verbose::Bool=False, prior_lim::Float64=0.0)
    BC_data = CSV.read(BC_file, DataFrame)
    sampling_data = CSV.read(sampling_file, DataFrame)

    results_table = DataFrame(BC=Vector{Int64}(), time=Vector{Int64}(), MAP=Vector{Float64}(), lower_CI=Vector{Float64}(), upper_CI=Vector{Float64}())
    #results_lock = ReentrantLock()

    sample_dict = Dict{Int64, Int64}(sampling_data[!, "day"] .=> sampling_data[!, "sampled"])
    #Threads.@threads for i in 1:length(BC_data[:, "start_num"])
    for i in 1:length(BC_data[:, "start_num"])
        if verbose
            print("\n")
            print(i)
        end

        start_time = BC_data[i,"time"]
        end_time = BC_data[i,"time"] + 1

        if ! (start_time in sampling_data[!, "time"]) & (end_time in sampling_data[!, "time"])
            continue
        end

        sampling_row = subset(sampling_data, :time => t -> t .== end_time)
        init_row = subset(sampling_data, :time => t -> t .== start_time)
        start_cells = init_row[1, "passaged"]
        end_cells = sampling_row[1, "counted"]
        start_day = init_row[1, "day"]
        end_day = sampling_row[1, "day"]

        mean_growth = log(end_cells/start_cells)/(end_day-start_day)

        if prior_lim > 0.0
            prior = Uniform(0,prior_lim)
        else
            prior = Uniform(0,mean_growth*3)
        end
        pop_template = BirthDeathEvoSim([0], [0.0], [0.0])

        end_sample = sampling_row[1, "sampled"]
        true_data = Matrix{Number}(undef, 1, 1)
        true_data[1,1] = BC_data[i,"end_num"]

        init_N = round(Int, start_cells*BC_data[i,"start_num"]/init_row[1, "sampled"])
        #print(true_data[1,1]/init_N)
        #print("\n")
        function cost_fn(birth_rate::Float64)
            params_to_set = (birth_rates=[birth_rate], death_rates=[death_rate])
            to_return = simulate_cost(pop_template, [init_N], (bd_evosim_growth, poisson_passage, poisson_sample, euclidean_cost), ([start_day, end_day], [start_cells, start_cells], [0, end_sample], [start_cells, end_cells]), true_data, params_to_set)
            return(to_return)
        end
        result_smc = smc(prior, cost_fn, nparticles=1000, parallel=false)
        MAP_growth, lower_CI_growth, upper_CI_growth = compute_mode_CI(result_smc[1][1].particles, 0.95)
        #lock(results_lock) do
        #    push!(results_table, [BC_data[i,"BC"], start_time, MAP_growth, lower_CI_growth, upper_CI_growth])
        #end
        push!(results_table, [BC_data[i,"BC"], start_time, MAP_growth, lower_CI_growth, upper_CI_growth])
    end

    CSV.write(out_file, results_table)
    return nothing
end

function run_gillespie_adjtimes(BC_file::String, sampling_file::String, out_file::String, death_rate::Float64, verbose::Bool=False, prior_lim::Float64=0.0)
    BC_data = CSV.read(BC_file, DataFrame)
    sampling_data = CSV.read(sampling_file, DataFrame)

    results_table = DataFrame(BC=Vector{Int64}(), time=Vector{Int64}(), MAP=Vector{Float64}(), lower_CI=Vector{Float64}(), upper_CI=Vector{Float64}())

    sample_dict = Dict{Int64, Int64}(sampling_data[!, "day"] .=> sampling_data[!, "sampled"])
    for i in 1:length(BC_data[:, "start_num"])
        if verbose
            print("\n")
            print(i)
        end

        start_time = BC_data[i,"time"]
        end_time = BC_data[i,"time"] + 1

        if ! (start_time in sampling_data[!, "time"]) & (end_time in sampling_data[!, "time"])
            continue
        end

        sampling_row = subset(sampling_data, :time => t -> t .== end_time)
        init_row = subset(sampling_data, :time => t -> t .== start_time)
        start_cells = init_row[1, "passaged"]
        end_cells = sampling_row[1, "counted"]
        start_day = init_row[1, "day"]
        end_day = sampling_row[1, "day"]

        mean_growth = log(end_cells/start_cells)/(end_day-start_day)

        if prior_lim > 0.0
            prior = Uniform(0,prior_lim)
        else
            prior = Uniform(0,mean_growth*3)
        end
        pop_template = BirthDeathGillespie([0], [0.0], [0.0])

        end_sample = sampling_row[1, "sampled"]
        true_data = Matrix{Number}(undef, 1, 1)
        true_data[1,1] = BC_data[i,"end_num"]

        init_N = round(Int, start_cells*BC_data[i,"start_num"]/init_row[1, "sampled"])
        #print(true_data[1,1]/init_N)
        #print("\n")
        function cost_fn(birth_rate::Float64)
            params_to_set = (birth_rates=[birth_rate], death_rates=[death_rate])
            to_return = simulate_cost(pop_template, [init_N], (bd_gillespie_growth, poisson_passage, poisson_sample, euclidean_cost), ([start_day, end_day], [start_cells, start_cells], [0, end_sample], [start_cells, end_cells]), true_data, params_to_set)
            return(to_return)
        end
        result_smc = smc(prior, cost_fn, nparticles=1000, parallel=true)
        MAP_growth, lower_CI_growth, upper_CI_growth = compute_mode_CI(result_smc[1][1].particles, 0.95)
        push!(results_table, [BC_data[i,"BC"], start_time, MAP_growth, lower_CI_growth, upper_CI_growth])
    end

    CSV.write(out_file, results_table)
    return nothing
end

function run_lazyhybrid_adjtimes(BC_file::String, sampling_file::String, out_file::String, death_rate::Float64, verbose::Bool=False, prior_lim::Float64=0.0)
    BC_data = CSV.read(BC_file, DataFrame)
    sampling_data = CSV.read(sampling_file, DataFrame)

    results_table = DataFrame(BC=Vector{Int64}(), time=Vector{Int64}(), MAP=Vector{Float64}(), lower_CI=Vector{Float64}(), upper_CI=Vector{Float64}())

    sample_dict = Dict{Int64, Int64}(sampling_data[!, "day"] .=> sampling_data[!, "sampled"])
    for i in 1:length(BC_data[:, "start_num"])
        if verbose
            print("\n")
            print(i)
        end

        start_time = BC_data[i,"time"]
        end_time = BC_data[i,"time"] + 1

        if ! (start_time in sampling_data[!, "time"]) & (end_time in sampling_data[!, "time"])
            continue
        end

        sampling_row = subset(sampling_data, :time => t -> t .== end_time)
        init_row = subset(sampling_data, :time => t -> t .== start_time)
        start_cells = init_row[1, "passaged"]
        end_cells = sampling_row[1, "counted"]
        start_day = init_row[1, "day"]
        end_day = sampling_row[1, "day"]

        mean_growth = log(end_cells/start_cells)/(end_day-start_day)

        if prior_lim > 0.0
            prior = Uniform(0,prior_lim)
        else
            prior = Uniform(0,mean_growth*3)
        end
        pop_template = BirthDeathLazyHybrid([0], [0.0], [0.0])

        end_sample = sampling_row[1, "sampled"]
        true_data = Matrix{Number}(undef, 1, 1)
        true_data[1,1] = BC_data[i,"end_num"]

        init_N = round(Int, start_cells*BC_data[i,"start_num"]/init_row[1, "sampled"])
        #print(true_data[1,1]/init_N)
        #print("\n")
        function cost_fn(birth_rate::Float64)
            params_to_set = (birth_rates=[birth_rate], death_rates=[death_rate])
            to_return = simulate_cost(pop_template, [init_N], (bd_lazyhybrid_growth, poisson_passage, poisson_sample, euclidean_cost), ([start_day, end_day], [start_cells, start_cells], [0, end_sample], [start_cells, end_cells]), true_data, params_to_set)
            return(to_return)
        end
        result_smc = smc(prior, cost_fn, nparticles=1000, parallel=true)
        MAP_growth, lower_CI_growth, upper_CI_growth = compute_mode_CI(result_smc[1][1].particles, 0.95)
        push!(results_table, [BC_data[i,"BC"], start_time, MAP_growth, lower_CI_growth, upper_CI_growth])
    end

    CSV.write(out_file, results_table)
    return nothing
end

function run_NegBin_adjtimes(BC_file::String, sampling_file::String, out_file::String, death_rate::Float64, verbose::Bool=False, prior_lim::Float64=0.0)
    BC_data = CSV.read(BC_file, DataFrame)
    sampling_data = CSV.read(sampling_file, DataFrame)

    results_table = DataFrame(BC=Vector{Int64}(), time=Vector{Int64}(), MAP=Vector{Float64}(), lower_CI=Vector{Float64}(), upper_CI=Vector{Float64}())

    sample_dict = Dict{Int64, Int64}(sampling_data[!, "day"] .=> sampling_data[!, "sampled"])
    for i in 1:length(BC_data[:, "start_num"])
        if verbose
            print("\n")
            print(i)
        end

        start_time = BC_data[i,"time"]
        end_time = BC_data[i,"time"] + 1

        if ! (start_time in sampling_data[!, "time"]) & (end_time in sampling_data[!, "time"])
            continue
        end

        sampling_row = subset(sampling_data, :time => t -> t .== end_time)
        init_row = subset(sampling_data, :time => t -> t .== start_time)
        start_cells = init_row[1, "passaged"]
        end_cells = sampling_row[1, "counted"]
        start_day = init_row[1, "day"]
        end_day = sampling_row[1, "day"]

        mean_growth = log(end_cells/start_cells)/(end_day-start_day)

        if prior_lim > 0.0
            prior = Uniform(0,prior_lim)
        else
            prior = Uniform(0,mean_growth*3)
        end
        pop_template = BirthDeathNegBin([0], [0.0], [0.0])

        end_sample = sampling_row[1, "sampled"]
        true_data = Matrix{Number}(undef, 1, 1)
        true_data[1,1] = BC_data[i,"end_num"]

        init_N = round(Int, start_cells*BC_data[i,"start_num"]/init_row[1, "sampled"])
        #print(true_data[1,1]/init_N)
        #print("\n")
        function cost_fn(birth_rate::Float64)
            params_to_set = (birth_rates=[birth_rate], death_rates=[death_rate])
            to_return = simulate_cost(pop_template, [init_N], (bd_NegBin_growth, poisson_passage, poisson_sample, euclidean_cost), ([start_day, end_day], [start_cells, start_cells], [0, end_sample], [start_cells, end_cells]), true_data, params_to_set)
            return(to_return)
        end
        result_smc = smc(prior, cost_fn, nparticles=1000, parallel=true)
        MAP_growth, lower_CI_growth, upper_CI_growth = compute_mode_CI(result_smc[1][1].particles, 0.95)
        push!(results_table, [BC_data[i,"BC"], start_time, MAP_growth, lower_CI_growth, upper_CI_growth])
    end

    CSV.write(out_file, results_table)
    return nothing
end
