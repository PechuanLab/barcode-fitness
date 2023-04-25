using CSV
using KissABC
using DataFrames

include("BC_fit_types.jl")
include("BC_fit_sim_fns.jl")
include("BC_fit_sample_fns.jl")
include("BC_fit_utils.jl")
include("BC_fit_growth_fns.jl")

function run_purebirth_adjtimes(BC_file::String, sampling_file::String, out_file::String, verbose::Bool=false, prior_lim::Float64=0.0)
    BC_data = CSV.read(BC_file, DataFrame)
    sampling_data = CSV.read(sampling_file, DataFrame)

    results_table = DataFrame(BC=Vector{Int64}(), time=Vector{Int64}(), MAP=Vector{Float64}(), lower_CI=Vector{Float64}(), upper_CI=Vector{Float64}())
    for i in 1:length(BC_data[:, "start_num"])
        print("\n")
        print(i)
        if verbose
            print("---")
            print("\n")
            print(BC_data[i,"BC"])
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

function run_purebirth_whole(BC_file::String, sampling_file::String, out_file::String, prior_lim::Float64=0.0, extended::Bool=false)
    BC_data = CSV.read(BC_file, DataFrame)
    sampling_data = CSV.read(sampling_file, DataFrame)

    start_time = sampling_data[1,"time"]
    end_time = sampling_data[length(sampling_data[!, "time"]),"time"]

    sampling_row = subset(sampling_data, :time => t -> t .== end_time)
    init_row = subset(sampling_data, :time => t -> t .== start_time)
    start_cells = init_row[1, "passaged"]
    end_cells = sampling_row[1, "counted"]
    start_day = init_row[1, "day"]
    end_day = sampling_row[1, "day"]
    mean_growth = log(end_cells/start_cells)/(end_day-start_day)

    results_table = DataFrame(BC=Vector{String}(), time=Vector{Int64}(), MAP=Vector{Float64}(), lower_CI=Vector{Float64}(), upper_CI=Vector{Float64}(), log_like=Vector{Float64}())
    for i in 2:length(BC_data[1, :])
        print("\n")
        print(names(BC_data)[i])



        if ! (start_time in sampling_data[!, "time"]) & (end_time in sampling_data[!, "time"])
            continue
        end

        init_N = BC_data[1,i]

        if BC_data[2,i] < 10
            continue
        end

        num_passages = length(BC_data[!,"time"])
        sampled_length = length(sampling_data[!, "sampled"])
        if extended
            true_data = BC_data[1:num_passages,i] #./ sampling_data[(sampled_length-num_passages+1):sampled_length,"sampled"]
            init_N = 15
        else
            true_data = BC_data[2:num_passages,i] #./ sampling_data[2:num_passages]
            init_N = round(Int, start_cells*init_N/init_row[1, "sampled"])
        end
        true_data = reshape(true_data, :, 1)
        print("\n")
        print(true_data)

        if prior_lim > 0.0
            prior = Uniform(0,prior_lim)
        else
            prior = Uniform(0,mean_growth*3)
        end
        pop_template = PureBirth([0], [0.0])

        sampling_data[1,"sampled"] = 0
        function cost_fn(birth_rate::Float64)
            params_to_set = (birth_rates=[birth_rate],)
            to_return = simulate_cost(pop_template, [init_N], (pure_birth_growth, poisson_passage, poisson_sample, weighted_euclidean_cost), (sampling_data[!,"day"], Int.(sampling_data[!,"passaged"]), Int.(sampling_data[!,"sampled"]), Int.(sampling_data[!,"counted"])), true_data, params_to_set)
            return(to_return)
        end
        result_smc = smc(prior, cost_fn, nparticles=1000, parallel=true)
        MAP_growth, lower_CI_growth, upper_CI_growth, log_like = compute_mode_CI_ll(result_smc[1][1].particles, 0.95, prior)
        push!(results_table, [names(BC_data)[i], start_time, MAP_growth, lower_CI_growth, upper_CI_growth, log_like])
        print("\n")
        print([names(BC_data)[i], start_time, MAP_growth, lower_CI_growth, upper_CI_growth, log_like])
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

function run_onemut_whole(BC_file::String, sampling_file::String, out_file::String, prior_lim::Float64=0.35, extended::Bool=false)
    BC_data = CSV.read(BC_file, DataFrame)
    sampling_data = CSV.read(sampling_file, DataFrame)

    start_time = sampling_data[1,"time"]
    end_time = sampling_data[length(sampling_data[!, "time"]),"time"]

    sampling_row = subset(sampling_data, :time => t -> t .== end_time)
    init_row = subset(sampling_data, :time => t -> t .== start_time)
    start_cells = init_row[1, "passaged"]
    end_cells = sampling_row[1, "counted"]
    start_day = init_row[1, "day"]
    end_day = sampling_row[1, "day"]

    results_table = DataFrame(BC=Vector{String}(), time=Vector{Int64}(), r0_MAP=Vector{Float64}(), r1_MAP=Vector{Float64}(), t1_MAP=Vector{Float64}(), r0_lower_CI=Vector{Float64}(), r1_lower_CI=Vector{Float64}(), t1_lower_CI=Vector{Float64}(), r0_upper_CI=Vector{Float64}(), r1_upper_CI=Vector{Float64}(), t1_upper_CI=Vector{Float64}(), log_like=Vector{Float64}())
    for i in 2:length(BC_data[1, :])
        print("\n")
        print(names(BC_data)[i])



        if ! (start_time in sampling_data[!, "time"]) & (end_time in sampling_data[!, "time"])
            continue
        end

        init_N = BC_data[1,i]

        if BC_data[2,i] < 10
            continue
        end

        if extended
            true_data = BC_data[1:length(BC_data[!,"time"]),i]
        else
            true_data = BC_data[2:length(BC_data[!,"time"]),i]
        end
        true_data = reshape(true_data, :, 1)
        print("\n")
        print(true_data)

        last_alive = end_day
        for i=length(true_data):-1:1
            if true_data[i]==0
                last_alive = sampling_data[i+1, "day"]
            else
                break
            end
        end

        prior = product_distribution([Uniform(0,prior_lim), Uniform(0,prior_lim), Uniform(start_day,last_alive)])
        #print(prior.v)

        #=
        N_WT::Vector{Int64}
        N_mut::Vector{Int64}
        birth_rates_mut::Vector{Float64}
        birth_rates_WT::Vector{Float64}
        death_rates_mut::Vector{Float64}
        death_rates_WT::Vector{Float64}
        mut_times::Vector{Float64}
        =#
        pop_template = BirthDeathOneMut([0], [0], [0.0], [0.0], [0.0], [0.0], [0.0])

        #print("\n")
        #print(start_cells)
        #print(init_row[1, "sampled"])
        if extended
            init_N = 15
        else
            init_N = round(Int, start_cells*init_N/init_row[1, "sampled"])
        end
        #print(true_data[1,1]/init_N)
        #print("\n")
        sampling_data[1,"sampled"] = 0
        function cost_fn(prior_sample)
            params_to_set = (birth_rates_WT=[prior_sample[1]], birth_rates_mut=[prior_sample[2]], mut_times=[prior_sample[3]], death_rates_WT=[0.], death_rates_mut=[0.])
            to_return = simulate_cost(pop_template, [init_N], (bd_onemut_growth, poisson_passage, poisson_sample, weighted_euclidean_cost), (sampling_data[!,"day"], Int.(sampling_data[!,"passaged"]), Int.(sampling_data[!,"sampled"]), Int.(sampling_data[!,"counted"])), true_data, params_to_set)
            return(to_return)
        end
        result_smc = smc(prior, cost_fn, nparticles=1000, parallel=true)
        print("\n")
        print(result_smc)
        result_smc = [result_smc[1][1].particles, result_smc[1][2].particles, result_smc[1][3].particles]
        result_smc = copy(transpose(reduce(hcat, result_smc)))
        #return result_smc
        MAP_growth, lower_CI_growth, upper_CI_growth, log_like = compute_mode_CI_multi(result_smc, 0.95, prior)
        push!(results_table, [names(BC_data)[i], start_time, MAP_growth[1], MAP_growth[2], MAP_growth[3], lower_CI_growth[1], lower_CI_growth[2], lower_CI_growth[3], upper_CI_growth[1], upper_CI_growth[2], upper_CI_growth[3], log_like])
        print("\n")
        print([names(BC_data)[i], start_time, MAP_growth, lower_CI_growth, upper_CI_growth, log_like])
    end

    CSV.write(out_file, results_table)
    return nothing
end

function run_CNA_whole(CNA_file::String, clone_file::String, sampling_file::String, out_file::String; WT_growth::Float64=0.1, prior_lim::Float64=0.3, npart::Int64=200)
    CNA_data = CSV.read(CNA_file, DataFrame)
    clone_data = CSV.read(clone_file, DataFrame)
    sampling_data = CSV.read(sampling_file, DataFrame)

    #print(clone_data)
    #print(CNA_data)
    n_clones = size(clone_data)[1]

    start_time = sampling_data[1,"time"]
    end_time = sampling_data[length(sampling_data[!, "time"]),"time"]

    init_row = subset(sampling_data, :time => t -> t .== start_time)
    start_cells = init_row[1, "passaged"]
    start_day = init_row[1, "day"]

    #true_data = Matrix(CNA_data)[2:size(CNA_data)[1],:]
    true_data = Matrix(CNA_data)
    clone_def = Matrix(clone_data)[:,3:(size(CNA_data)[2]+2)]
    first_freqs = clone_data[!,"first_freqs"]
    first_times = clone_data[!,"first_times"]
    prior = product_distribution([Uniform(0,prior_lim) for x in 1:(n_clones-1)])

    pop_template = SubcloneCNA([0], [0.0], [0.0], [0], [0.0], copy(clone_def))

    init_N = zeros(n_clones)
    init_mask = first_times .== start_day
    init_N[(1:n_clones)[init_mask]] .= first_freqs[init_mask]
    init_N = Int.(round.(init_N .* start_cells))
    #print(init_N)
    function cost_fn(prior_sample)
        params_to_set = (birth_rates=[[0.0]; prior_sample] .+ WT_growth, death_rates=[0. for x in 1:(n_clones)], first_times=first_times, first_freqs=first_freqs, CNA_clones=clone_def)
        to_return = simulate_cost(pop_template, init_N, (bd_insertclone_growth, multinomial_passage, CNA_sample, euclidean_cost), (sampling_data[!,"day"], Int.(sampling_data[!,"passaged"]), Int.(sampling_data[!,"sampled"]), Int.(sampling_data[!,"counted"])), true_data, params_to_set)
        return(to_return)
    end
    result_smc = smc(prior, cost_fn, nparticles=npart, parallel=true)
    result_smc = [result_smc[1][x].particles for x in 1:(n_clones-1)]
    result_smc = copy(transpose(reduce(hcat, result_smc)))
    MAP_growth, lower_CI_growth, upper_CI_growth, log_like = compute_mode_CI_multi(result_smc, 0.95, prior)

    results_table = [MAP_growth; lower_CI_growth; upper_CI_growth; log_like]
    col_labels = [[string("MAP_", x) for x in 1:(n_clones-1)]; [string("lower_", x) for x in 1:(n_clones-1)]; [string("upper_", x) for x in 1:(n_clones-1)]; ["log_like"]]
    results_table = DataFrame([[x] for x in results_table], col_labels)
    CSV.write(out_file, results_table)
    return nothing
end
