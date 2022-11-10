using CSV
using KissABC
using DataFrames

include("BC_fit_types.jl")
include("BC_fit_sim_fns.jl")

parent_dir = "/Users/dve/Documents/Curtis_Lab/organoid_ECB/ECB_input_data/"
sample_name = "PC1_R3"
out_file = string(parent_dir, "output/smoothed_birth_SMC/", sample_name, ".csv")

time_period = 14 # days between passages

#BC_data = CSV.read(ARGS[1], DataFrame)
BC_file = string(parent_dir, "trimmed_long_data/above_100/", sample_name, ".csv")
BC_data = CSV.read(BC_file, DataFrame)
#sampling_data = CSV.read(ARGS[2], DataFrame)
sampling_file = string(parent_dir, "sampling_params/", sample_name, "_smoothed.csv")
sampling_data = CSV.read(sampling_file, DataFrame)

results_table = DataFrame(BC=Vector{Int64}(), time=Vector{Int64}(), MAP=Vector{Float64}(), lower_CI=Vector{Float64}(), upper_CI=Vector{Float64}())
for i in 1:length(BC_data[:, "start_num"])
    print("\n")
    print(i)

    start_time = BC_data[i,"time"]
    end_time = BC_data[i,"time"] + 1

    if ! (start_time in sampling_data[!, "time"]) & (end_time in sampling_data[!, "time"])
        continue
    end

    sampling_row = subset(sampling_data, :time => t -> t .== end_time)
    init_row = subset(sampling_data, :time => t -> t .== start_time)
    start_cells = init_row[1, "passaged"]
    end_cells = sampling_row[1, "counted"]

    mean_growth = log(end_cells/start_cells)/time_period

    prior = Uniform(0,mean_growth*3)
    pop_template = PureBirth([0], [0.0])

    end_sample = sampling_row[1, "sampled"]
    true_data = Matrix{Number}(undef, 1, 1)
    true_data[1,1] = BC_data[i,"end_num"]

    init_N = round(Int, start_cells*BC_data[i,"start_num"]/init_row[1, "sampled"])
    #print(true_data[1,1]/init_N)
    #print("\n")
    function cost_fn(birth_rate::Float64)
        params_to_set = (birth_rates=[birth_rate],)
        to_return = simulate_cost(pop_template, [init_N], (pure_birth_growth, poisson_passage, poisson_sample, euclidean_cost), ([start_time, start_time+time_period], [start_cells, start_cells], [0, end_sample], [start_cells, end_cells]), true_data, params_to_set)
        return(to_return)
    end
    result_smc = smc(prior, cost_fn, nparticles=1000, parallel=true)
    MAP_growth, lower_CI_growth, upper_CI_growth = compute_mode_CI(result_smc[1][1].particles, 0.95)
    push!(results_table, [BC_data[i,"BC"], start_time, MAP_growth, lower_CI_growth, upper_CI_growth])
end

CSV.write(out_file, results_table)
