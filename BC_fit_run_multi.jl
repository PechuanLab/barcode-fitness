using CSV
using KissABC
using DataFrames

include("/Users/dve/Documents/Curtis_Lab/organoid_ECB/fitness_inference/BC_fit_types.jl")
include("/Users/dve/Documents/Curtis_Lab/organoid_ECB/fitness_inference/BC_fit_sim_fns.jl")

time_period = 14 # days between passages

parent_dir = "/Users/dve/Documents/Curtis_Lab/organoid_ECB/ECB_input_data/test_data/two_BC/"
#sample_name = "uniform_death0"
out_file = string(parent_dir, "output/pure_birth_test/two_BC.csv")



results_table = DataFrame(trial=Vector{Int64}(), BC=Vector{Int64}(), time=Vector{Int64}(), MAP=Vector{Float64}(), lower_CI=Vector{Float64}(), upper_CI=Vector{Float64}())
for i in 1:10

    #BC_data = CSV.read(ARGS[1], DataFrame)
    BC_file = string(parent_dir, "run_", string(i), "_BCs.csv")
    BC_data = CSV.read(BC_file, DataFrame)
    #sampling_data = CSV.read(ARGS[2], DataFrame)
    sampling_file = string(parent_dir, "run_", string(i), "_sample.csv")
    sampling_data = CSV.read(sampling_file, DataFrame)

    print("\n")
    print(i)

    start_time = 1
    end_time = 2

    if ! (start_time in sampling_data[!, "time"]) & (end_time in sampling_data[!, "time"])
        continue
    end

    sampling_row = subset(sampling_data, :time => t -> t .== end_time)
    init_row = subset(sampling_data, :time => t -> t .== start_time)
    start_cells = sampling_row[1, "passaged"]
    end_cells = sampling_row[1, "counted"]

    mean_growth = log(sampling_row[1, "counted"]/sampling_row[1, "passaged"])/time_period

    prior = Factored(Uniform(0,mean_growth*3), Uniform(0,mean_growth*3))
    pop_template = PureBirth([0, 0], [0.0, 0.0])

    end_sample = sampling_row[1, "sampled"]
    true_data = Matrix{Number}(undef, 1, 2)
    true_data[1,1] = BC_data[1,"end_num"]
    true_data[1,2] = BC_data[2,"end_num"]

    init_N = [round(Int, start_cells*BC_data[1,"start_num"]/init_row[1, "sampled"]), round(Int, start_cells*BC_data[2,"start_num"]/init_row[1, "sampled"])]
    #print(true_data[1,1]/init_N)
    #print("\n")
    function cost_fn(birth_rates)
        birth_rate1, birth_rate2 = birth_rates
        params_to_set = (birth_rates=[birth_rate1, birth_rate2],)
        to_return = simulate_cost(pop_template, init_N, (pure_birth_growth, multinomial_passage, multinomial_sample, euclidean_cost), ([start_time, start_time+time_period], [start_cells, start_cells], [0, end_sample], [start_cells, end_cells]), true_data, params_to_set)
        return(to_return)
    end
    result_smc = smc(prior, cost_fn, nparticles=100, parallel=false)
    MAP_growth, lower_CI_growth, upper_CI_growth = compute_mode_CI(result_smc[1][1].particles, 0.95)
    push!(results_table, [i, BC_data[1,"BC"], start_time, MAP_growth[1], lower_CI_growth[1], upper_CI_growth[1]])
    push!(results_table, [i, BC_data[2,"BC"], start_time, MAP_growth[2], lower_CI_growth[2], upper_CI_growth[2]])
end

CSV.write(out_file, results_table)
