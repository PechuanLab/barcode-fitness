#=
include("/Users/dve/Documents/Curtis_Lab/organoid_ECB/fitness_inference/BC_fit_run_fns.jl")

parent_dir = "/Users/dve/Documents/Curtis_Lab/organoid_ECB/ECB_input_data/"
sample_name = "PC1_R1"
out_file = string("/Users/dve/Documents/Curtis_Lab/organoid_ECB/inference_output/birth_death_test/negbin_", sample_name, ".csv")

BC_file = string(parent_dir, "trimmed_long_data/above_100/", sample_name, ".csv")
sampling_file = string(parent_dir, "sampling_params/", sample_name, "_smoothed.csv")

run_NegBin_adjtimes(BC_file, sampling_file, out_file, 0.001, true)
=#

include("gillespie_sampler.jl")
out_file = "/Users/dve/Documents/Curtis_Lab/organoid_ECB/ECB_input_data/test_data/one_type/neg_bin_sim.txt"
num_samples = 10000

to_save = Vector{Float64}(undef, num_samples)
for i in 1:num_samples
    new_pop = sample_NegBinBD_nomut(100, 0.3, 0.1, 14.0)
    to_save[i] = new_pop
end

io = open(out_file, "w") do io
  for x in to_save
    println(io, x)
  end
end
