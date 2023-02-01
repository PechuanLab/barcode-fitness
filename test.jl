
include("/Users/dve/Documents/Curtis_Lab/organoid_ECB/fitness_inference/BC_fit_run_fns.jl")

#=
parent_dir = "/Users/dve/Documents/Curtis_Lab/organoid_ECB/ECB_input_data/"

donors = ["PC3", "PC5"]
#donors = ["PC1"]
reps = ["R1","R2","R3"]

for donor=donors
  for rep=reps
    BC_file = string(parent_dir, "full_wide_data/", donor, "_", rep, ".csv")
    sampling_file = string(parent_dir, "sampling_params/",  donor, "_", rep, "_smoothed.csv")
    out_file = string("/Users/dve/Documents/Curtis_Lab/organoid_ECB/inference_outputs/constant_final/", donor, "_", rep, ".csv")
    run_purebirth_whole(BC_file, sampling_file, out_file, 0.35, false)
    out_file = string("/Users/dve/Documents/Curtis_Lab/organoid_ECB/inference_outputs/mut_final/", donor, "_", rep, ".csv")
    run_onemut_whole(BC_file, sampling_file, out_file, 0.35, false)
  end
end
=#

parent_dir = "/Users/dve/Documents/Curtis_Lab/organoid_ECB/ECB_input_data/julia_test_mut/inputs/"
out_dir = "/Users/dve/Documents/Curtis_Lab/organoid_ECB/ECB_input_data/julia_test_mut/outputs/"
time_sweep = [ 12,  32,  52,  72,  92, 112, 132, 152]
fit_sweep = [0.14, 0.16, 0.18, 0.2 , 0.22, 0.24, 0.26, 0.28, 0.3 ]
num_sims = 20

for t=time_sweep
  for mut=fit_sweep
    for i=0:(num_sims-1)
      BC_file = string(parent_dir, "data_",  mut, "_", t, "_", i, ".csv")
      sampling_file = string(parent_dir, "sampling_",  mut, "_", t, "_", i, ".csv")
      out_file = string(out_dir, "nomut_",  mut, "_", t, "_", i, ".csv")
      run_purebirth_whole(BC_file, sampling_file, out_file, 0.35, false)
      out_file = string(out_dir, "mut_",  mut, "_", t, "_", i, ".csv")
      run_onemut_whole(BC_file, sampling_file, out_file, 0.35, false)
    end
  end
end

#=
sample_name = "PC1_R1"
out_file = string("/Users/dve/Documents/Curtis_Lab/organoid_ECB/inference_outputs/constant_final/", sample_name, "_extended.csv")

BC_file = string(parent_dir, "full_wide_data/", sample_name, ".csv")
sampling_file = string(parent_dir, "sampling_params/", sample_name, "_smoothed_extended.csv")

#test_smc = run_onemut_whole(BC_file, sampling_file, out_file, 0.35, true)
run_purebirth_whole(BC_file, sampling_file, out_file, 0.35, true)

sample_name = "PC1_R3"
out_file = string("/Users/dve/Documents/Curtis_Lab/organoid_ECB/inference_outputs/constant_final/", sample_name, "_extended.csv")

BC_file = string(parent_dir, "full_wide_data/", sample_name, ".csv")
sampling_file = string(parent_dir, "sampling_params/", sample_name, "_smoothed_extended.csv")

#test_smc = run_onemut_whole(BC_file, sampling_file, out_file, 0.35, true)
run_purebirth_whole(BC_file, sampling_file, out_file, 0.35, true)





parent_dir = "/Users/dve/Documents/Curtis_Lab/organoid_ECB/ECB_input_data/"
sample_name = "PC1_R2"
out_file = string("/Users/dve/Documents/Curtis_Lab/organoid_ECB/inference_outputs/mut_final/", sample_name, "_extended.csv")

BC_file = string(parent_dir, "full_wide_data/", sample_name, ".csv")
sampling_file = string(parent_dir, "sampling_params/", sample_name, "_smoothed_extended.csv")

run_onemut_whole(BC_file, sampling_file, out_file, 0.35, true)
#run_purebirth_whole(BC_file, sampling_file, out_file, 0.35, false)

sample_name = "PC1_R1"
out_file = string("/Users/dve/Documents/Curtis_Lab/organoid_ECB/inference_outputs/mut_final/", sample_name, "_extended.csv")

BC_file = string(parent_dir, "full_wide_data/", sample_name, ".csv")
sampling_file = string(parent_dir, "sampling_params/", sample_name, "_smoothed_extended.csv")

run_onemut_whole(BC_file, sampling_file, out_file, 0.35, true)
#run_purebirth_whole(BC_file, sampling_file, out_file, 0.35, false)

sample_name = "PC1_R3"
out_file = string("/Users/dve/Documents/Curtis_Lab/organoid_ECB/inference_outputs/mut_final/", sample_name, "_extended.csv")

BC_file = string(parent_dir, "full_wide_data/", sample_name, ".csv")
sampling_file = string(parent_dir, "sampling_params/", sample_name, "_smoothed_extended.csv")

run_onemut_whole(BC_file, sampling_file, out_file, 0.35, true)
#run_purebirth_whole(BC_file, sampling_file, out_file, 0.35, false)


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
=#
