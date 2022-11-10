ENV["PYCALL_JL_RUNTIME_PYTHON"] = "/Users/dve/Documents/Curtis_Lab/organoid_ECB/pycall_venv/bin/python"
using PyCall
using DataFrames

es = pyimport("evosim")

function pd_to_df(df_pd)
    df= DataFrame()
    for col in df_pd.columns
        df[!, col] = getproperty(df_pd, col).values
    end
    return df
end

function sample_evosim_nomut(init_cells::Int64, birth_rate::Float64, death_rate::Float64, end_time::Float64)
    sim=es.Simulator()
    initial_subpopulations = Dict{String, Float64}("A"=>init_cells)
    sim.initialize(initial_subpopulations, carrying_capacity=Inf, new_folder=false)
    birth_rates = Dict{String, Float64}("A"=>birth_rate)
    death_rates = Dict{String, Float64}("A"=>death_rate)
    mut_probs=Dict()

    _,tumor_growth=sim.evolve(max_sim_time = end_time,recording_interval = end_time,birth_rates = birth_rates,death_rates=death_rates,mut_probs=mut_probs, to_save=false)
    tumor_growth = pd_to_df(tumor_growth)
    return last(tumor_growth[!,"A"])
end
