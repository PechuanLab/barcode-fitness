############### Top Types
abstract type Population end

############### Lineage Types
# Requires a compatible reset_pop function for each Population type for inference

# Lineages without mutation
mutable struct PureBirth <: Population
    N::Vector{Int64}
    birth_rates::Vector{Float64}
end

# currently useless
mutable struct BirthDeath <: Population
    N::Vector{Int64}
    growth_rates::Vector{Float64}
    variances::Vector{Float64}
end

mutable struct BirthDeathSample <: Population
    N::Vector{Int64}
    growth_rates::Vector{Float64}
    variances::Vector{Float64}
    samples::Dict{Int64, Int64}
end

mutable struct BirthDeathEvoSim <: Population
    N::Vector{Int64}
    birth_rates::Vector{Float64}
    death_rates::Vector{Float64}
end

mutable struct BirthDeathGillespie <: Population
    N::Vector{Int64}
    birth_rates::Vector{Float64}
    death_rates::Vector{Float64}
end

mutable struct BirthDeathLazyHybrid <: Population
    N::Vector{Int64}
    birth_rates::Vector{Float64}
    death_rates::Vector{Float64}
end

mutable struct BirthDeathNegBin <: Population
    N::Vector{Int64}
    birth_rates::Vector{Float64}
    death_rates::Vector{Float64}
end

# Lineages with mutation

# general strategy/equations from Sherlock (2015) paper
mutable struct BirthDeathMut <: Population
    N::Vector{Int64}
    growth_rates::Vector{Float64}
    variances::Vector{Float64}
    samples::Dict{Int64, Int64}
end

mutable struct BirthDeathOneMut <: Population
    N_WT::Vector{Int64}
    N_mut::Vector{Int64}
    birth_rates_mut::Vector{Float64}
    birth_rates_WT::Vector{Float64}
    death_rates_mut::Vector{Float64}
    death_rates_WT::Vector{Float64}
    mut_times::Vector{Float64}
end
