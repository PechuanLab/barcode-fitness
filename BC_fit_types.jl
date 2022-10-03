############### Top Types
abstract type Population end

############### Lineage Types
# Requires a compatible reset_pop function for each Population type for inference

# Lineages without mutation
mutable struct PureBirth <: Population
    N::Vector{Int64}
    birth_rates::Vector{Float64}
end

mutable struct BirthDeath <: Population
    N::Vector{Int64}
    growth_rates::Vector{Float64}
    variances::Vector{Float64}
end

mutable struct BirthDeathSample <: Population
    N::Vector{Int64}
    growth_rates::Vector{Float64}
    variances::Vector{Float64}
    samples::Dict{Float64, Integer}
end

# Lineages with mutation
