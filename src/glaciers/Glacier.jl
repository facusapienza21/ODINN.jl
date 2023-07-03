
include("Climate.jl")

@kwdef mutable struct Glacier{F <: AbstractFloat, I <: Int} 
    rgi_id::Union{String, Nothing} = nothing
    gdir::Union{PyObject, Nothing} = nothing 
    climate::Union{Climate, Nothing} = nothing
    H₀::Union{Matrix{F}, Nothing} = nothing
    S::Union{Matrix{F}, Nothing} = nothing
    B::Union{Matrix{F}, Nothing} = nothing
    V::Union{Matrix{F}, Nothing} = nothing
    slope::Union{Matrix{F}, Nothing} = nothing
    dist_border::Union{Matrix{F}, Nothing} = nothing
    S_coords::Union{PyObject, Nothing} = nothing
    Δx::Union{F, Nothing} = nothing
    Δy::Union{F, Nothing} = nothing
    nx::Union{I, Nothing} = nothing
    ny::Union{I, Nothing} = nothing

end

###############################################
################### UTILS #####################
###############################################

include("climate_utils.jl")
include("glacier_utils.jl")

