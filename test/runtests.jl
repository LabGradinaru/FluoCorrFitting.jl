using FCSFitting, Test, Random, LsqFit, StatsAPI

Random.seed!(42)

include("modelling.jl")
include("fitting.jl")
include("data_structures.jl")
include("naming.jl")
include("extensions.jl")
include("FCSFittingCairoMakieExt.jl")