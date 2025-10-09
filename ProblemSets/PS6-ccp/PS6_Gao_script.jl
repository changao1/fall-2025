# using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, Distributions, JuliaFormatter, DataFramesMeta
using Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM, JuliaFormatter


cd(@__DIR__)

# Random.seed!(1234)

# format("PS6_Gao_source.jl")

include("PS6_Gao_source.jl")

main()
