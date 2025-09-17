using Random, LinearAlgebra, Distributions, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, JuliaFormatter

cd(@__DIR__)

format("PS2_Gao_source.jl")
# Read in the function
include("PS2_Gao_source.jl")

# Execute the function
allwrap()
# The outcome is the same as Stata.