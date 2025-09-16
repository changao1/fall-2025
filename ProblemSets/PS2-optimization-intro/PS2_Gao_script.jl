using Random, LinearAlgebra, Distributions, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, ForwardDiff

cd(@__DIR__)

# Read in the function
include("PS2_Gao_source.jl")

# Execute the function
allwrap()
