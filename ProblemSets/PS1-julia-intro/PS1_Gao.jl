# Problem Set 1 - ECON 6343 Econometrics III
# Chang Gao

using JLD
using Random
using LinearAlgebra
using Statistics
using CSV
using DataFrames
using FreqTables
using Distributions

Random.seed!(1234)

# Question 1: Initializing variables and practice with basic matrix operations
function q1()
    
    # (a) Create four matrices

    A = -5 .+ 15 * rand(10, 7)
    

    B = rand(Normal(-2, 15), 10, 7)
    

    C = [A[1:5, 1:5] B[1:5, 6:7]]
    

    D = A .* (A .<= 0)
    
    # (b) Number of elements of A
    println("Number of elements of A: ", length(A))
    
    # (c) Number of unique elements of D
    println("Number of unique elements of D: ", length(unique(D)))
    
    # (d) Create E using reshape (vec operator applied to B)
    E = reshape(B, :, 1) 
    
    # (e) Create 3-dimensional array F containing A and B
    F = cat(A, B, dims=3)
    
    # (f) Use permutedims to make F₂ₓ₁₀ₓ₇ instead of F₁₀ₓ₇ₓ₂
    F = permutedims(F, [3, 1, 2])
    
    # (g) Create matrix G = B⊗C (Kronecker product)
    G = kron(B, C)
    # Try C⊗F - this should give an error due to dimension mismatch
    try
        kron(C, F)
    catch e
        println("Error with C⊗F: ", e)
    end
    
    # (h) Save matrices A,B,C,D,E,F,G as matrixpractice.jld
    save("matrixpractice.jld", "A", A, "B", B, "C", C, "D", D, "E", E, "F", F, "G", G)
    
    # (i) Save only A,B,C,D as firstmatrix.jld
    save("firstmatrix.jld", "A", A, "B", B, "C", C, "D", D)
    
    # (j) Export C as CSV file
    C_df = DataFrame(C, :auto)
    CSV.write("Cmatrix.csv", C_df)
    
    # (k) Export D as tab-delimited .dat file
    D_df = DataFrame(D, :auto)
    CSV.write("Dmatrix.dat", D_df, delim='\t')
    
    return A, B, C, D
end

# Question 2: Practice with loops and comprehensions
function q2(A, B, C)
    # (a) Element-by-element product of A and B
    AB = [A[i,j] * B[i,j] for i in 1:size(A,1), j in 1:size(A,2)]
    AB2 = A .* B
    
    # (b) Create Cprime with elements of C between -5 and 5
    Cprime = Float64[]
    for val in C
        if -5 <= val <= 5
            push!(Cprime, val)
        end
    end
    Cprime2 = C[(-5 .<= C) .& (C .<= 5)]
    
    # (c) Create 3-dimensional array X (N=15169, K=6, T=5)
    N, K, T = 15169, 6, 5
    X = zeros(N, K, T)
    
    for t in 1:T
        X[:, 1, t] .= 1  # intercept
        X[:, 2, t] = rand(N) .< (0.75 * (6-t) / 5)  # dummy variable
        X[:, 3, t] = rand(Normal(15 + t - 1, 5 * (t - 1)), N)  # normal variable
        X[:, 4, t] = rand(Normal(π * (6-t) / 3, 1/ℯ), N)  # normal variable
        X[:, 5, t] = rand(Binomial(20, 0.6), N)  # discrete normal (binomial)
        X[:, 6, t] = rand(Binomial(20, 0.5), N)  # binomial
    end
    
    # (d) Create matrix β (K×T)
    β = zeros(K, T)
    for t in 1:T
        β[1, t] = 1 + 0.25 * (t - 1)  # 1, 1.25, 1.5, ...
        β[2, t] = log(t)
        β[3, t] = -sqrt(t)
        β[4, t] = ℯ^t - ℯ^(t+1)
        β[5, t] = t
        β[6, t] = t / 3
    end
    
    # (e) Create matrix Y (N×T)
    Y = zeros(N, T)
    for t in 1:T
        ε = rand(Normal(0, 0.36), N)
        Y[:, t] = X[:, :, t] * β[:, t] + ε
    end
    
    return nothing
end


# Question 3: Reading in Data and calculating summary statistics
function q3()
    # (a) Import nlsw88.csv
    nlsw88 = CSV.read("nlsw88.csv", DataFrame)
    
    # Process missing values and save as processed file
    CSV.write("nlsw88_processed.csv", nlsw88)
    
    # (b) Percentage never married and college graduates
    never_married_pct = sum(nlsw88.never_married .== 1) / nrow(nlsw88) * 100
    college_grad_pct = sum(nlsw88.collgrad .== 1) / nrow(nlsw88) * 100
    println("Never married: $(round(never_married_pct, digits=2))%")
    println("College graduates: $(round(college_grad_pct, digits=2))%")
    
    # (c) Race categories using freqtable
    race_freq = freqtable(nlsw88.race)
    println("Race distribution:")
    println(race_freq)
    
    # (d) Summary statistics using describe
    summarystats = describe(nlsw88)
    println(summarystats)
    grade_missing = sum(ismissing.(nlsw88.grade))
    println("Missing grade observations: $grade_missing")
    
    # (e) Joint distribution of industry and occupation
    joint_dist = freqtable(nlsw88.industry, nlsw88.occupation)
    println("Industry × Occupation cross-tabulation:")
    println(joint_dist)
    
    # (f) Mean wage over industry and occupation
    wage_data = select(nlsw88, :industry, :occupation, :wage)
    wage_data = dropmissing(wage_data)
    wage_means = combine(groupby(wage_data, [:industry, :occupation]), :wage => mean)
    println("Mean wages by industry and occupation:")
    println(wage_means)
    
    return nothing
end


# Question 4: Writing functions and error handling



# part (b) of question 4
"""
peform three matrix operations on two input matrices A and B
1. element wise product of A and B
2. product A'B
3. sum of all elements of A+B
"""
function matrixops(A::Matrix{Float64}, B::Matrix{Float64})
    if size(A) != size(B)
        error("inputs must have the same size")
    end
    # (i) element wise product of A and B
    out1 = A .* B
    # (ii) product A'B
    out2 = A' * B
    # (iii) sum of all elements of A+B
    out3 = sum(A + B)
    return out1, out2, out3
end

function q4()
    # three ways to load the .jld file
    # load("matrixpractice.jld", "A", "B", "C", "D", "E", "F", "G")
    # @load "matrixpractice.jld" A B C D E F G
    @load "matrixpractice.jld"


    #part (d) of question 4
    matrixops(A, B)
    #part (f) of question 4
    try
        matrixops(C, D)
    catch e
        @show e
    end
    #part (g) of question 4
    nlsw88 = CSV.read("nlsw88_processed.csv", DataFrame)
    ttl_exp = reshape(convert(Array, nlsw88.ttl_exp), :, 1)
    wage = reshape(convert(Array, nlsw88.wage), :, 1)
    matrixops(ttl_exp, wage)

    return nothing
end

# Execute all functions
A, B, C, D = q1()
q2(A, B, C)
q3()
q4()

