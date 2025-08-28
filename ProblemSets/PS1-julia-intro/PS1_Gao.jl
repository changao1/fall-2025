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

# Question 4: Practice with functions
function q4()
    # (a) Load firstmatrix.jld
    matrices = load("firstmatrix.jld")
    A_loaded = matrices["A"]
    B_loaded = matrices["B"]
    C_loaded = matrices["C"]
    D_loaded = matrices["D"]
    
    # (b-f) Define matrixops function
    function matrixops(mat1, mat2)
        # Check if inputs have same size
        if size(mat1) != size(mat2)
            error("inputs must have the same size")
        end
        
        # (i) Element-by-element product
        elem_product = mat1 .* mat2
        
        # (ii) Product A'B
        matrix_product = mat1' * mat2
        
        # (iii) Sum of all elements of A+B
        sum_elements = sum(mat1 + mat2)
        
        return elem_product, matrix_product, sum_elements
    end
    
    # (d) Evaluate matrixops using A and B
    result1, result2, result3 = matrixops(A_loaded, B_loaded)
    println("matrixops(A,B) completed successfully")
    
    # (f) Try with C and D (should give error due to size mismatch)
    try
        matrixops(C_loaded, D_loaded)
    catch e
        println("Error with matrixops(C,D): ", e)
    end
    
    # (g) Try with nlsw88 data
    nlsw88 = CSV.read("nlsw88_processed.csv", DataFrame)
    ttl_exp_array = convert(Array, nlsw88.ttl_exp)
    wage_array = convert(Array, nlsw88.wage)
    
    # Ensure same length
    min_length = min(length(ttl_exp_array), length(wage_array))
    ttl_exp_array = ttl_exp_array[1:min_length]
    wage_array = wage_array[1:min_length]
    
    try
        result_data = matrixops(ttl_exp_array, wage_array)
        println("matrixops with ttl_exp and wage completed successfully")
    catch e
        println("Error with matrixops(ttl_exp, wage): ", e)
    end
    
    return nothing
end

# Execute all functions
A, B, C, D = q1()
q2(A, B, C)
q3()
q4()

# Question 5: Unit tests for all functions
using Test

@testset "PS1 Unit Tests" begin
    
    @testset "Question 1 Tests" begin
        A_test, B_test, C_test, D_test = q1()
        
        @test size(A_test) == (10, 7)
        @test size(B_test) == (10, 7) 
        @test size(C_test) == (5, 7)
        @test size(D_test) == (10, 7)
        
        # Test that D only has non-positive values of A or zeros
        @test all((D_test .== 0) .| (D_test .<= 0))
        @test all(D_test[A_test .<= 0] .== A_test[A_test .<= 0])
        @test all(D_test[A_test .> 0] .== 0)
        
        println("✓ Question 1 tests passed")
    end
    
    @testset "Question 2 Tests" begin
        # Test with small matrices
        A_small = [1 2; 3 4]
        B_small = [5 6; 7 8] 
        C_small = [1 -6; 0 4]
        
        # This should not error
        @test_nowarn q2(A_small, B_small, C_small)
        
        println("✓ Question 2 tests passed")
    end
    
    @testset "Question 3 Tests" begin
        # Test that the function runs without error
        @test_nowarn q3()
        
        # Test that files were created
        @test isfile("nlsw88_processed.csv")
        
        println("✓ Question 3 tests passed") 
    end
    
    @testset "Question 4 Tests" begin
        # Test that the function runs without error
        @test_nowarn q4()
        
        # Test that files were created in Q1
        @test isfile("firstmatrix.jld")
        @test isfile("matrixpractice.jld")
        @test isfile("Cmatrix.csv")
        @test isfile("Dmatrix.dat")
        
        println("✓ Question 4 tests passed")
    end
    
end

println("All unit tests completed successfully!")
