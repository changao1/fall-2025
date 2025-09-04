# Clean Unit Tests for PS1_Gao.jl
using Test
using JLD
using Random
using LinearAlgebra
using Statistics
using CSV
using DataFrames
using FreqTables
using Distributions

# Include the main file
include("PS1_Gao.jl")

println("Starting comprehensive tests for PS1_Gao.jl")

@testset "matrixops function" begin
    A = [1.0 2.0; 3.0 4.0]
    B = [2.0 1.0; 1.0 2.0]
    
    out1, out2, out3 = matrixops(A, B)
    
    @test out1 ≈ [2.0 2.0; 3.0 8.0]
    @test out2 ≈ [1.0 2.0; 3.0 4.0]' * [2.0 1.0; 1.0 2.0]
    @test out3 ≈ 16.0
    
    # Test error handling
    C = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    @test_throws ErrorException matrixops(C, A)
end

@testset "q1 matrix operations" begin
    # Clean slate for q1
    Random.seed!(1234)
    A, B, C, D = q1()
    
    @test size(A) == (10, 7)
    @test size(B) == (10, 7) 
    @test size(C) == (5, 7)
    @test size(D) == (10, 7)
    
    @test all(-5 .<= A .<= 10)
    @test all(D .<= 0)
    @test isfile("matrixpractice.jld")
    @test isfile("Cmatrix.csv")
end

@testset "q2 loops and comprehensions" begin
    Random.seed!(1234)
    test_A = rand(3, 3)
    test_B = rand(3, 3) 
    test_C = randn(3, 3)
    
    @test q2(test_A, test_B, test_C) === nothing
    
    # Test comprehension equivalence
    AB_comp = [test_A[i,j] * test_B[i,j] for i in 1:3, j in 1:3]
    AB_elementwise = test_A .* test_B
    @test AB_comp ≈ AB_elementwise
end

@testset "q3 data analysis" begin
    if isfile("nlsw88.csv")
        @test q3() === nothing
        
        # Test with fresh data load
        nlsw88 = CSV.read("nlsw88.csv", DataFrame)
        
        # Basic structure tests
        @test nrow(nlsw88) > 0
        @test ncol(nlsw88) > 0
        
        # Test key columns exist
        @test "wage" in names(nlsw88)
        @test "race" in names(nlsw88)
        @test "never_married" in names(nlsw88)
        @test "collgrad" in names(nlsw88)
        
        # Test wage data validity
        @test all(nlsw88.wage .> 0)
        
        # Test percentage calculations
        never_married_pct = sum(nlsw88.never_married .== 1) / nrow(nlsw88) * 100
        @test 0 <= never_married_pct <= 100
        
        # Test race frequency table
        race_freq = freqtable(nlsw88.race)
        @test length(race_freq) == 3
        @test sum(race_freq) == nrow(nlsw88)
    else
        @test_skip "nlsw88.csv not found"
    end
end

@testset "q4 function loading and testing" begin
    # Ensure we have clean files
    Random.seed!(1234) 
    q1()  # Generate fresh files
    
    @test q4() === nothing
    
    # Test file existence
    @test isfile("matrixpractice.jld")
    
    # Test loading
    @load "matrixpractice.jld"
    @test @isdefined A
    @test @isdefined B
    
    # Test matrixops with loaded matrices
    result = matrixops(A, B)
    @test length(result) == 3
end

@testset "Integration test" begin
    # Full pipeline test
    Random.seed!(1234)
    
    A, B, C, D = q1()
    q2(A, B, C)
    
    if isfile("nlsw88.csv")
        q3()
        q4()
    end
    
    # Verify key files exist
    @test isfile("matrixpractice.jld")
    @test isfile("Cmatrix.csv")
    @test isfile("Dmatrix.dat")
end

println("All tests completed successfully!")