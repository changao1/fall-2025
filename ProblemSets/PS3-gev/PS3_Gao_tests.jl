
using Test
using Random, Optim
using DataFrames, CSV, HTTP, LinearAlgebra, Statistics

include("PS3_Gao_source.jl") 

cd(@__DIR__)

# --- Unit Tests for mlogit_with_Z ---
@testset "mlogit_with_Z Unit Tests" begin
    # Create mock data for testing
    N = 100
    K = 3
    J = 8
    Random.seed!(1234)
    X = rand(N, K)
    Z = rand(N, J)
    y = rand(1:J, N)

    # Test 1: Check output type
    @testset "Type Check" begin
        theta = rand(K * (J - 1) + 1)
        loglike = mlogit_with_Z(theta, X, Z, y)
        @test isa(loglike, Real)
        @test loglike >= 0.0
    end

    # Test 2: Verify logic with a simple, known case
    @testset "Simple Case Logic" begin
        theta = zeros(K * (J - 1) + 1)
        expected_loglike = -sum(log.(ones(N, J) ./ J))
        calculated_loglike = mlogit_with_Z(theta, X, Z, y)
        @test calculated_loglike ≈ expected_loglike atol = 1e-9
    end

    # Test 3: Check dimensions of inputs
    @testset "Input Dimensions" begin
        theta_wrong = rand(K * (J - 1))
        @test_throws BoundsError mlogit_with_Z(theta_wrong, X, Z, y)
    end
end


# --- Unit Tests for optimize_mlogit ---
@testset "optimize_mlogit Unit Tests" begin
    # Create mock data for testing
    N = 100
    K = 3
    J = 8
    Random.seed!(1234)
    X = rand(N, K)
    Z = rand(N, J)
    y = rand(1:J, N)

    # Test 1: Check if the function runs and returns a result of the correct dimension
    @testset "Function Execution and Output" begin
        result = optimize_mlogit(X, Z, y)
        expected_length = K * (J - 1) + 1
        @test length(result) == expected_length
    end
end