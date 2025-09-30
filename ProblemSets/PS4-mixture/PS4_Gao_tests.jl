# it keeps running but never ends
using Test
using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, Distributions

include("PS4_Gao_source.jl")

@testset "PS4 Mixed Logit Tests" begin
    # Set seed for reproducibility
    Random.seed!(1234)
    
    # Load data once for all tests
    df, X, Z, y = load_data()
    
    @testset "Data Loading" begin
        @test size(X, 1) > 0
        @test size(X, 2) == 3
        @test size(Z, 2) == 8
        @test length(y) == size(X, 1)
        @test all(y .>= 1) && all(y .<= 8)
    end
    
    @testset "Multinomial Logit Function" begin
        # Test with simple parameters
        K = size(X, 2)
        J = length(unique(y))
        theta_test = [zeros(K*(J-1)); 0.0]
        
        # Should return a positive number (negative log-likelihood)
        ll = mlogit_with_Z(theta_test, X, Z, y)
        @test ll > 0
        @test !isnan(ll)
        @test !isinf(ll)
        
        # Test with random parameters
        theta_random = randn(K*(J-1) + 1)
        ll_random = mlogit_with_Z(theta_random, X, Z, y)
        @test ll_random > 0
        @test !isnan(ll_random)
    end
    
    @testset "Quadrature Integration" begin
        # Test lgwt function
        nodes, weights = lgwt(7, -4, 4)
        @test length(nodes) == 7
        @test length(weights) == 7
        @test isapprox(sum(weights), 8.0, atol=1e-10)  # integral of 1 over [-4,4]
        
        # Test standard normal integration
        d = Normal(0, 1)
        integral = sum(weights .* pdf.(d, nodes))
        @test isapprox(integral, 1.0, atol=0.01)
        
        # Test mean computation
        mean_val = sum(weights .* nodes .* pdf.(d, nodes))
        @test isapprox(mean_val, 0.0, atol=0.01)
    end
    
    @testset "Variance Quadrature" begin
        # Test variance computation for N(0,2)
        σ = 2
        d = Normal(0, σ)
        nodes, weights = lgwt(10, -5*σ, 5*σ)
        variance = sum(weights .* (nodes .^ 2) .* pdf.(d, nodes))
        @test isapprox(variance, σ^2, rtol=0.01)
    end
    
    @testset "Monte Carlo Integration" begin
        σ = 2
        d = Normal(0, σ)
        a, b = -5*σ, 5*σ
        
        # Define MC integration function
        mc_integrate = function(f, a, b, D)
            draws = rand(D) * (b - a) .+ a
            return (b - a) * mean(f.(draws))
        end
        
        # Test with large D
        D = 100_000
        
        # Test density integral
        density_mc = mc_integrate(x -> pdf(d, x), a, b, D)
        @test isapprox(density_mc, 1.0, atol=0.05)
        
        # Test mean
        mean_mc = mc_integrate(x -> x * pdf(d, x), a, b, D)
        @test isapprox(mean_mc, 0.0, atol=0.05)
        
        # Test variance
        var_mc = mc_integrate(x -> x^2 * pdf(d, x), a, b, D)
        @test isapprox(var_mc, σ^2, rtol=0.05)
    end
    
    @testset "Mixed Logit Quadrature" begin
        # Test with small problem
        K = size(X, 2)
        J = length(unique(y))
        theta_test = [zeros(K*(J-1)); 0.0; 1.0]  # mu=0, sigma=1
        
        # Should return a positive log-likelihood
        ll = mixed_logit_quad(theta_test, X, Z, y, 5)
        @test ll > 0
        @test !isnan(ll)
        @test !isinf(ll)
    end
    
    @testset "Mixed Logit Monte Carlo" begin
        # Test with small problem
        K = size(X, 2)
        J = length(unique(y))
        theta_test = [zeros(K*(J-1)); 0.0; 1.0]
        
        # Should return a positive log-likelihood
        ll = mixed_logit_mc(theta_test, X, Z, y, 100)
        @test ll > 0
        @test !isnan(ll)
        @test !isinf(ll)
    end
    
    @testset "Optimization Setup" begin
        # Test that optimization functions return correct dimensions
        K = size(X, 2)
        J = length(unique(y))
        
        # Test starting values
        startvals_mlogit = [2*rand(K*(J-1)) .- 1; 0.1]
        @test length(startvals_mlogit) == K*(J-1) + 1
        
        startvals_mixed = [2*rand(K*(J-1)) .- 1; 0.1; 1.0]
        @test length(startvals_mixed) == K*(J-1) + 2
    end
    
    @testset "Probability Constraints" begin
        # Test that probabilities sum to 1
        K = size(X, 2)
        J = length(unique(y))
        N = length(y)
        
        # Create test parameters
        alpha = randn(K*(J-1))
        gamma = 0.5
        bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
        
        # Compute probabilities manually
        num = zeros(N, J)
        for j = 1:J
            num[:, j] = exp.(X * bigAlpha[:, j] .+ gamma .* (Z[:, j] .- Z[:, J]))
        end
        dem = sum(num, dims = 2)
        P = num ./ dem
        
        # Test that each row sums to 1
        row_sums = sum(P, dims=2)
        @test all(isapprox.(row_sums, 1.0, atol=1e-10))
        
        # Test that all probabilities are between 0 and 1
        @test all(P .>= 0)
        @test all(P .<= 1)
    end
    
    @testset "Edge Cases" begin
        # Test with extreme parameter values
        K = size(X, 2)
        J = length(unique(y))
        
        # Very large parameters (should still be finite)
        theta_large = [10 * ones(K*(J-1)); 10.0]
        ll_large = mlogit_with_Z(theta_large, X, Z, y)
        @test isfinite(ll_large)
        
        # Very small gamma
        theta_small_gamma = [zeros(K*(J-1)); 1e-10]
        ll_small = mlogit_with_Z(theta_small_gamma, X, Z, y)
        @test isfinite(ll_small)
    end
end

# Run all tests
println("\n=== Running PS4 Tests ===")
@time @testset "All PS4 Tests" begin
    include("PS4_Gao_tests.jl")
end
