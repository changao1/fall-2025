
# using Test, Random, Optim, DataFrames, CSV, HTTP, LinearAlgebra, Statistics

include("PS3_Gao_source.jl") 

cd(@__DIR__)

# --- Unit Tests for load_data ---
@testset "load_data Unit Tests" begin
    @testset "Data Loading" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS3-gev/nlsw88w.csv"
        X, Z, y = load_data(url)
        
        @test isa(X, Matrix)
        @test isa(Z, Matrix) 
        @test isa(y, Vector)
        @test size(X, 2) == 3  # age, white, collgrad
        @test size(Z, 2) == 8  # 8 wage alternatives
        @test size(X, 1) == size(Z, 1) == length(y)  # same number of observations
        @test all(y .>= 1) && all(y .<= 8)  # occupation choices 1-8
    end
end

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

    # Test 1: Check output type and positivity
    @testset "Type and Positivity Check" begin
        theta = rand(K * (J - 1) + 1)
        loglike = mlogit_with_Z(theta, X, Z, y)
        @test isa(loglike, Real)
        @test loglike >= 0.0
        @test isfinite(loglike)
    end

    # Test 2: Verify logic with zero parameters
    @testset "Zero Parameters Logic" begin
        theta = zeros(K * (J - 1) + 1)
        loglike = mlogit_with_Z(theta, X, Z, y)
        @test isfinite(loglike)
        @test loglike > 0.0
    end

    # Test 3: Check parameter dimensions
    @testset "Parameter Dimensions" begin
        theta_correct = rand(K * (J - 1) + 1)  # 22 parameters
        @test length(theta_correct) == 22
        
        theta_wrong = rand(K * (J - 1))  # Missing gamma
        @test_throws BoundsError mlogit_with_Z(theta_wrong, X, Z, y)
    end

    # Test 4: Mathematical consistency
    @testset "Mathematical Properties" begin
        theta = 0.1 * randn(K * (J - 1) + 1)
        loglike1 = mlogit_with_Z(theta, X, Z, y)
        loglike2 = mlogit_with_Z(theta, X, Z, y)
        
        # Function should be deterministic
        @test loglike1 == loglike2
        
        # Different parameter values should generally give different likelihoods
        theta2 = theta .+ 0.1
        loglike3 = mlogit_with_Z(theta2, X, Z, y)
        @test loglike1 != loglike3
    end
end

# --- Unit Tests for nested_logit_with_Z ---
@testset "nested_logit_with_Z Unit Tests" begin
    N = 50  # Smaller sample for faster testing
    K = 3
    J = 8
    Random.seed!(1234)
    X = rand(N, K)
    Z = rand(N, J)
    y = rand(1:J, N)
    nesting_structure = [[1, 2, 3], [4, 5, 6, 7]]

    @testset "Type and Positivity Check" begin
        theta = [rand(2*K); 1.0; 1.0; 0.1]  # 6 alphas + 2 lambdas + 1 gamma
        loglike = nested_logit_with_Z(theta, X, Z, y, nesting_structure)
        @test isa(loglike, Real)
        @test loglike >= 0.0
        @test isfinite(loglike)
    end

    @testset "Parameter Dimensions" begin
        theta_correct = [rand(2*K); 1.0; 1.0; 0.1]  # 9 parameters total
        @test length(theta_correct) == 9
        
        theta_wrong = rand(2*K)  # Missing lambdas and gamma
        @test_throws BoundsError nested_logit_with_Z(theta_wrong, X, Z, y, nesting_structure)
    end

    @testset "Lambda Parameters" begin
        # Test with positive lambda values
        theta = [0.1*randn(2*K); 0.8; 1.2; 0.1]
        loglike = nested_logit_with_Z(theta, X, Z, y, nesting_structure)
        @test isfinite(loglike)
        @test loglike >= 0.0
    end
end

# --- Unit Tests for optimize_mlogit ---
@testset "optimize_mlogit Unit Tests" begin
    # Use smaller dataset for faster optimization
    N = 50
    K = 3
    J = 8
    Random.seed!(1234)
    X = rand(N, K)
    Z = rand(N, J)
    y = rand(1:J, N)

    @testset "Optimization Output" begin
        result = optimize_mlogit(X, Z, y)
        expected_length = K * (J - 1) + 1  # 22 parameters
        @test length(result) == expected_length
        @test all(isfinite.(result))
    end

    @testset "Optimization Consistency" begin
        # Run optimization twice with same seed - should get similar results
        Random.seed!(5678)
        result1 = optimize_mlogit(X, Z, y)
        Random.seed!(5678)
        result2 = optimize_mlogit(X, Z, y)
        @test norm(result1 - result2) < 1e-6
    end
end

# --- Unit Tests for optimize_nested_logit ---
@testset "optimize_nested_logit Unit Tests" begin
    N = 30  # Small dataset for faster testing
    K = 3
    J = 8
    Random.seed!(1234)
    X = rand(N, K)
    Z = rand(N, J)
    y = rand(1:J, N)
    nesting_structure = [[1, 2, 3], [4, 5, 6, 7]]

    @testset "Optimization Output" begin
        result = optimize_nested_logit(X, Z, y, nesting_structure)
        expected_length = 2*K + 2 + 1  # 6 alphas + 2 lambdas + 1 gamma = 9
        @test length(result) == expected_length
        @test all(isfinite.(result))
    end

    @testset "Lambda Constraints" begin
        result = optimize_nested_logit(X, Z, y, nesting_structure)
        lambda_WC = result[end-2]
        lambda_BC = result[end-1]
        # Lambda values should be positive for identification
        @test lambda_WC > 0.0
        @test lambda_BC > 0.0
    end
end

# --- Integration Tests ---
@testset "Integration Tests" begin
    @testset "Data Loading and Model Estimation" begin
        # Test the full pipeline with real data (but smaller sample)
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS3-gev/nlsw88w.csv"
        X, Z, y = load_data(url)
        
        # Take a small subsample for testing
        n_test = min(100, size(X, 1))
        X_test = X[1:n_test, :]
        Z_test = Z[1:n_test, :]
        y_test = y[1:n_test]
        
        # Test multinomial logit
        theta_mle = optimize_mlogit(X_test, Z_test, y_test)
        @test length(theta_mle) == 22
        
        # Test that we can evaluate likelihood at estimated parameters
        loglike = mlogit_with_Z(theta_mle, X_test, Z_test, y_test)
        @test isfinite(loglike)
        @test loglike >= 0.0
    end

    @testset "Model Comparison" begin
        # Small synthetic dataset
        Random.seed!(9999)
        N = 50
        K = 3
        J = 8
        X = rand(N, K)
        Z = rand(N, J)
        y = rand(1:J, N)
        nesting_structure = [[1, 2, 3], [4, 5, 6, 7]]
        
        # Estimate both models
        theta_mle = optimize_mlogit(X, Z, y)
        theta_nested = optimize_nested_logit(X, Z, y, nesting_structure)
        
        # Compute log-likelihoods
        ll_mle = mlogit_with_Z(theta_mle, X, Z, y)
        ll_nested = nested_logit_with_Z(theta_nested, X, Z, y, nesting_structure)
        
        @test isfinite(ll_mle)
        @test isfinite(ll_nested)
        @test ll_mle >= 0.0
        @test ll_nested >= 0.0
    end
end

println("All unit tests completed!")
println("Run these tests using: include(\"PS3_Gao_tests.jl\")")