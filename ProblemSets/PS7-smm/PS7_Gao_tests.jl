using Test
using Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM, FreqTables, Distributions

include("PS7_Gao_source.jl")

# Define tests and run them

@testset "PS7 Econometrics Tests" begin
    
    #--------------------------------------------------------------------------
    # Test 1: Data Loading Functions
    #--------------------------------------------------------------------------
    @testset "Data Loading" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        
        @testset "load_data" begin
            df, X, y = load_data(url)
            
            @test size(df, 1) > 0  # DataFrame has rows
            @test size(X, 2) == 4  # 4 columns: intercept, age, race, collgrad
            @test size(X, 1) == length(y)  # Matching dimensions
            @test all(.!isnan.(y))  # No missing values in log wage
        end
        
        @testset "prepare_occupation_data" begin
            df_raw, _, _ = load_data(url)
            df, X, y = prepare_occupation_data(df_raw)
            
            @test size(X, 2) == 4  # intercept + age + white + collgrad
            @test maximum(y) == 7  # 7 occupation categories after collapsing
            @test minimum(y) == 1
            @test size(X, 1) == length(y)
        end
    end
    
    #--------------------------------------------------------------------------
    # Test 2: OLS via GMM
    #--------------------------------------------------------------------------
    @testset "OLS via GMM" begin
        # Create simple test data
        Random.seed!(123)
        N = 100
        X_test = hcat(ones(N), randn(N, 2))
        β_true = [1.0, 0.5, -0.3]
        y_test = X_test * β_true + 0.1 * randn(N)
        
        @testset "ols_gmm function properties" begin
            # Test that function returns a scalar
            obj_val = ols_gmm(β_true, X_test, y_test)
            @test obj_val isa Number
            @test obj_val >= 0  # Objective should be non-negative
            
            # Test that true parameters give small objective
            @test obj_val < 10.0  # Should be small for true params + small noise
        end
        
        @testset "ols_gmm optimization" begin
            result = optimize(b -> ols_gmm(b, X_test, y_test), 
                            rand(3), 
                            LBFGS(), 
                            Optim.Options(g_tol=1e-6))
            
            β_gmm = result.minimizer
            β_ols = X_test \ y_test
            
            # GMM should match OLS closely
            @test isapprox(β_gmm, β_ols, rtol=0.01)
        end
    end
    
    #--------------------------------------------------------------------------
    # Test 3: Data Simulation Functions
    #--------------------------------------------------------------------------
    @testset "Multinomial Logit Simulation" begin
        Random.seed!(456)
        N_sim = 1000
        J_sim = 4
        
        @testset "sim_logit output dimensions" begin
            Y, X = sim_logit(N_sim, J_sim)
            
            @test length(Y) == N_sim
            @test size(X, 1) == N_sim
            @test size(X, 2) == 4  # intercept + 3 covariates
            @test all(Y .>= 1) && all(Y .<= J_sim)
            @test all(isinteger.(Y))
        end
        
        @testset "sim_logit_w_gumbel output dimensions" begin
            Y, X = sim_logit_w_gumbel(N_sim, J_sim)
            
            @test length(Y) == N_sim
            @test size(X, 1) == N_sim
            @test all(Y .>= 1) && all(Y .<= J_sim)
        end
        
        @testset "Simulation produces all choices" begin
            Y, X = sim_logit(10000, 4)
            # With 10000 observations, should see all 4 choices
            @test length(unique(Y)) == 4
        end
    end
    
    #--------------------------------------------------------------------------
    # Test 4: Multinomial Logit MLE and GMM
    #--------------------------------------------------------------------------
    @testset "Multinomial Logit Estimation" begin
        # Use simulated data with known parameters
        Random.seed!(789)
        N_test = 500
        Y_sim, X_sim = sim_logit(N_test, 4)
        
        K = size(X_sim, 2)
        J = 4
        α_start = randn((J-1) * K) * 0.1
        
        @testset "mlogit_mle function" begin
            obj_val = mlogit_mle(α_start, X_sim, Y_sim)
            @test obj_val isa Number
            @test obj_val > 0  # Negative log-likelihood should be positive
        end
        
        @testset "mlogit_gmm function" begin
            obj_val = mlogit_gmm(α_start, X_sim, Y_sim)
            @test obj_val isa Number
            @test obj_val >= 0
        end
        
        @testset "mlogit_gmm_overid function" begin
            obj_val = mlogit_gmm_overid(α_start, X_sim, Y_sim)
            @test obj_val isa Number
            @test obj_val >= 0
        end
    end
    
    #--------------------------------------------------------------------------
    # Test 5: SMM Function
    #--------------------------------------------------------------------------
    @testset "SMM Estimation" begin
        Random.seed!(101)
        N_test = 200
        Y_sim, X_sim = sim_logit(N_test, 4)
        
        K = size(X_sim, 2)
        J = 4
        α_start = randn((J-1) * K) * 0.1
        
        @testset "mlogit_smm_overid function" begin
            # Test with small D for speed
            obj_val = mlogit_smm_overid(α_start, X_sim, Y_sim, 10)
            
            @test obj_val isa Number
            @test obj_val >= 0
            @test isfinite(obj_val)
        end
        
        @testset "SMM with different D values" begin
            obj_10 = mlogit_smm_overid(α_start, X_sim, Y_sim, 10)
            obj_50 = mlogit_smm_overid(α_start, X_sim, Y_sim, 50)
            
            # Both should be finite
            @test isfinite(obj_10)
            @test isfinite(obj_50)
        end
    end
    
    #--------------------------------------------------------------------------
    # Test 6: Parameter Recovery
    #--------------------------------------------------------------------------
    @testset "Parameter Recovery from Simulation" begin
        Random.seed!(111)
        N_large = 5000
        Y_sim, X_sim = sim_logit_w_gumbel(N_large, 4)
        
        K = size(X_sim, 2)
        J = 4
        
        # Try to recover parameters
        α_start = randn((J-1) * K) * 0.1
        
        result = optimize(a -> mlogit_mle(a, X_sim, Y_sim),
                         α_start,
                         LBFGS(),
                         Optim.Options(g_tol=1e-4, iterations=1000))
        
        @test Optim.converged(result)
        @test result.minimum < 10000  # Log-likelihood should be reasonable
    end
    
    #--------------------------------------------------------------------------
    # Test 7: Consistency Checks
    #--------------------------------------------------------------------------
    @testset "Consistency Checks" begin
        Random.seed!(222)
        N = 300
        Y, X = sim_logit(N, 4)
        
        @testset "Choice probabilities sum to 1" begin
            K = size(X, 2)
            J = 4
            α_test = randn((J-1) * K) * 0.1
            
            bigα = [reshape(α_test, K, J-1) zeros(K)]
            P = exp.(X * bigα) ./ sum.(eachrow(exp.(X * bigα)))
            
            # Each row should sum to approximately 1
            row_sums = sum(P, dims=2)
            @test all(isapprox.(row_sums, 1.0, atol=1e-10))
        end
        
        @testset "Moment conditions at optimum" begin
            # At the MLE, moment conditions should be close to zero
            α_start = randn(12) * 0.1
            
            result = optimize(a -> mlogit_mle(a, X, Y),
                            α_start,
                            LBFGS(),
                            Optim.Options(g_tol=1e-5, iterations=1000))
            
            if Optim.converged(result)
                α_hat = result.minimizer
                
                # Compute moment conditions
                K = size(X, 2)
                J = 4
                bigY = zeros(N, J)
                for j = 1:J
                    bigY[:,j] = Y .== j
                end
                
                bigα = [reshape(α_hat, K, J-1) zeros(K)]
                P = exp.(X * bigα) ./ sum.(eachrow(exp.(X * bigα)))
                
                g = zeros((J-1) * K)
                for j = 1:(J-1)
                    for k = 1:K
                        g[(j-1)*K + k] = mean((bigY[:,j] .- P[:,j]) .* X[:,k])
                    end
                end
                
                # Moments should be close to zero
                @test maximum(abs.(g)) < 0.1
            end
        end
    end
    
    #--------------------------------------------------------------------------
    # Test 8: Edge Cases
    #--------------------------------------------------------------------------
    @testset "Edge Cases" begin
        @testset "Small sample behavior" begin
            Random.seed!(333)
            Y_small, X_small = sim_logit(10, 3)
            
            @test length(Y_small) == 10
            @test size(X_small, 1) == 10
        end
        
        @testset "Different J values" begin
            for J_test in [3, 5, 6]
                Y, X = sim_logit(100, J_test)
                @test maximum(Y) <= J_test
                @test minimum(Y) >= 1
            end
        end
    end
    
end

println("\n" * "="^80)
println("All tests completed!")
println("="^80)