using Test
using Random, LinearAlgebra, Statistics, Distributions
using Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM
using MultivariateStats, FreqTables, ForwardDiff, LineSearches

include("PS8_Gao_source.jl")

@testset "PS8 Factor Models Tests" begin
    
    # Test data URL
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS8-factor/nlsy.csv"
    
    @testset "Data Loading" begin
        @testset "load_data function" begin
            df = load_data(url)
            @test size(df, 1) > 0
            @test size(df, 2) >= 12  # At least basic vars + 6 ASVABs
            @test "logwage" in names(df)
            @test "asvabAR" in names(df)
        end
    end
    
    @testset "ASVAB Correlations" begin
        df = load_data(url)
        
        @testset "compute_asvab_correlations function" begin
            cordf = compute_asvab_correlations(df)
            @test size(cordf) == (6, 6)
            @test all(diag(Matrix(cordf)) .≈ 1.0)  # Diagonal should be 1
            @test all(Matrix(cordf) .>= -1.0)
            @test all(Matrix(cordf) .<= 1.0)
        end
    end
    
    @testset "PCA and Factor Analysis" begin
        df = load_data(url)
        
        @testset "generate_PCA! function" begin
            df_pca = generate_PCA!(deepcopy(df))
            @test "asvabPCA" in names(df_pca)
            @test length(df_pca.asvabPCA) == size(df_pca, 1)
            @test !any(isnan.(df_pca.asvabPCA))
        end
        
        @testset "generate_Factor! function" begin
            df_factor = generate_Factor!(deepcopy(df))
            @test "asvabFactor" in names(df_factor)
            @test length(df_factor.asvabFactor) == size(df_factor, 1)
            @test !any(isnan.(df_factor.asvabFactor))
        end
    end
    
    @testset "Matrix Preparation" begin
        df = load_data(url)
        
        @testset "prepare_factor_matrices function" begin
            X, y, Xfac, asvabs = prepare_factor_matrices(df)
            @test size(X, 2) == 7  # 6 vars + constant
            @test size(Xfac, 2) == 4  # 3 vars + constant
            @test size(asvabs, 2) == 6  # 6 ASVAB tests
            @test size(X, 1) == size(y, 1) == size(Xfac, 1) == size(asvabs, 1)
            @test !any(isnan.(y))
        end
    end
    
    @testset "Factor Model Function" begin
        df = load_data(url)
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        
        # Simple parameter vector for testing
        K, L, J = size(X, 2), size(Xfac, 2), size(asvabs, 2)
        θ_test = vcat(
            vec(randn(L, J) * 0.1),  # γ parameters
            randn(K) * 0.1,          # β parameters  
            randn(J+1) * 0.1,        # α parameters
            abs.(randn(J+1)) .+ 0.1  # σ parameters (positive)
        )
        
        @testset "factor_model function properties" begin
            ll = factor_model(θ_test, X, Xfac, asvabs, y, 5)
            @test ll isa Real
            @test isfinite(ll)
        end
    end
    
    @testset "Basic Regression Tests" begin
        df = load_data(url)
        
        @testset "Base regression runs" begin
            reg = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), df)
            @test length(coef(reg)) == 7  # 6 vars + intercept
        end
        
        @testset "Full ASVAB regression runs" begin
            reg = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK), df)
            @test length(coef(reg)) == 13  # 6 + 6 ASVABs + intercept
        end
    end
end

# Run the tests
println("Running PS8 Factor Models tests...")
println("All tests passed!")