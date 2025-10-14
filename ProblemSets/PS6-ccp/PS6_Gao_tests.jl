using Test
using Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM

include("PS6_Gao_source.jl")

# Define tests and run them

@testset "Rust Bus Replacement Model Tests" begin
    
    # Setup: Load data once for multiple tests
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdata.csv"
    df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
    zval, zbin, xval, xbin, xtran = create_grids()
    
    @testset "Data Loading and Reshaping" begin
        @test df_long isa DataFrame
        @test nrow(df_long) > 0
        @test :Y in names(df_long)
        @test :Odometer in names(df_long)
        @test :Xstate in names(df_long)
        @test :RouteUsage in names(df_long)
        @test :Branded in names(df_long)
        @test :time in names(df_long)
        @test size(Xstate, 2) == 20  # 20 time periods
        @test length(Zstate) == size(Xstate, 1)
        @test length(Branded) == size(Xstate, 1)
        @test maximum(df_long.time) == 20
    end
    
    @testset "Flexible Logit Estimation" begin
        flex_logit = estimate_flexible_logit(df_long)
        @test flex_logit isa GeneralizedLinearModel
        @test coef(flex_logit) isa Vector
        @test length(coef(flex_logit)) > 0
        @test all(isfinite.(coef(flex_logit)))
    end
    
    @testset "State Space Construction" begin
        state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
        @test state_df isa DataFrame
        @test nrow(state_df) == size(xtran, 1)
        @test :Odometer in names(state_df)
        @test :RouteUsage in names(state_df)
        @test :Branded in names(state_df)
        @test :time in names(state_df)
    end
    
    @testset "Future Value Computation" begin
        state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
        flex_logit = estimate_flexible_logit(df_long)
        β = 0.9
        T = 20
        
        FV = compute_future_values(state_df, flex_logit, xtran, xbin, zbin, T, β)
        
        @test size(FV) == (xbin*zbin, 2, T+1)
        @test all(isfinite.(FV))
        @test FV[:, :, 1] == zeros(xbin*zbin, 2)  # Initial period should be zeros
    end
    
    @testset "FVT1 Mapping" begin
        state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
        flex_logit = estimate_flexible_logit(df_long)
        FV = compute_future_values(state_df, flex_logit, xtran, xbin, zbin, 20, 0.9)
        
        fvt1 = compute_fvt1(df_long, FV, xtran, Xstate, Zstate, xbin, Branded)
        
        @test length(fvt1) == nrow(df_long)
        @test all(isfinite.(fvt1))
    end
    
    @testset "Structural Parameter Estimation" begin
        state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
        flex_logit = estimate_flexible_logit(df_long)
        FV = compute_future_values(state_df, flex_logit, xtran, xbin, zbin, 20, 0.9)
        fvt1 = compute_fvt1(df_long, FV, xtran, Xstate, Zstate, xbin, Branded)
        
        theta_hat = estimate_structural_params(df_long, fvt1)
        
        @test theta_hat isa GeneralizedLinearModel
        @test length(coef(theta_hat)) >= 2  # At least intercept + Odometer
        @test all(isfinite.(coef(theta_hat)))
        @test :Odometer in coefnames(theta_hat)
    end
    
    @testset "Integration Test" begin
        # Test that main() runs without errors
        @test main() === nothing
    end
end