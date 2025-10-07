using Test
using Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM

include("PS5_Gao_source.jl")

@testset "PS5 Dynamic Discrete Choice Tests" begin
    
    @testset "Data Loading Tests" begin
        @testset "Static Data Loading" begin
            df_long = load_static_data()
            @test isa(df_long, DataFrame)
            @test nrow(df_long) > 0
            @test "Y" in names(df_long)
            @test "Odometer" in names(df_long)
            @test "RouteUsage" in names(df_long)
            @test "Branded" in names(df_long)
            @test "bus_id" in names(df_long)
            @test "time" in names(df_long)
            @test all(df_long.Y .∈ Ref([0, 1]))
            @test maximum(df_long.time) == 20
        end
        
        @testset "Dynamic Data Loading" begin
            d = load_dynamic_data()
            @test haskey(d, :Y)
            @test haskey(d, :X)
            @test haskey(d, :B)
            @test haskey(d, :Xstate)
            @test haskey(d, :Zstate)
            @test haskey(d, :N)
            @test haskey(d, :T)
            @test haskey(d, :xval)
            @test haskey(d, :xbin)
            @test haskey(d, :zbin)
            @test haskey(d, :xtran)
            @test haskey(d, :β)
            @test d.β == 0.9
            @test size(d.Y) == (d.N, d.T)
            @test size(d.X) == (d.N, d.T)
            @test length(d.B) == d.N
            @test size(d.Xstate) == (d.N, d.T)
            @test length(d.Zstate) == d.N
            @test all(d.Y .∈ Ref([0, 1]))
            @test all(d.B .∈ Ref([0, 1]))
        end
    end
    
    @testset "Static Estimation Tests" begin
        df_long = load_static_data()
        @test_nowarn estimate_static_model(df_long)
    end
    
    @testset "Future Value Computation Tests" begin
        d = load_dynamic_data()
        θ = [2.0, -0.15, 1.0]
        FV = zeros(d.zbin * d.xbin, 2, d.T + 1)
        
        @test_nowarn compute_future_value!(FV, θ, d)
        
        # Test terminal condition: FV at T+1 should be zero
        @test all(FV[:, :, d.T+1] .== 0.0)
        
        # Test that FV values are finite and reasonable
        @test all(isfinite.(FV[:, :, 1:d.T]))
        
        # Test dimensions
        @test size(FV) == (d.zbin * d.xbin, 2, d.T + 1)
        
        # Test that future values generally decrease with time
        # (due to discounting, values should be higher at earlier periods)
        @test mean(FV[:, :, 1]) >= mean(FV[:, :, d.T])
    end
    
    @testset "Log Likelihood Tests" begin
        d = load_dynamic_data()
        θ = [2.0, -0.15, 1.0]
        
        # Test that likelihood returns a finite number
        ll = log_likelihood_dynamic(θ, d)
        @test isfinite(ll)
        @test isa(ll, Float64)
        
        # Test that likelihood is negative (since we return -loglike)
        @test ll < 0
        
        # Test with different parameter values
        θ2 = [1.0, -0.1, 0.5]
        ll2 = log_likelihood_dynamic(θ2, d)
        @test isfinite(ll2)
        @test ll != ll2
        
        # Test with extreme parameter values
        θ_extreme = [10.0, -1.0, 2.0]
        ll_extreme = log_likelihood_dynamic(θ_extreme, d)
        @test isfinite(ll_extreme)
    end
    
    @testset "Dynamic Model Estimation Tests" begin
        d = load_dynamic_data()
        θ_start = [2.0, -0.15, 1.0]
        
        # Test that estimation function runs without error
        @test_nowarn estimate_dynamic_model(d, θ_start=θ_start)
        
        # Test with default starting values
        @test_nowarn estimate_dynamic_model(d)
    end
    
    @testset "Main Function Tests" begin
        # Test that main function runs without error
        @test_nowarn main()
    end
    
    @testset "Utility and Edge Case Tests" begin
        @testset "Create Grids Function" begin
            zval, zbin, xval, xbin, xtran = create_grids()
            
            @test length(zval) == zbin
            @test length(xval) == xbin
            @test size(xtran) == (zbin * xbin, xbin)
            @test all(0 .<= xtran .<= 1)  # Transition probabilities
            
            # Test that each row sums to 1 (probability constraint)
            row_sums = sum(xtran, dims=2)
            @test all(abs.(row_sums .- 1.0) .< 1e-10)
        end
        
        @testset "Future Value Edge Cases" begin
            d = load_dynamic_data()
            
            # Test with zero parameters
            θ_zero = [0.0, 0.0, 0.0]
            FV_zero = zeros(d.zbin * d.xbin, 2, d.T + 1)
            @test_nowarn compute_future_value!(FV_zero, θ_zero, d)
            @test all(isfinite.(FV_zero))
            
            # Test with negative mileage coefficient (expected)
            θ_neg = [1.0, -0.5, 0.5]
            FV_neg = zeros(d.zbin * d.xbin, 2, d.T + 1)
            @test_nowarn compute_future_value!(FV_neg, θ_neg, d)
            @test all(isfinite.(FV_neg))
        end
        
        @testset "Data Consistency Tests" begin
            d = load_dynamic_data()
            
            # Test state space consistency
            @test d.xbin > 0
            @test d.zbin > 0
            @test d.T > 0
            @test d.N > 0
            
            # Test that state indices are within bounds
            @test all(1 .<= d.Xstate .<= d.xbin)
            @test all(1 .<= d.Zstate .<= d.zbin)
            
            # Test discount factor is reasonable
            @test 0 < d.β < 1
        end
    end
end