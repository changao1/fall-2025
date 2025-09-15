# Unit Tests for PS1_Gao.jl
# Chang Gao - ECON 6343 Econometrics III

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

# Set seed for reproducible tests
Random.seed!(1234)

@testset "PS1_Gao Tests" begin
    
    @testset "matrixops function tests" begin
        # Test with simple 2x2 matrices
        A = [1.0 2.0; 3.0 4.0]
        B = [2.0 1.0; 1.0 2.0]
        
        out1, out2, out3 = matrixops(A, B)
        
        # Test element-wise product
        @test out1 ≈ [2.0 2.0; 3.0 8.0]
        
        # Test A'B (transpose A times B)
        expected_out2 = [1.0 2.0; 3.0 4.0]' * [2.0 1.0; 1.0 2.0]
        @test out2 ≈ expected_out2
        
        # Test sum of all elements of A+B
        @test out3 ≈ sum([3.0 3.0; 4.0 6.0])
        @test out3 ≈ 16.0
        
        # Test error handling for different sized matrices
        C = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # 3x2 matrix
        D = [1.0 2.0; 3.0 4.0]           # 2x2 matrix
        
        @test_throws ErrorException matrixops(C, D)
        
        # Test with column vectors (should work after reshaping)
        vec1 = reshape([1.0, 2.0, 3.0], :, 1)
        vec2 = reshape([2.0, 3.0, 4.0], :, 1)
        
        out1_vec, out2_vec, out3_vec = matrixops(vec1, vec2)
        @test size(out1_vec) == (3, 1)
        @test out1_vec ≈ [2.0; 6.0; 12.0]
        @test out3_vec ≈ sum([3.0; 5.0; 7.0])
    end
    
    @testset "q1 function tests" begin
        # Run q1 and test outputs
        A, B, C, D = q1()
        
        # Test matrix dimensions
        @test size(A) == (10, 7)
        @test size(B) == (10, 7)
        @test size(C) == (5, 7)
        @test size(D) == (10, 7)
        
        # Test that A elements are in range [-5, 10]
        @test all(-5 .<= A .<= 10)
        
        # Test that D contains only non-positive elements of A or zeros
        @test all(D .<= 0)
        @test all((D .== 0) .| (D .== A .* (A .<= 0)))
        
        # Test that C is constructed correctly from A and B
        @test C == [A[1:5, 1:5] B[1:5, 6:7]]
        
        # Test file creation
        @test isfile("matrixpractice.jld")
        @test isfile("firstmatrix.jld")
        @test isfile("Cmatrix.csv")
        @test isfile("Dmatrix.dat")
        
        # Test loading saved matrices
        loaded_data = load("matrixpractice.jld")
        @test haskey(loaded_data, "A")
        @test haskey(loaded_data, "B")
        @test haskey(loaded_data, "G")
        
        # Test CSV file structure
        C_csv = CSV.read("Cmatrix.csv", DataFrame)
        @test size(C_csv) == (5, 7)
    end
    
    @testset "q2 function tests" begin
        # Create test matrices
        Random.seed!(1234)
        test_A = rand(3, 3)
        test_B = rand(3, 3)
        test_C = randn(3, 3)
        
        # Test that q2 runs without error
        @test q2(test_A, test_B, test_C) === nothing
        
        # Test element-wise product logic
        AB_expected = test_A .* test_B
        AB_manual = [test_A[i,j] * test_B[i,j] for i in 1:size(test_A,1), j in 1:size(test_A,2)]
        @test AB_expected ≈ AB_manual
        
        # Test filtering logic
        Cprime_expected = test_C[(-5 .<= test_C) .& (test_C .<= 5)]
        Cprime_manual = Float64[]
        for val in test_C
            if -5 <= val <= 5
                push!(Cprime_manual, val)
            end
        end
        @test sort(Cprime_expected) ≈ sort(Cprime_manual)
    end
    
    @testset "q3 function tests" begin
        # Skip if nlsw88.csv doesn't exist
        if isfile("nlsw88.csv")
            # Test that q3 runs without error
            @test q3() === nothing
            
            # Test that processed file is created
            @test isfile("nlsw88_processed.csv")
            
            # Load the data and test basic properties
            nlsw88 = CSV.read("nlsw88.csv", DataFrame)
            
            # Test that required columns exist
            required_cols = [:never_married, :collgrad, :race, :grade, :industry, :occupation, :wage, :ttl_exp]
            for col in required_cols
                @test col in names(nlsw88)
            end
            
            # Test that percentages are reasonable
            never_married_pct = sum(nlsw88.never_married .== 1) / nrow(nlsw88) * 100
            college_grad_pct = sum(nlsw88.collgrad .== 1) / nrow(nlsw88) * 100
            
            @test 0 <= never_married_pct <= 100
            @test 0 <= college_grad_pct <= 100
            
            # Test wage statistics
            @test all(nlsw88.wage .> 0)  # wages should be positive
            
            # Test that race frequency table has expected structure
            race_freq = freqtable(nlsw88.race)
            @test length(race_freq) == 3  # should have 3 race categories
            @test sum(race_freq) == nrow(nlsw88)  # total should equal number of rows
        else
            @test_skip "nlsw88.csv file not found, skipping q3 tests"
        end
    end
    
    @testset "q4 function tests" begin
        # First run q1 to create necessary files
        q1()
        
        # Test that q4 runs without error
        @test q4() === nothing
        
        # Test that required files exist
        @test isfile("matrixpractice.jld")
        @test isfile("nlsw88_processed.csv")
        
        # Test loading matrices from JLD file
        @load "matrixpractice.jld"
        @test @isdefined A
        @test @isdefined B
        @test @isdefined C
        @test @isdefined D
        
        # Test that matrixops works with loaded matrices
        result = matrixops(A, B)
        @test length(result) == 3  # Should return 3 outputs
        
        # Test error handling for mismatched sizes
        try
            matrixops(C, D)  # Different sizes should throw error
            @test false  # Should not reach here
        catch e
            @test isa(e, ErrorException)
            @test occursin("same size", e.msg)
        end
    end
    
    @testset "Integration tests" begin
        # Test the full pipeline
        Random.seed!(1234)
        
        # Run all functions in sequence
        A, B, C, D = q1()
        q2(A, B, C)
        
        if isfile("nlsw88.csv")
            q3()
            q4()
        else
            @test_skip "nlsw88.csv not available for integration test"
        end
        
        # Verify all expected files are created
        expected_files = ["matrixpractice.jld", "firstmatrix.jld", "Cmatrix.csv", "Dmatrix.dat"]
        for file in expected_files
            @test isfile(file)
        end
        
        # Test matrix properties are maintained
        @test size(A) == (10, 7)
        @test size(B) == (10, 7)
        @test size(C) == (5, 7)
        @test size(D) == (10, 7)
    end
end

println("All tests completed!")