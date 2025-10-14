#=
Problem Set 6 - Starter Code
ECON 6343: Econometrics III
Rust (1987) Bus Engine Replacement Model - CCP Estimation
=#

# Load required packages
using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV

include("create_grids.jl")

#========================================
QUESTION 1: Data Loading and Reshaping
========================================#

function load_and_reshape_data(url::String)

    # url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    
    # Create bus id variable
    df = @transform(df, :bus_id = 1:size(df,1))
    
    #---------------------------------------------------
    # Reshape from wide to long (done twice because 
    # stack() requires doing one variable at a time)
    #---------------------------------------------------
    
    # First reshape the decision variable (Y1-Y20)
    dfy = @select(df, :bus_id, :Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10,
                      :Y11, :Y12, :Y13, :Y14, :Y15, :Y16, :Y17, :Y18, :Y19, :Y20,
                      :RouteUsage, :Branded)
    dfy_long = DataFrames.stack(dfy, Not([:bus_id, :RouteUsage, :Branded]))
    rename!(dfy_long, :value => :Y)
    dfy_long = @transform(dfy_long, :time = kron(collect(1:20), ones(size(df,1))))
    select!(dfy_long, Not(:variable))
    
    # Next reshape the odometer variable (Odo1-Odo20)
    dfx = @select(df, :bus_id, :Odo1, :Odo2, :Odo3, :Odo4, :Odo5, :Odo6, :Odo7, 
                      :Odo8, :Odo9, :Odo10, :Odo11, :Odo12, :Odo13, :Odo14, :Odo15,
                      :Odo16, :Odo17, :Odo18, :Odo19, :Odo20)
    dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
    rename!(dfx_long, :value => :Odometer)
    dfx_long = @transform(dfx_long, :time = kron(collect(1:20), ones(size(df,1))))
    select!(dfx_long, Not(:variable))
    
    # Join reshaped dataframes back together
    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id, :time])
    sort!(df_long, [:bus_id, :time])

    return df_long
end


#========================================
QUESTION 2: Flexible Logit Estimation
========================================#

"""
    estimate_flexible_logit(df::DataFrame)

Estimate a flexible logit model with fully interacted terms up to 7th order.

# Arguments
- `df::DataFrame`: Long panel data frame

# Returns
- Fitted GLM model object

# TODO:
- Create squared terms for continuous variables
- Use GLM with formula syntax to specify fully interacted model
- Remember: Odometer * RouteUsage * Branded * time means all interactions up to the product of all four
"""
function estimate_flexible_logit(df::DataFrame)

     flex_logit = glm(@formula(Y ~ Odometer * RouteUsage * Branded * time * time * Odometer * RouteUsage), 
                     df, 
                     Binomial(), 
                     LogitLink())

    
    return flex_logit
end


#========================================
QUESTION 3: CCP-Based Estimation
========================================#

"""
    construct_state_space(xbin::Int, zbin::Int, xval::Vector, zval::Vector)

Construct the state space grid for all possible states.

# Arguments
- `xbin::Int`: Number of mileage bins
- `zbin::Int`: Number of route usage bins  
- `xval::Vector`: Grid points for mileage
- `zval::Vector`: Grid points for route usage

# Returns
- `state_df::DataFrame`: Data frame with all possible state combinations

"""
function construct_state_space(xbin::Int, zbin::Int, xval::Vector, zval::Matrix)
function construct_state_space(xbin::Int, zbin::Int, xval::Vector, zval::Vector)
    
    state_df = DataFrame(
        Odometer = kron(xval, ones(zbin)),
        RouteUsage = kron(ones(xbin), zval),
        Branded = zeros(size(xtrans,1)),  # Placeholder, will be updated in loop   
        time = zeros(size(xtrans,1))  # Placeholder, will be updated in loop
        Branded = zeros(xbin * zbin),  # Placeholder, will be updated in loop   
        time = zeros(xbin * zbin)  # Placeholder, will be updated in loop
    )
    
    return state_df
end


"""
    compute_future_values(state_df::DataFrame, 
                          flex_logit::GeneralizedLinearModel,
                          xtran::Matrix, 
                          xbin::Int, 
                          T::Int, 
                          β::Float64)

Compute future value terms using CCPs from the flexible logit.

# Arguments
- `state_df::DataFrame`: State space data frame
- `flex_logit`: Fitted flexible logit model
- `xtran::Matrix`: State transition matrix
- `xbin::Int`: Number of mileage bins
- `T::Int`: Number of time periods
- `β::Float64`: Discount factor (0.9)

# Returns
- `FV::Array{Float64,3}`: Future value array (states × brand × time)

# TODO:
- Initialize FV as zeros(size(xtran,1), 2, T+1)
- Loop over t = 2 to T
- Loop over brand states b ∈ {0,1}
- Update state_df with current t and b values
- Use predict() to get p0 (probability of replacement)
- Store -β * log(p0) in FV[:, b+1, t+1]
"""
function compute_future_values(state_df::DataFrame, 
                                flex_logit::GeneralizedLinearModel,
                                xtran::Matrix, 
                                xbin::Int, 
                                T::Int, 
                                β::Float64)
    

    FV = zeros(xbin*zbin, 2, T+1)
    zbin = Int(size(state_df, 1) / xbin)
    FV = zeros(size(state_df, 1), 2, T+1)
    
    for t in 2:T
        for b in 0:1
            # Update state_df
            @with(state_df, :time .= t)
            state_df.time .= t
            @with(state_df, :Branded .= b)

            
            # Compute p0 using predict()
            # p0 = predict(flex_logit, state_df, type = :response)
            p0 = 1 .- convert(Array{Float64}, predict(flex_logit, state_df, type = :response))  # Probability of not replacing
            # Store -β * log.(p0) in FV
            FV[:, b+1, t+1] = -β * log.(p0)
            FV[:, b+1, t] = -β * log.(p0)
        end
    end
    
    return FV
end


"""
    compute_fvt1(df_long::DataFrame, 
                 FV::Array{Float64,3},
                 xtran::Matrix,
                 Xstate::Vector,
                 Zstate::Vector,
                 xbin::Int)

Map future values from state space to actual data.

# Arguments
- `df_long::DataFrame`: Original long data frame
- `FV::Array{Float64,3}`: Future value array
- `xtran::Matrix`: State transition matrix
- `Xstate::Vector`: Mileage state for each observation
- `Zstate::Vector`: Route usage state for each observation  
- `xbin::Int`: Number of mileage bins

# Returns
- `FVT1::Vector`: Future value term for each observation in long format

# TODO:
- Initialize FVT1 matrix to store results
- Loop over observations i and time periods t
- Compute row indices in xtran based on Xstate[i] and Zstate[i]
- Calculate FVT1[i,t] = (xtran[row1,:] - xtran[row0,:])'* FV[row0:row0+xbin-1, B[i]+1, t+1]
- Convert to long format vector
"""
function compute_fvt1(df_long::DataFrame, 
                      FV::Array{Float64,3},
                      xtran::Matrix,
                      Xstate::Vector,
                      Zstate::Vector,
                      xbin::Int)
                      B::Vector,
                      xbin::Int,
                      T::Int)
    
    # Get dimensions
    N = length(unique(df_long.bus_id))  # Adjust column name as needed
    T = 20
    N = size(df_long, 1) ÷ T
    

    FVT1 = zeros(N, T)
    FVT1 = zeros(N*T)
    
    for i in 1:N
        row0 = 1 + (Zstate[i]-1)*xbin
        bus_indices = (1:T) .+ (i-1)*T
        row0_z = 1 + (Zstate[bus_indices[1]]-1)*xbin
        for t in 1:T
            # Compute row0 and row1 indices
            row1 = row0 + Xstate[i,t]-1
            FVT1[i,t] = dot((xtran[row1,:] .- xtran[row0,:]), FV[row0:row0+xbin-1, B[i]+1, t+1])
            
            current_obs_index = (i-1)*T + t
            row1_x = row0_z + Xstate[current_obs_index] - 1
            FVT1[current_obs_index] = dot((xtran[row1_x,:] .- xtran[row0_z,:]), FV[row0_z:row0_z+xbin-1, B[current_obs_index]+1, t+1])
        end
    end
    

    fvt1_long = FVT1'[:]
    
    return fvt1_long
    return FVT1
end


"""
    estimate_structural_params(df_long::DataFrame, fvt1::Vector)

Estimate structural parameters θ using GLM with offset.

# Arguments
- `df_long::DataFrame`: Long panel data with decision variable
- `fvt1::Vector`: Future value term to use as offset

# Returns
- Fitted GLM model with structural parameters

# TODO:
- Add fvt1 as a column to df_long
- Estimate logit with Odometer and Branded as regressors
- Use offset argument to include future value with coefficient = 1
"""
function estimate_structural_params(df_long::DataFrame, fvt1::Vector)
    # TODO: Add future value to data frame
    # df_long = @transform(df_long, fv = fvt1)
    df_with_fv = copy(df_long)
    df_with_fv.fv = fvt1
    
    # TODO: Estimate structural model
    # theta_hat = glm(@formula(Y ~ Odometer + Branded), 
    #                 df_long, 
    #                 Binomial(), 
    #                 LogitLink(), 
    #                 offset=df_long.fv)
    theta_hat = glm(@formula(Y ~ Odometer + Branded), 
                    df_with_fv, 
                    Binomial(), 
                    LogitLink(), 
                    offset=df_with_fv.fv)
    
    return theta_hat
end


#========================================
MAIN WRAPPER FUNCTION
========================================#

"""
    main()

Main wrapper function to estimate the Rust model using CCPs.
This function calls all the helper functions in sequence.

# TODO:
- Call each function in order
- Pass results between functions appropriately
- Print results at the end
"""
function main()
    println("="^60)
    println("Rust (1987) Bus Engine Replacement Model - CCP Estimation")
    println("="^60)
    
    # Set parameters
    β = 0.9  # Discount factor
    T = 20

    zval, zbin, xval, xbin, xtran = create_grids()
    
    
    # Step 1: Load and reshape data
    println("\nStep 1: Loading and reshaping data...")
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdata.csv"
    df_long = load_and_reshape_data(url)
    println("Data loaded with $(nrow(df_long)) rows and $(ncol(df_long)) columns.")
    println("First few rows of the data:")
    first(df_long, 5) |> println
    
    
    # Step 2: Estimate flexible logit
    println("\nStep 2: Estimating flexible logit...")
    flexlogitresults = estimate_flexible_logit(df_long)
    println(flexlogitresults)

    
    # Step 3a: Construct state transition matrices
    println("\nStep 3a: Constructing state transition matrices...")
    xbin, zbin, xval, zval, xtran = create_grids()

    
    # Step 3b: Construct state space
    println("\nStep 3b: Constructing state space...")
    state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
    state_df = construct_state_space(xbin, zbin, xval, zval)
    
    
    # Step 3c: Compute future values
    println("\nStep 3c: Computing future values...")
    FV = compute_future_values(state_df, flexlogitresults, xtran, xbin, zbin, 20, β)
    FV = compute_future_values(state_df, flexlogitresults, xtran, xbin, T, β)

    
    # Step 3d: Map to actual data
    println("\nStep 3d: Mapping future values to data...")
    efvt1 = compute_fvt1(df_long, FV, xtran, df_long.Odometer, df_long.RouteUsage, xbin, df_long.branded)
    return
    # Discretize Odometer and RouteUsage to get state indices
    Xstate = map(x -> findmin(abs.(xval .- x))[2], df_long.Odometer)
    Zstate = map(z -> findmin(abs.(zval .- z))[2], df_long.RouteUsage)

    fvt1 = compute_fvt1(df_long, FV, xtran, Xstate, Zstate, df_long.Branded, xbin, T)
    
    # Step 3e: Estimate structural parameters
    println("\nStep 3e: Estimating structural parameters...")
    # TODO: Call estimate_structural_params()
    theta_hat = estimate_structural_params(df_long, fvt1)
    
    # Print results
    println("\n" * "="^60)
    println("RESULTS")
    println("="^60)
    # TODO: Print coefficient estimates and standard errors
    println("Structural Parameter Estimates (θ):")
    println(theta_hat)
    
    return nothing
end

# Run the estimation (uncomment when ready to test)
# @time main()