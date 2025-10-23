# Load data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
         df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occ_code




function mlogit_with_Z(theta, X, Z, y)
    # Extract parameters
    # theta = [alpha1, alpha2, ..., alpha21, gamma]
    # alpha has K*(J-1) = 3*7 = 21 elements  
    # gamma is the coefficient on Z
    alpha = theta[1:(end-1)]  # first 21 elements
    gamma = theta[end]      # last element

    K = size(X, 2)  # number of covariates in X (3)
    J = length(unique(y))  # number of choices (8)
    N = length(y)   # number of observations

    # Create choice indicator matrix
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end

    # Reshape alpha into K x (J-1) matrix, add zeros for normalized choice J
    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]

    # Initialize probability matrix  
    T = promote_type(eltype(X), eltype(theta))
    num = zeros(T, N, J)
    dem = zeros(T, N)

    # Fill in: compute numerator for each choice j
    for j = 1:J
        num[:, j] = exp.(X * bigAlpha[:, j] .+ gamma .* (Z[:, j] .- Z[:, J]))
    end

    # Fill in: compute denominator (sum of numerators)
    dem = sum(num, dims = 2)

    # Fill in: compute probabilities
    P = num ./ dem

    # Fill in: compute negative log-likelihood
    loglike = -sum(bigY .* log.(P))

    return loglike
end







function plogit(θ, X, Z, J)
    K = size(X, 2)
    N = size(X, 1)
    α, γ = θ[1:end-1], θ[end]

    bigAlpha = [reshape(α, K, J-1) zeros(K)]

    # Initialize probability matrix
    T = promote_type(eltype(X), eltype(θ))
    num = zeros(T, N, J)
    dem = zeros(T, N)

    # compute numerator for each choice j
    for j = 1:J
        num[:, j] = exp.(X * bigAlpha[:, j] .+ γ .* (Z[:, j] .- Z[:, J]))
    end

    # compute denominator (sum of numerators)
    dem = sum(num, dims = 2)

    # compute probabilities
    P = num ./ dem

    return P
end

# Starting values
θ_start = [.0403744; .2439942; -1.57132; .0433254; .1468556; -2.959103; 
           .1020574; .7473086; -4.12005; .0375628; .6884899; -3.65577; 
           .0204543; -.3584007; -4.376929; .1074636; -.5263738; -6.199197; 
           .1168824; -.2870554; -5.322248; 1.307477]

function main_cfls()
    println("Estimating model...")
    td = TwiceDifferentiable(b -> mlogit_with_Z(b, X, Z, y), θ_start; autodiff = :forward)
    θ̂_optim = optimize(td, θ_start, LBFGS(), Optim.Options(g_tol=1e-5, iterations=100_000))
    θ̂_mle = θ̂_optim.minimizer
    H = Optim.hessian!(td, θ̂_mle)

    println("Wage coefficient (γ): ", round(θ̂_mle[end], digits=4))

    # Compute model fit
    J = length(unique(y))
    N = length(y)

    # Create choice indicator matrix
    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end

    P = plogit(θ̂_mle, X, Z, J)

    # Create comparison table
    modelfit_df = DataFrame(
        choice = 1:J,
        observed_share = vec(mean.(eachcol(bigY))),
        predicted_share = vec(mean.(eachcol(P)))
    )



    # Set γ = 0
    θ̂_cfl1 = vcat(θ̂_mle[1:end-1], 0.0)

    # Compute counterfactual predictions
    P_cfl1 = plogit(θ̂_cfl1, X, Z, J)

    # Add to table
    modelfit_df.cfl1_share = vec(mean.(eachcol(P_cfl1)))
    modelfit_df.cfl1_effect = modelfit_df.cfl1_share .- modelfit_df.predicted_share
    modelfit_df.avg_wage = [mean(Z[:, j]) for j in 1:J]



    # Increase all wages by 10%
    Z_cfl2 = Z .* 1.10
    P_cfl2 = plogit(θ̂_mle, X, Z_cfl2, J)

    modelfit_df.cfl2_share = vec(mean.(eachcol(P_cfl2)))
    modelfit_df.cfl2_effect = modelfit_df.cfl2_share .- modelfit_df.predicted_share

    println("\nModel fit and counterfactual results:")
    println(modelfit_df)

    return modelfit_df
end


