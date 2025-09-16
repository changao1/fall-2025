using Optim, DataFrames, CSV, HTTP, GLM, FreqTables, LinearAlgebra, Random, Statistics

cd(@__DIR__)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 1
#:::::::::::::::::::::::::::::::::::::::::::::::::::
f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
startval = rand(1)   # random starting value
result = optimize(minusf, startval, BFGS())
println("argmin (minimizer) is ",Optim.minimizer(result)[1])
println("min is ",Optim.minimum(result))

result_better = optimize(minusf, [-7.0], BFGS())
println(result_better)


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1

function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end

beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println("ols", beta_hat_ols.minimizer)

bols = inv(X'*X)*X'*y

# standard errors
MSE = sum((y.-X*bols).^2)
VCOV=MSE*inv(X'*X)
se_bols = sqrt.(diag(VCOV))

println("OLS estimates: ", bols)
println("OLS standard errors: ", se_bols)
df.white = df.race.==1
bols_lm = lm(@formula(married ~ age + white + collgrad), df)


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3
#:::::::::::::::::::::::::::::::::::::::::::::::::::
function logit(alpha, X, d)


    loglike= -sum(y .* log.(1 ./(1 .+ exp.(-X * alpha))) + (1 .- y) .* log.(1 .- 1 ./(1 .+ exp.(-X * alpha))))


    return loglike
end

beta_hat_logit = optimize(b -> logit(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))

println("logit", beta_hat_logit.minimizer)




#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 4
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# Make sure white and collgrad are 0/1 variables
df.white = df.race .== 1
df.collgrad = df.collgrad .== 1

# Fit the logit model using glm
model = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())

# Print results

println("check", coef(model))      # coefficients
# println("check std", stderror(model))  # standard errors
# println(deviance(model))  # residual deviance

println("ols", beta_hat_ols.minimizer)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 5
#:::::::::::::::::::::::::::::::::::::::::::::::::::
freqtable(df, :occupation) # note small number of obs in some occupations
df = dropmissing(df, :occupation)
df[df.occupation.==8 ,:occupation] .= 7
df[df.occupation.==9 ,:occupation] .= 7
df[df.occupation.==10,:occupation] .= 7
df[df.occupation.==11,:occupation] .= 7
df[df.occupation.==12,:occupation] .= 7
df[df.occupation.==13,:occupation] .= 7
freqtable(df, :occupation) # problem solved

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation


function mlogit(alpha, X, d)

    N, K = size(X)
    
    J = 7  # number of choices
    base = J
    beta = reshape(alpha, K, J-1)  # K x (J-1)
    ll = 0.0
    for i in 1:N
        xb = [dot(X[i, :], beta[:, j]) for j in 1:J-1]
        xb = vcat(xb, 0.0)  # base category utility is 0
        denom = sum(exp.(xb))
        numer = exp(xb[y[i]])
        ll += log(numer / denom)
    end
    return -ll
end



K = size(X,2)
J = 7
startval = zeros(K*(J-1))  # or try rand(K*(J-1)), rand(K*(J-1)).*2 .- 1, etc.

result = optimize(b -> mlogit(b, X, y), startval, LBFGS(), Optim.Options(g_tol=1e-5))
println("Estimated coefficients: ", Optim.minimizer(result))

