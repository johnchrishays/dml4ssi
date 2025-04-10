using Dates
using Distributions
using DataFrames
using LaTeXStrings
using Statistics



function sigmoid(x)
    return 1 / (1 + exp(-x))
end

function h_star_sb(D_prev, X_prev, H_prev)
    """ 
    Compute the shared state given the previous state observations
    """
    dim = size(X_prev, 1) + 1
    H_new = reduce(vcat, [D_prev, X_prev, H_prev[1:size(H_prev, 1)-dim]])
    return H_new
end

function h_star_ade(D_prev, X_prev, Y_prev, H_prev)
    """
    Compute the shared state given the previous state observations
    """
    rho = 0.75
    sigma = 1

    mu = 1
    H_new = mu + rho * (H_prev[1] - mu) + randn()*sigma
    return H_new
end


function h_star_global(D, X, H, treatment_assignment, est_ade)
    """
    Compute the shared state under global treatment or control
    """
    T = length(D)
    d_H = size(H, 2)
    H_global = zeros(T, d_H)
    H_global[1,:] = randn(d_H) # random initial state
    for t in 2:T
        if est_ade
            H_global[t,:] = h_star_ade(treatment_assignment, X[t-1,:], Y[t-1], H_global[t-1,:]) 
        else
            H_global[t,:] = h_star_sb(treatment_assignment, X[t-1,:], H_global[t-1,:])
        end
    end
    return H_global
end

function f_star(D, X, H, alpha, est_ade; beta = 0.5)
    """ 
    Compute the outcome at time t given covariates and the shared state
    """
    T = length(D)
    x_effect = sin(X[1]*2*pi)
    d_effect = 2*D - 1  
    if est_ade
        h_effect = H[1] 
        return x_effect + d_effect + alpha * h_effect * D
    else
        rng = collect(1:6:size(H, 1))
        h_effect = exp.(-H[rng] / 3)
        h_effect = 2 * sigmoid(sum(h_effect)) - 1 
        return 2 * beta * x_effect + d_effect + 2 * (1 - beta) * alpha * h_effect
    end
end


function dgp(T, m, n_covariates, switch_period; mode="switch", est_ade = true, alpha = 1, sd_error = 1, eps = 0.1, beta = 0.5)
      
    sigma = 1
    if est_ade
        X = (randn(T, n_covariates) .+ 1) .* sigma
    else
        X = rand(T, n_covariates)
    end
    
    if mode == "switch"
        D = zeros(Int, T)
        first_switch = rand(1:switch_period)
        D[1:first_switch-1] .= rand() < 0.5
        for i in 1:ceil(Int, T/switch_period)
            start_idx = (i-1)*switch_period + first_switch
            end_idx = min(i*switch_period + first_switch - 1, T) 
            D[start_idx:end_idx] .= rand() < 0.5 
        end
    elseif mode == "global treatment"
        D = ones(Int, T)
    elseif mode == "global control"
        D = zeros(Int, T)
    elseif mode == "observational"
        D = [rand() < max(min(X[t,1], 1-eps), eps) ? 1 : 0 for t in 1:T]
    end 
    @assert !isnothing(D) "D must be defined"
    
    Y = zeros(T)
    if est_ade
        d_H = 1
    else
        d_H = (size(X, 2) + 1)*m
    end
    H = zeros(T, d_H)
    for t in 2:T
        if est_ade
            H[t,1] = h_star_ade(D[t-1], X[t-1,:], Y[t-1], H[t-1,:]) 
        else
            H[t,:] = h_star_sb(D[t-1], X[t-1,:], H[t-1,:])
        end
        Y[t] = f_star(D[t], X[t,:], H[t,:], alpha, est_ade; beta = beta) + randn()/sd_error
    end
    
    return D, X, H, Y
end

function create_features(X, D, H)
    return reduce(hcat, [X, D, H])
end
