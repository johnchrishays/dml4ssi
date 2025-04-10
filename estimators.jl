

function naive_dml_phi(D, X, Y, naive_outcome_model, propensity_model, eps)
    T = length(D)
    W_observed = reduce(hcat, [X, D])
    W_observed_treat = copy(W_observed)
    W_observed_ctrl = copy(W_observed)
    Y_pred = predict(naive_outcome_model, W_observed)
    propensity_pred = predict(propensity_model, X)
    propensity_pred = clamp.(propensity_pred, eps, 1-eps)
    residuals = Y .- Y_pred
    curr_d_ind = size(X, 2) + 1
    W_observed_treat[:,curr_d_ind] .= 1
    W_observed_ctrl[:,curr_d_ind] .= 0
    plugin_estimator = predict(naive_outcome_model, W_observed_treat) .- predict(naive_outcome_model, W_observed_ctrl)

    debiasing_term = [D[t]/propensity_pred[t] - (1-D[t])/(1-propensity_pred[t]) for t in 1:T] .* residuals

    return plugin_estimator .+ debiasing_term, plugin_estimator
end

function naive_dml(D, X, Y, naive_outcome_model, propensity_model, eps)
    return mean(naive_dml_phi(D, X, Y, naive_outcome_model, propensity_model, eps)[1])
end

function dml4ssi_phi_ade(D, X, H, Y, outcome_model, propensity_model, eps)
    T = length(D)
    # Calculate residuals
    W_observed = create_features(X, D, H)
    curr_d_ind = size(X, 2) + 1
    Y_pred = predict(outcome_model, W_observed)
    propensity_pred = predict(propensity_model, X)
    propensity_pred = clamp.(propensity_pred, eps, 1-eps)

    W_observed_treat = copy(W_observed)
    W_observed_ctrl = copy(W_observed)
    W_observed_treat[:,curr_d_ind] .= 1
    W_observed_ctrl[:,curr_d_ind] .= 0
    plugin_estimator = predict(outcome_model, W_observed_treat) .- predict(outcome_model, W_observed_ctrl)

    residuals = Y .- Y_pred
    @assert sum(propensity_pred .<= 0 .|| propensity_pred .>= 1 ) == 0 "Propensity scores must be between 0 and 1"
    debiasing_term = [D[t]/propensity_pred[t] - (1-D[t])/(1-propensity_pred[t]) for t in 1:T] .* residuals

    treatment_effects = plugin_estimator .+ debiasing_term

    return treatment_effects, plugin_estimator

end

function dml4ssi_ade(D, X, H, Y, outcome_model, propensity_model, eps)
    return mean(dml4ssi_phi_ade(D, X, H, Y, outcome_model, propensity_model, eps))
end

function switchback_ht(D, Y, k)
    """
    Calculate Horwitz-Thomson style estimator in switchback experiment
    """
    treatment_sum = 0.0
    control_sum = 0.0
    n_treatment = 0
    n_control = 0
    switches = diff(D) .!= 0
    switch_indices = findall(switches)

    # Create mask for observations to keep (false for m periods after each switch)
    valid_obs = trues(length(D))
    for switch_idx in switch_indices
        start_idx = switch_idx + 1
        end_idx = min(start_idx + k - 1, length(D))
        valid_obs[start_idx:end_idx] .= false
    end

    for i in 1:length(D)
        if valid_obs[i]
            if D[i] == 1
                treatment_sum += Y[i]
                n_treatment += 1
            else
                control_sum += Y[i]
                n_control += 1
            end
        end
    end

    treatment_effect = (treatment_sum / n_treatment) - (control_sum / n_control)
    return treatment_effect, valid_obs
end

function dml4ssi_phi_switchback(D, X, H, Y, predictor, m, switch_period)
    """
    Calculate DML4SSI phi estimator in switchback experiment
    """
    T = length(D)

    # Calculate predicted values under all treatment and all control
    H_all_treat = h_star_global(D, X, H, 1, false)
    H_all_control = h_star_global(D, X, H, 0, false)
    
    W_all_treat = create_features(X, ones(length(D)), H_all_treat)
    W_all_control = create_features(X, zeros(length(D)), H_all_control)
    
    Y_pred_treat = predict(predictor, W_all_treat)
    Y_pred_control = predict(predictor, W_all_control)
    
    plugin_estimator = Y_pred_treat - Y_pred_control
    
    W_observed = create_features(X, D, H)
    Y_pred =  predict(predictor, W_observed)
    residuals = Y - Y_pred
    
    last_m_treatment = [all(D[max(t-m+1, 1):t] .== 1) for t in 1:T]
    last_m_control = [all(D[max(t-m+1, 1):t] .== 0) for t in 1:T]
    
    prob_m_ones = 1/2 - (m-1)/(4*switch_period)
    prob_m_zeros = 1/2 - (m-1)/(4*switch_period)
    debiasing_term = (last_m_treatment ./ prob_m_ones .- last_m_control ./ prob_m_zeros) .* residuals
    
    treatment_effects = plugin_estimator .+ debiasing_term
    
    return treatment_effects, plugin_estimator
end

function dml4ssi_switchback(D, X, Y, predictor, k, switch_period)
    return mean(dml4ssi_phi_switchback(D, X, Y, predictor, k, switch_period))
end

function dml4ssi_var(T, phis, m; sb)
    est = mean(phis)
    covariance_term = 0
    if sb
        variance_term = sum((phis .- est).^2)
        for t in 1:T
            for i in 1:(min(t,m)-1)
                covariance_term += 2*(phis[t] - est)*(phis[t-i] - est)
            end
        end
        return T^(-1) * (variance_term + covariance_term)
    else 
        theta = 1/2
        T1 = floor(T^theta)
        T2 = floor(T/T1)
        for t in 1:T1
            cov_sum = 0
            for s in ((t-1)*T2 + 1):(t*T2)
                cov_sum += (phis[Int(s)] - est)
            end
            covariance_term += cov_sum^2
        end
        return covariance_term / (T2*(T1-1))
    end
end

function ssac_var(T, phis)
    """Shared state as covariates approach. Doesn't account for covariances."""
    est = mean(phis)
    variance_term = sum((phis .- est).^2)
    return T^(-1) * variance_term
end

function ipw_var(D, X, Y, propensity_model, eps)
    T = length(D)
    Y0_mean = mean(Y[D.==0])
    Y1_mean = mean(Y[D.==1])
    propensity_pred = predict(propensity_model, X)
    s = mean(((Y .- Y1_mean) .* D ./ propensity_pred .- (Y .- Y0_mean) .* (1 .- D) ./ (1 .- propensity_pred)).^2)
    return s / T
end

function calculate_true_direct_effect(D, X, H, alpha; est_ade)
    T = size(D, 1)
    effect = [f_star(1, X[t,:], H[t,:], alpha, est_ade) - f_star(0, X[t,:], H[t,:], alpha, est_ade) for t in 1:T]
    return effect
end

function ipw(D, X, Y, propensity_model, eps)
    propensity_pred = predict(propensity_model, X)
    propensity_pred = clamp.(propensity_pred, eps, 1-eps)
    return mean(Y[D.==1] ./ propensity_pred[D.==1]) - mean(Y[D.==0] ./ (1 .- propensity_pred[D.==0]))
end

function make_gaussian_cis(estimates, ses, alpha)
    z_score = quantile(Normal(), 1 - alpha/2)
    ci_lower = estimates .- z_score .* ses
    ci_upper = estimates .+ z_score .* ses
    return ci_lower, ci_upper
end