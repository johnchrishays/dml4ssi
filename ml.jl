using DecisionTree
function calculate_metrics(y_true, y_pred)
    # Calculate RMSE
    rmse = sqrt(mean((y_true .- y_pred).^2))
    
    # Calculate R²
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Print 
    println("RMSE: ", round(rmse, digits=4))
    println("R²: ", round(r2, digits=4))
    
    return rmse, r2
end


function generate_models(T, m, n_covariates, switch_period; mode, est_ade, alpha, sd_error, n_trees, max_depth, eps, beta = 0.5)
    D, X, H, Y = dgp(T, m, n_covariates, switch_period; mode=mode, est_ade=est_ade, alpha=alpha, sd_error=sd_error, eps=eps, beta = beta)

    W = create_features(X, D, H)

    outcome_model = RandomForestRegressor(n_trees=n_trees, max_depth=max_depth)
    fit!(outcome_model, W, Y)

    propensity_model = RandomForestRegressor()
    fit!(propensity_model, X, D)

    return outcome_model, propensity_model
end

function generate_naive_models(T, m, n_covariates, switch_period; mode, est_ade, alpha, sd_error, n_trees, max_depth, eps, beta = 0.5)
    D, X, H, Y = dgp(T, m, n_covariates, switch_period; mode=mode, est_ade=est_ade, alpha=alpha_val, sd_error=sd_error, eps=eps, beta = beta)
    W = reduce(hcat, [X, D])

    outcome_model = RandomForestRegressor(n_trees=n_trees, max_depth=max_depth)
    fit!(outcome_model, W, Y)

    propensity_model = RandomForestRegressor(n_trees=n_trees, max_depth=max_depth)
    fit!(propensity_model, X, D)

    return outcome_model, propensity_model
end


function evaluate_models(outcome_model, propensity_model; mode, est_ade, alpha, sd_error, eps)
    D, X, H, Y = dgp(T, m, n_covariates, switch_period; mode=mode, est_ade=est_ade, alpha=alpha_val, sd_error=sd_error, eps=eps)
    X_test = create_features(X, D, H)
    y_pred = predict(outcome_model, X_test)
    println("\nOutcome model -- Out of sample:")
    calculate_metrics(Y, y_pred)

    propensity_pred = predict(propensity_model, X)
    println("Propensity model -- Out of sample:")
    calculate_metrics(D, propensity_pred)
    return propensity_pred
end