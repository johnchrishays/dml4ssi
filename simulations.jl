using ProgressMeter

n_bins = 90

LEGEND_SIZE = 14
TICK_SIZE = 12
opacity = 0.5

pct = 0.01
sz = (800, 400)
legendsz = (1000, 100)
margin = 5mm


function run_simulations_ade(N, T, m, n_covariates, switch_period, n_trees, max_depth, alpha, sd_error, eps = 0.05)
    dml_ade_estimates = zeros(N)
    ipw_ade_estimates = zeros(N)
    true_direct_effects = zeros(N)
    plugin_ade_estimates = zeros(N)
    dml_naive_estimates = zeros(N)
    plugin_naive_estimates = zeros(N)
    dml_ade_ses = zeros(N) 
    ssac_ses = zeros(N)
    plugin_ses = zeros(N)
    ipw_ade_ses = zeros(N)
    dml_naive_ses = zeros(N)

    @showprogress 1 "ADE simulations:" for i in 1:N
        # Train new predictor on auxiliary data for direct effect
        outcome_model, propensity_model = generate_models(T, m, n_covariates, switch_period; mode="observational", est_ade = true, alpha=alpha, sd_error=sd_error, n_trees=n_trees, max_depth=max_depth, eps=eps)

        if i == 1
            println("\nDirect effect model -- Out of sample:")
            evaluate_models(outcome_model, propensity_model; mode="observational", est_ade=true, alpha=alpha, sd_error=sd_error, eps=eps)
        end
        
        D, X, H, Y = dgp(T, m, n_covariates, switch_period; mode = "observational", est_ade = true, alpha=alpha, sd_error=sd_error)
        dml_ade_estimates_phi, plugin_ade_estimates_phi = dml4ssi_phi_ade(D, X, H, Y, outcome_model, propensity_model, eps)
        dml_ade_estimates[i] = mean(dml_ade_estimates_phi)
        plugin_ade_estimates[i] = mean(plugin_ade_estimates_phi)
        dml_ade_ses[i] = sqrt(dml4ssi_var(T, dml_ade_estimates_phi, m, sb=false) / T)
        plugin_ses[i] = sqrt(dml4ssi_var(T, plugin_ade_estimates_phi, m, sb=false) / T)
        ssac_ses[i] = sqrt(ssac_var(T, dml_ade_estimates_phi) / T)
        true_direct_effects[i] = mean(calculate_true_direct_effect(D, X, H, alpha; est_ade=true))
        ipw_ade_estimates[i] = ipw(D, X, Y, propensity_model, eps)
        naive_outcome_model, naive_propensity_model = generate_naive_models(T, m, n_covariates, switch_period; mode="observational", est_ade=true, alpha=alpha, sd_error=sd_error, n_trees=n_trees, max_depth=max_depth, eps=eps)
        dml_naive_estimates_phi, plugin_naive_estimates_phi = naive_dml_phi(D, X, Y, naive_outcome_model, naive_propensity_model, eps)
        dml_naive_estimates[i], plugin_naive_estimates[i] = mean(dml_naive_estimates_phi[1]), mean(plugin_naive_estimates_phi[1])
        dml_naive_ses[i] = sqrt(ssac_var(T, dml_naive_estimates_phi) / T)

        ipw_ade_ses[i] = sqrt(ipw_var(D, X, Y, propensity_model, eps) / T)
    end
    
    return dml_ade_estimates, ipw_ade_estimates, plugin_ade_estimates, true_direct_effects, dml_ade_ses, ssac_ses, dml_naive_estimates, ipw_ade_ses, plugin_ses, dml_naive_ses
end

function run_simulations_sb(N, T, m, n_covariates, switch_period, n_trees, max_depth, alpha, sd_error, eps = 0.05; beta = 0.5)
    switchback_estimates = zeros(N)
    dml_switchback_estimates = zeros(N)
    naive_sb_estimates = zeros(N)
    true_gte = zeros(N)
    plugin_sb_estimates = zeros(N)
    dml_naive_estimates = zeros(N)
    plugin_naive_estimates = zeros(N)
    ssac_estimates = zeros(N)

    switchback_ses = zeros(N)
    dml_switchback_ses = zeros(N) 
    dml_naive_ses = zeros(N) 
    plugin_ses = zeros(N) 
    ipw_ses = zeros(N) 
    ssac_ses = zeros(N)
    @showprogress 1 "Switchback simulations:" for i in 1:N
        # Train new predictor on auxiliary data for switchback
        switchback_model, switchback_propensity_model = generate_models(T, m, n_covariates, switch_period; mode="switch", est_ade = false, alpha=alpha, sd_error=sd_error, n_trees=n_trees, max_depth=max_depth, eps=eps)

        if i == 1
            println("Switchback model -- Out of sample:")
            evaluate_models(switchback_model, switchback_propensity_model; mode="switch", est_ade=false, alpha=alpha, sd_error=sd_error, eps=eps)
        end
        
        # Switchback experiments
        D, X, H, Y = dgp(T, m, n_covariates, switch_period; mode = "switch", est_ade = false, alpha=alpha, sd_error=sd_error, beta = beta)
        switchback_estimates[i], valid_obs = switchback_ht(D, Y, m)
        dml_switchback_estimates_phi, plugin_sb_estimates_phi = dml4ssi_phi_switchback(D, X, H, Y, switchback_model, m, switch_period)
        ssac_estimates_phi, _ = dml4ssi_phi_ade(D, X, H, Y, switchback_model, switchback_propensity_model, eps)
        dml_switchback_estimates[i] = mean(dml_switchback_estimates_phi)
        plugin_sb_estimates[i] = mean(plugin_sb_estimates_phi)
        naive_sb_estimates[i] = mean(Y[D.==1]) - mean(Y[D.==0])
        ssac_estimates[i] = mean(ssac_estimates_phi)
        naive_outcome_model, naive_propensity_model = generate_naive_models(T, m, n_covariates, switch_period; mode="switch", est_ade=false, alpha=alpha, sd_error=sd_error, n_trees=n_trees, max_depth=max_depth, eps=eps, beta = beta)
        dml_naive_estimates[i] = naive_dml(D, X, Y, naive_outcome_model, naive_propensity_model, eps)

        dml_switchback_ses[i] = sqrt(max(dml4ssi_var(T, dml_switchback_estimates_phi, m, sb=true) / T, 0))
        plugin_ses[i] = sqrt(max(dml4ssi_var(T, plugin_sb_estimates_phi, m, sb=true) / T, 0))
        ipw_ses[i] = sqrt(max(ipw_var(D, X, Y, naive_propensity_model, eps) / T, 0))
        dml_naive_ses[i] = sqrt(max(ssac_var(T, dml_naive_estimates) / T, 0))
        ssac_ses[i] = sqrt(max(ssac_var(T, ssac_estimates_phi) / T, 0))
        # Compute switchback ses from Bojinov et al. (2021)
        num_switches = Int(floor(T / m))
        switchback_ses[i] = 8*mean(Y[m+1:2m])^2 + 8*mean(Y[T-m+1:T])^2
        for j in 1:(num_switches-3)
            switchback_ses[i] += 32*mean(Y[m*(j+1)+1:m*(j+2)] .* valid_obs[m*(j+1)+1:m*(j+2)])^2 
        end
        switchback_ses[i] = sqrt(switchback_ses[i] / (T-m)^2)



        D_global_control, X_global_control, H_global_control, Y_global_control = dgp(T, m, n_covariates, switch_period; mode = "global control", est_ade = false, alpha=alpha, sd_error=sd_error)
        D_global_treatment, X_global_treatment, H_global_treatment, Y_global_treatment= dgp(T, m, n_covariates, switch_period; mode = "global treatment", est_ade = false, alpha=alpha, sd_error=sd_error)
        true_gte[i] = mean(Y_global_treatment) - mean(Y_global_control)

    end
    
    return switchback_estimates, dml_switchback_estimates, naive_sb_estimates, true_gte, plugin_sb_estimates, dml_switchback_ses, dml_naive_estimates, switchback_ses, ipw_ses, plugin_ses, dml_naive_ses, ssac_estimates, ssac_ses
end

function run_coverage_and_width_simulations_ade(N, sample_sizes, m, n_covariates, switch_period, n_trees, max_depth, alpha_val, sd_error, eps)
    dml_avg_bias_n_ade = zeros(length(sample_sizes))
    ssac_avg_bias_n_ade = zeros(length(sample_sizes))
    naive_avg_bias_n_ade = zeros(length(sample_sizes))
    plugin_avg_bias_n_ade = zeros(length(sample_sizes))
    dml_naive_avg_bias_n_ade = zeros(length(sample_sizes))

    dml_coverages_n_ade = zeros(length(sample_sizes))
    ssac_coverages_n_ade = zeros(length(sample_sizes))
    naive_coverages_n_ade = zeros(length(sample_sizes))
    plugin_coverages_n_ade = zeros(length(sample_sizes))
    dml_naive_coverages_n_ade = zeros(length(sample_sizes))

    dml_widths_n_ade = zeros(length(sample_sizes))
    ssac_widths_n_ade = zeros(length(sample_sizes))
    naive_widths_n_ade = zeros(length(sample_sizes))
    plugin_widths_n_ade = zeros(length(sample_sizes))
    dml_naive_widths_n_ade = zeros(length(sample_sizes))

    dml_std_bias_n_ade = zeros(length(sample_sizes))
    ssac_std_bias_n_ade = zeros(length(sample_sizes))
    naive_std_bias_n_ade = zeros(length(sample_sizes))
    plugin_std_bias_n_ade = zeros(length(sample_sizes))
    dml_naive_std_bias_n_ade = zeros(length(sample_sizes))

    dml_sds_ade = zeros(length(sample_sizes))
    ssac_sds_ade = zeros(length(sample_sizes))
    naive_sds_ade = zeros(length(sample_sizes))
    plugin_sds_ade = zeros(length(sample_sizes))
    dml_naive_sds_ade = zeros(length(sample_sizes))

    dml_widths_sds_ade = zeros(length(sample_sizes))
    ssac_widths_sds_ade = zeros(length(sample_sizes))
    naive_widths_sds_ade = zeros(length(sample_sizes))
    plugin_widths_sds_ade = zeros(length(sample_sizes))
    dml_naive_widths_sds_ade = zeros(length(sample_sizes))

    psi_star = mean(true_direct_effects)
    alph = 0.05

    @showprogress 1 "ADE simulations over scaling T:" for (i, t) in enumerate(sample_sizes)
        
        dml_ade_estimates_i, ipw_ade_estimates_i, plugin_ade_estimates_i, true_direct_effects_i, dml_ade_ses_i, ssac_ses_i, dml_naive_estimates_i, ipw_ade_ses_i, plugin_ses_i, dml_naive_ses_i = run_simulations_ade(N, t, m, n_covariates, switch_period, n_trees, max_depth, alpha_val, sd_error, eps)

        dml_avg_bias_n_ade[i] = mean(dml_ade_estimates_i) - psi_star
        ssac_avg_bias_n_ade[i] = mean(ssac_ses_i) - psi_star
        naive_avg_bias_n_ade[i] = mean(ipw_ade_estimates_i) - psi_star
        plugin_avg_bias_n_ade[i] = mean(plugin_ade_estimates_i) - psi_star
        dml_naive_avg_bias_n_ade[i] = mean(dml_naive_estimates_i) - psi_star

        dml_ci_lower, dml_ci_upper = make_gaussian_cis(dml_ade_estimates_i, dml_ade_ses_i, alph)
        ssac_ci_lower, ssac_ci_upper = make_gaussian_cis(dml_ade_estimates_i, ssac_ses_i, alph)
        naive_ci_lower, naive_ci_upper = make_gaussian_cis(ipw_ade_estimates_i, ipw_ade_ses_i, alph)
        plugin_ci_lower, plugin_ci_upper = make_gaussian_cis(plugin_ade_estimates_i, plugin_ses_i, alph)
        dml_naive_ci_lower, dml_naive_ci_upper = make_gaussian_cis(dml_naive_estimates_i, dml_naive_ses_i, alph)
        
        dml_coverages_n_ade[i] = mean((dml_ci_lower .<= psi_star .<= dml_ci_upper))
        ssac_coverages_n_ade[i] = mean((ssac_ci_lower .<= psi_star .<= ssac_ci_upper))
        naive_coverages_n_ade[i] = mean((naive_ci_lower .<= psi_star .<= naive_ci_upper))
        plugin_coverages_n_ade[i] = mean((plugin_ci_lower .<= psi_star .<= plugin_ci_upper))
        dml_naive_coverages_n_ade[i] = mean((dml_naive_ci_lower .<= psi_star .<= dml_naive_ci_upper))

        dml_widths_n_ade[i] = mean(dml_ci_upper - dml_ci_lower)
        ssac_widths_n_ade[i] = mean(ssac_ci_upper - ssac_ci_lower)
        naive_widths_n_ade[i] = mean(naive_ci_upper - naive_ci_lower)
        plugin_widths_n_ade[i] = mean(plugin_ci_upper - plugin_ci_lower)
        dml_naive_widths_n_ade[i] = mean(dml_naive_ci_upper - dml_naive_ci_lower)

        dml_sds_ade[i] = std((dml_ci_lower .<= psi_star .<= dml_ci_upper)) / sqrt(N)
        ssac_sds_ade[i] = std((ssac_ci_lower .<= psi_star .<= ssac_ci_upper)) / sqrt(N)
        naive_sds_ade[i] = std((naive_ci_lower .<= psi_star .<= naive_ci_upper)) / sqrt(N)
        plugin_sds_ade[i] = std((plugin_ci_lower .<= psi_star .<= plugin_ci_upper)) / sqrt(N)
        dml_naive_sds_ade[i] = std((dml_naive_ci_lower .<= psi_star .<= dml_naive_ci_upper)) / sqrt(N)

        dml_widths_sds_ade[i] = std(dml_ci_upper - dml_ci_lower) / sqrt(N)
        ssac_widths_sds_ade[i] = std(ssac_ci_upper - ssac_ci_lower) / sqrt(N)
        naive_widths_sds_ade[i] = std(naive_ci_upper - naive_ci_lower) / sqrt(N)
        plugin_widths_sds_ade[i] = std(plugin_ci_upper - plugin_ci_lower) / sqrt(N)
        dml_naive_widths_sds_ade[i] = std(dml_naive_ci_upper - dml_naive_ci_lower) / sqrt(N)

        dml_std_bias_n_ade[i] = std(dml_ade_estimates_i)
        ssac_std_bias_n_ade[i] = std(ssac_ses_i)
        naive_std_bias_n_ade[i] = std(ipw_ade_estimates_i)
        plugin_std_bias_n_ade[i] = std(plugin_ade_estimates_i)
        dml_naive_std_bias_n_ade[i] = std(dml_naive_estimates_i)
    end
    return dml_avg_bias_n_ade, ssac_avg_bias_n_ade, naive_avg_bias_n_ade, plugin_avg_bias_n_ade, dml_naive_avg_bias_n_ade, dml_coverages_n_ade, ssac_coverages_n_ade, naive_coverages_n_ade, plugin_coverages_n_ade, dml_naive_coverages_n_ade, dml_widths_n_ade, ssac_widths_n_ade, naive_widths_n_ade, plugin_widths_n_ade, dml_naive_widths_n_ade, dml_sds_ade, ssac_sds_ade, naive_sds_ade, plugin_sds_ade, dml_naive_sds_ade, dml_widths_sds_ade, ssac_widths_sds_ade, naive_widths_sds_ade, plugin_widths_sds_ade, dml_naive_widths_sds_ade, dml_std_bias_n_ade, ssac_std_bias_n_ade, naive_std_bias_n_ade, plugin_std_bias_n_ade, dml_naive_std_bias_n_ade
end

function run_coverage_and_width_simulations_sb(N, sample_sizes, m, n_covariates, switch_period, n_trees, max_depth, alpha_val, sd_error, eps)
    dml_avg_bias_n_sb = zeros(length(sample_sizes))
    naive_avg_bias_n_sb = zeros(length(sample_sizes))
    plugin_avg_bias_n_sb = zeros(length(sample_sizes))
    dml_naive_avg_bias_n_sb = zeros(length(sample_sizes))
    switchback_avg_bias_n_sb = zeros(length(sample_sizes))
    ssac_avg_bias_n_sb = zeros(length(sample_sizes))

    dml_coverages_n_sb = zeros(length(sample_sizes))
    naive_coverages_n_sb = zeros(length(sample_sizes))
    plugin_coverages_n_sb = zeros(length(sample_sizes))
    dml_naive_coverages_n_sb = zeros(length(sample_sizes))
    switchback_coverages_n_sb = zeros(length(sample_sizes))
    ssac_coverages_n_sb = zeros(length(sample_sizes))

    dml_widths_n_sb = zeros(length(sample_sizes))
    naive_widths_n_sb = zeros(length(sample_sizes))
    plugin_widths_n_sb = zeros(length(sample_sizes))
    dml_naive_widths_n_sb = zeros(length(sample_sizes))
    switchback_widths_n_sb = zeros(length(sample_sizes))
    ssac_widths_n_sb = zeros(length(sample_sizes))

    dml_std_bias_n_sb = zeros(length(sample_sizes))
    naive_std_bias_n_sb = zeros(length(sample_sizes))
    plugin_std_bias_n_sb = zeros(length(sample_sizes))
    dml_naive_std_bias_n_sb = zeros(length(sample_sizes))
    switchback_std_bias_n_sb = zeros(length(sample_sizes))
    ssac_std_bias_n_sb = zeros(length(sample_sizes))

    dml_sds_n_sb = zeros(length(sample_sizes))
    naive_sds_n_sb = zeros(length(sample_sizes))
    plugin_sds_n_sb = zeros(length(sample_sizes))
    dml_naive_sds_n_sb = zeros(length(sample_sizes))
    switchback_sds_n_sb = zeros(length(sample_sizes))
    ssac_sds_n_sb = zeros(length(sample_sizes))

    dml_width_sds_n_sb = zeros(length(sample_sizes))
    naive_width_sds_n_sb = zeros(length(sample_sizes))
    plugin_width_sds_n_sb = zeros(length(sample_sizes))
    dml_naive_width_sds_n_sb = zeros(length(sample_sizes))
    switchback_width_sds_n_sb = zeros(length(sample_sizes))
    ssac_width_sds_n_sb = zeros(length(sample_sizes))

    psi_star_sb = mean(true_gte)
    alph = 0.05

    @showprogress 1 "Switchback simulations over scaling T:" for (i, t) in enumerate(sample_sizes)
        
        switchback_estimates_i, dml_switchback_estimates_i, naive_sb_estimates_i, true_gte_i, plugin_sb_estimates_i, dml_switchback_ses_i, dml_naive_estimates_sb_i, switchback_ses_i, ipw_ses_i, plugin_ses_i, dml_naive_ses_i, ssac_estimates_i, ssac_ses_i = run_simulations_sb(N, t, m, n_covariates, switch_period, n_trees, max_depth, alpha_val, sd_error, eps)
        
        dml_avg_bias_n_sb[i] = mean(dml_switchback_estimates_i) - psi_star_sb
        naive_avg_bias_n_sb[i] = mean(filter(!isnan, naive_sb_estimates_i)) - psi_star_sb
        plugin_avg_bias_n_sb[i] = mean(plugin_sb_estimates_i) - psi_star_sb
        dml_naive_avg_bias_n_sb[i] = mean(dml_naive_estimates_sb_i) - psi_star_sb
        switchback_avg_bias_n_sb[i] = mean(switchback_estimates_i) - psi_star_sb
        ssac_avg_bias_n_sb[i] = mean(ssac_estimates_i) - psi_star_sb

        
        dml_ci_lower, dml_ci_upper = make_gaussian_cis(dml_switchback_estimates_i, dml_switchback_ses_i, alph)
        naive_ci_lower, naive_ci_upper = make_gaussian_cis(naive_sb_estimates_i, ipw_ses_i, alph)
        plugin_ci_lower, plugin_ci_upper = make_gaussian_cis(plugin_sb_estimates_i, plugin_ses_i, alph)
        dml_naive_ci_lower, dml_naive_ci_upper = make_gaussian_cis(dml_naive_estimates_sb_i, dml_naive_ses_i, alph)
        switchback_ci_lower, switchback_ci_upper = make_gaussian_cis(switchback_estimates_i, switchback_ses_i, alph)
        sb_ssac_ci_lower, sb_ssac_ci_upper = make_gaussian_cis(ssac_estimates_i, ssac_ses_i, alph)

        dml_coverages_n_sb[i] = mean((dml_ci_lower .<= psi_star_sb .<= dml_ci_upper))
        switchback_coverages_n_sb[i] = mean((switchback_ci_lower .<= psi_star_sb .<= switchback_ci_upper))
        naive_coverages_n_sb[i] = mean((naive_ci_lower .<= psi_star_sb .<= naive_ci_upper))
        plugin_coverages_n_sb[i] = mean((plugin_ci_lower .<= psi_star_sb .<= plugin_ci_upper))
        dml_naive_coverages_n_sb[i] = mean((dml_naive_ci_lower .<= psi_star_sb .<= dml_naive_ci_upper))
        ssac_coverages_n_sb[i] = mean((sb_ssac_ci_lower .<= psi_star_sb .<= sb_ssac_ci_upper))

        dml_widths_n_sb[i] = mean(dml_ci_upper - dml_ci_lower)
        naive_widths_n_sb[i] = mean(filter(!isnan, naive_ci_upper - naive_ci_lower))
        plugin_widths_n_sb[i] = mean(plugin_ci_upper - plugin_ci_lower)
        dml_naive_widths_n_sb[i] = mean(dml_naive_ci_upper - dml_naive_ci_lower)
        switchback_widths_n_sb[i] = mean(switchback_ci_upper - switchback_ci_lower)
        ssac_widths_n_sb[i] = mean(sb_ssac_ci_upper - sb_ssac_ci_lower)

        dml_std_bias_n_sb[i] = std(dml_switchback_estimates_i)
        naive_std_bias_n_sb[i] = std(naive_sb_estimates_i)
        plugin_std_bias_n_sb[i] = std(plugin_sb_estimates_i)
        dml_naive_std_bias_n_sb[i] = std(dml_naive_estimates_sb_i)
        switchback_std_bias_n_sb[i] = std(switchback_estimates_i)
        ssac_std_bias_n_sb[i] = std(ssac_estimates_i)

        dml_sds_n_sb[i] = std((dml_ci_lower .<= psi_star_sb .<= dml_ci_upper)) / sqrt(N)
        naive_sds_n_sb[i] = std((naive_ci_lower .<= psi_star_sb .<= naive_ci_upper)) / sqrt(N)
        plugin_sds_n_sb[i] = std((plugin_ci_lower .<= psi_star_sb .<= plugin_ci_upper)) / sqrt(N)
        dml_naive_sds_n_sb[i] = std((dml_naive_ci_lower .<= psi_star_sb .<= dml_naive_ci_upper)) / sqrt(N)
        switchback_sds_n_sb[i] = std((switchback_ci_lower .<= psi_star_sb .<= switchback_ci_upper)) / sqrt(N)
        ssac_sds_n_sb[i] = std((sb_ssac_ci_lower .<= psi_star_sb .<= sb_ssac_ci_upper)) / sqrt(N)

        dml_width_sds_n_sb[i] = std(dml_ci_upper - dml_ci_lower) / sqrt(N)
        naive_width_sds_n_sb[i] = std(filter(!isnan, naive_ci_upper - naive_ci_lower)) / sqrt(N)
        plugin_width_sds_n_sb[i] = std(plugin_ci_upper - plugin_ci_lower) / sqrt(N)
        dml_naive_width_sds_n_sb[i] = std(dml_naive_ci_upper - dml_naive_ci_lower) / sqrt(N)
        switchback_width_sds_n_sb[i] = std(switchback_ci_upper - switchback_ci_lower) / sqrt(N)
        ssac_width_sds_n_sb[i] = std(sb_ssac_ci_upper - sb_ssac_ci_lower) / sqrt(N)


    end
    return dml_avg_bias_n_sb, naive_avg_bias_n_sb, plugin_avg_bias_n_sb, dml_naive_avg_bias_n_sb, switchback_avg_bias_n_sb, ssac_avg_bias_n_sb, dml_coverages_n_sb, naive_coverages_n_sb, plugin_coverages_n_sb, dml_naive_coverages_n_sb, switchback_coverages_n_sb, ssac_coverages_n_sb, dml_widths_n_sb, naive_widths_n_sb, plugin_widths_n_sb, dml_naive_widths_n_sb, switchback_widths_n_sb, ssac_widths_n_sb, dml_std_bias_n_sb, naive_std_bias_n_sb, plugin_std_bias_n_sb, dml_naive_std_bias_n_sb, switchback_std_bias_n_sb, ssac_std_bias_n_sb, dml_sds_n_sb, naive_sds_n_sb, plugin_sds_n_sb, dml_naive_sds_n_sb, switchback_sds_n_sb, ssac_sds_n_sb, dml_width_sds_n_sb, naive_width_sds_n_sb, plugin_width_sds_n_sb, dml_naive_width_sds_n_sb, switchback_width_sds_n_sb, ssac_width_sds_n_sb
end