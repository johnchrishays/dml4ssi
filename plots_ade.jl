using ColorBrewer
using CSV
using DataFrames
using Measures
using ProgressMeter
using Plots


gr()
default(fontfamily="Times")

include("dgp.jl")
include("estimators.jl")
include("ml.jl")
include("simulations.jl")

N = 50000
T = 1000
m = 5
n_covariates = 10
switch_period = 10
sd_error = 10

n_trees = 100
max_depth = 10
alpha_val = 2
eps = 0.1

use_saved = length(ARGS) > 0 
if use_saved 
    @assert ARGS[1] == "--use_saved"
end

if use_saved
    df = CSV.read("ade_simulation_results.csv", DataFrame)
    dml_ade_estimates = df.dml_ade
    naive_ade_estimates = df.naive_ade
    plugin_ade_estimates = df.plugin_ade
    true_direct_effects = df.true_effects
    dml_ade_ses = df.dml_ade_se
    ssac_ses = df.ssac_se
    dml_naive_estimates = df.dml_naive
    ipw_ade_ses = df.ipw_ade_se
    plugin_ses = df.plugin_se
    dml_naive_ses = df.dml_naive_se
else
    dml_ade_estimates, naive_ade_estimates, plugin_ade_estimates, true_direct_effects, dml_ade_ses, ssac_ses, dml_naive_estimates, ipw_ade_ses, plugin_ses, dml_naive_ses = run_simulations_ade(N, T, m, n_covariates, switch_period, n_trees, max_depth, alpha_val, sd_error, eps)
    results_df = DataFrame(
        dml_ade = dml_ade_estimates,
        naive_ade = naive_ade_estimates, 
        plugin_ade = plugin_ade_estimates,
        true_effects = true_direct_effects,
        dml_ade_se = dml_ade_ses,
        ssac_se = ssac_ses,
        dml_naive = dml_naive_estimates,
        ipw_ade_se = ipw_ade_ses,
        plugin_se = plugin_ses,
        dml_naive_se = dml_naive_ses
    )

    CSV.write("ade_simulation_results.csv", results_df)
end

psi_star = mean(true_direct_effects)
dml_naive_lower = quantile(dml_naive_estimates, 0.075) - psi_star
dml_naive_upper = quantile(dml_naive_estimates, 0.95) - psi_star
dml_ade_lower = quantile(dml_ade_estimates, pct) - psi_star
dml_ade_upper = quantile(dml_ade_estimates, 1-pct) - psi_star
naive_ade_lower = quantile(naive_ade_estimates, pct) - psi_star
naive_ade_upper = quantile(naive_ade_estimates, 0.995) - psi_star
plugin_ade_lower = quantile(plugin_ade_estimates, pct) - psi_star
plugin_ade_upper = quantile(plugin_ade_estimates, 1-pct) - psi_star

bin_lower = min(dml_ade_lower, naive_ade_lower, plugin_ade_lower, dml_naive_lower)
bin_upper = max(dml_ade_upper, naive_ade_upper, plugin_ade_upper, dml_naive_upper)
incr = (bin_upper - bin_lower) / n_bins
bins = bin_lower:incr:bin_upper

colors_ade = ColorBrewer.palette("Set1", 9)


p = plot(legendfontsize=LEGEND_SIZE,
         tickfontsize=TICK_SIZE,
         label=L"\psi(W_{1:T}; \hat \eta)",
         normalize=true,
         legend=:outertop,
         legend_columns=2,
         foreground_color_legend = nothing,
         bins=bins,
         size=sz,
         left_margin=10mm, 
         bottom_margin=margin,
         yticks = ([], []))
plegend = plot(legend=:outertop,
               legend_columns=4,
               legendfontsize=LEGEND_SIZE,
               tickfontsize=TICK_SIZE,
               foreground_color_legend = nothing,
               size=legendsz)

ests_ade = [dml_ade_estimates, naive_ade_estimates, plugin_ade_estimates, dml_naive_estimates]
biases_ade = [est .- psi_star for est in ests_ade]

labels_ade = [L"\psi(W_{1:T}; \hat \eta)", L"\psi^{\mathrm{HT}}(W_{1:T}; \hat \eta)", L"\psi^{\mathrm{pi}}(W_{1:T}; \hat \eta)", L"\psi^{\mathrm{DML-N}}(W_{1:T}; \hat \eta')"]

for i in 1:length(biases_ade)
       histogram!(p, biases_ade[i],
               alpha=opacity,
               label=labels_ade[i],
               normalize=true,
               bins=bins,
               color=colors_ade[i])
       plot!(plegend, sin,
               alpha=opacity,
               label=labels_ade[i],
               normalize=true,
               color=colors_ade[i])
end
vline!(p, [0], 
       linestyle=:dash, 
       color=:grey, 
       label=nothing,
       linewidth=3)
xlabel!("Bias")
ylabel!("Density")
xlims!(bin_lower, bin_upper)
savefig("direct_effect.svg");

N = 1000
sample_sizes = Int.(round.(exp10.(range(2, 4, length=10))))
if use_saved
    df = CSV.read("ade_coverage_and_width_results.csv", DataFrame)
    
    dml_avg_bias_n_ade = df.dml_avg_bias_n_ade
    ssac_avg_bias_n_ade = df.ssac_avg_bias_n_ade
    naive_avg_bias_n_ade = df.naive_avg_bias_n_ade
    plugin_avg_bias_n_ade = df.plugin_avg_bias_n_ade
    dml_naive_avg_bias_n_ade = df.dml_naive_avg_bias_n_ade

    dml_coverages_n_ade = df.dml_coverages_n_ade
    ssac_coverages_n_ade = df.ssac_coverages_n_ade
    naive_coverages_n_ade = df.naive_coverages_n_ade
    plugin_coverages_n_ade = df.plugin_coverages_n_ade
    dml_naive_coverages_n_ade = df.dml_naive_coverages_n_ade

    dml_widths_n_ade = df.dml_widths_n_ade  
    ssac_widths_n_ade = df.ssac_widths_n_ade
    naive_widths_n_ade = df.naive_widths_n_ade
    plugin_widths_n_ade = df.plugin_widths_n_ade
    dml_naive_widths_n_ade = df.dml_naive_widths_n_ade

    dml_sds_ade = df.dml_sds_ade
    ssac_sds_ade = df.ssac_sds_ade
    naive_sds_ade = df.naive_sds_ade
    plugin_sds_ade = df.plugin_sds_ade
    dml_naive_sds_ade = df.dml_naive_sds_ade

    dml_widths_sds_ade = df.dml_widths_sds_ade  
    ssac_widths_sds_ade = df.ssac_widths_sds_ade
    naive_widths_sds_ade = df.naive_widths_sds_ade
    plugin_widths_sds_ade = df.plugin_widths_sds_ade
    dml_naive_widths_sds_ade = df.dml_naive_widths_sds_ade

    dml_std_bias_n_ade = df.dml_std_bias_n_ade
    ssac_std_bias_n_ade = df.ssac_std_bias_n_ade
    naive_std_bias_n_ade = df.naive_std_bias_n_ade
    plugin_std_bias_n_ade = df.plugin_std_bias_n_ade
    dml_naive_std_bias_n_ade = df.dml_naive_std_bias_n_ade
    
else
    dml_avg_bias_n_ade, ssac_avg_bias_n_ade, naive_avg_bias_n_ade, plugin_avg_bias_n_ade, dml_naive_avg_bias_n_ade, dml_coverages_n_ade, ssac_coverages_n_ade, naive_coverages_n_ade, plugin_coverages_n_ade, dml_naive_coverages_n_ade, dml_widths_n_ade, ssac_widths_n_ade, naive_widths_n_ade, plugin_widths_n_ade, dml_naive_widths_n_ade, dml_sds_ade, ssac_sds_ade, naive_sds_ade, plugin_sds_ade, dml_naive_sds_ade, dml_widths_sds_ade, ssac_widths_sds_ade, naive_widths_sds_ade, plugin_widths_sds_ade, dml_naive_widths_sds_ade, dml_std_bias_n_ade, ssac_std_bias_n_ade, naive_std_bias_n_ade, plugin_std_bias_n_ade, dml_naive_std_bias_n_ade = run_coverage_and_width_simulations_ade(N, sample_sizes, m, n_covariates, switch_period, n_trees, max_depth, alpha_val, sd_error, eps)
    results_df = DataFrame(

        dml_avg_bias_n_ade = dml_avg_bias_n_ade,
        ssac_avg_bias_n_ade = ssac_avg_bias_n_ade,
        naive_avg_bias_n_ade = naive_avg_bias_n_ade,
        plugin_avg_bias_n_ade = plugin_avg_bias_n_ade,
        dml_naive_avg_bias_n_ade = dml_naive_avg_bias_n_ade,

        dml_coverages_n_ade = dml_coverages_n_ade,
        ssac_coverages_n_ade = ssac_coverages_n_ade,
        naive_coverages_n_ade = naive_coverages_n_ade,
        plugin_coverages_n_ade = plugin_coverages_n_ade,
        dml_naive_coverages_n_ade = dml_naive_coverages_n_ade,

        dml_widths_n_ade = dml_widths_n_ade,
        ssac_widths_n_ade = ssac_widths_n_ade,
        naive_widths_n_ade = naive_widths_n_ade,
        plugin_widths_n_ade = plugin_widths_n_ade,
        dml_naive_widths_n_ade = dml_naive_widths_n_ade,

        dml_sds_ade = dml_sds_ade,
        ssac_sds_ade = ssac_sds_ade,
        naive_sds_ade = naive_sds_ade,
        plugin_sds_ade = plugin_sds_ade,
        dml_naive_sds_ade = dml_naive_sds_ade,

        dml_widths_sds_ade = dml_widths_sds_ade,
        ssac_widths_sds_ade = ssac_widths_sds_ade,
        naive_widths_sds_ade = naive_widths_sds_ade,
        plugin_widths_sds_ade = plugin_widths_sds_ade,
        dml_naive_widths_sds_ade = dml_naive_widths_sds_ade,

        dml_std_bias_n_ade = dml_std_bias_n_ade,
        ssac_std_bias_n_ade = ssac_std_bias_n_ade,
        naive_std_bias_n_ade = naive_std_bias_n_ade,
        plugin_std_bias_n_ade = plugin_std_bias_n_ade,
        dml_naive_std_bias_n_ade = dml_naive_std_bias_n_ade,

    )
    CSV.write("ade_coverage_and_width_results.csv", results_df)
end

margin = 5mm
max_ind = 10 
plot(legend=:outertop,
     legend_columns=3,
     legendfontsize=LEGEND_SIZE,
     tickfontsize=TICK_SIZE,
     xlabel=L"$T$",
     ylabel="Coverage", 
     xscale=:log10,
     margin=margin,
     foreground_color_legend = nothing,
     size=(600, 400))

coverages = [dml_coverages_n_ade, naive_coverages_n_ade, plugin_coverages_n_ade, dml_naive_coverages_n_ade, ssac_coverages_n_ade]
labels = ["DML4SSI", "HT", "Plug-in", "DML-N", "SSAC"]
sds = [dml_sds_ade, naive_sds_ade, plugin_sds_ade, dml_naive_sds_ade, ssac_sds_ade]

for i in 1:length(coverages)
    plot!(sample_sizes[1:max_ind], coverages[i][1:max_ind],
          ribbon=sds[i][1:max_ind],
          fillalpha=0.3,
          label=labels[i], 
          linewidth=2,
          color=colors_ade[i])
end
plot!(sample_sizes[1:max_ind], fill(0.95, length(sample_sizes[1:max_ind])),
      label=nothing,
      linewidth=2,
      linestyle=:dash,
      color=:grey)
      
xticks!(10 .^ [2,2.5,3,3.5,4])
savefig("coverage_vs_sample_size.svg")

plot(legend=:outertop,
     legend_columns=3,
     legendfontsize=LEGEND_SIZE,
     tickfontsize=TICK_SIZE,
     xlabel=L"$T$",
     ylabel="Width", 
     linewidth=2,
     foreground_color_legend = nothing,
     xscale=:log10,
     margin=margin,
     size=(600, 400))

widths = [dml_widths_n_ade, naive_widths_n_ade, plugin_widths_n_ade, dml_naive_widths_n_ade, ssac_widths_n_ade]
labels = ["DML4SSI", "HT", "Plug-in", "DML-N", "SSAC"]
sds = [dml_widths_sds_ade, ssac_widths_sds_ade, naive_widths_sds_ade, plugin_widths_sds_ade, dml_naive_widths_sds_ade]
for i in 1:length(widths)
    plot!(sample_sizes[1:max_ind], widths[i][1:max_ind],
          ribbon=sds[i][1:max_ind],
          fillalpha=0.3,
          label=labels[i], 
          linewidth=2,
          color=colors_ade[i])
end


xticks!(10 .^ [2,2.5,3,3.5,4])
savefig("width_vs_sample_size.svg")
