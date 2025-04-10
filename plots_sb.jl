using ColorBrewer
using Measures
using ProgressMeter
using Plots

gr()
default(fontfamily="Times")

include("dgp.jl")
include("estimators.jl")
include("ml.jl")
include("simulations.jl")

N = 100
T = 1000
m = 5
n_covariates = 1
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
    df = CSV.read("switchback_simulation_results.csv", DataFrame)
    switchback_estimates = df.switchback_estimates
    dml_switchback_estimates = df.dml_switchback_estimates
    naive_sb_estimates = df.naive_sb_estimates
    true_gte = df.true_gte
    plugin_sb_estimates = df.plugin_sb_estimates
    dml_switchback_ses = df.dml_switchback_ses
    dml_naive_estimates = df.dml_naive_estimates
    switchback_ses = df.switchback_ses
    ipw_ses = df.ipw_ses
    plugin_ses = df.plugin_ses
    dml_naive_ses = df.dml_naive_ses
else
    switchback_estimates, dml_switchback_estimates, naive_sb_estimates, true_gte, plugin_sb_estimates, dml_switchback_ses, dml_naive_estimates_sb, switchback_ses, ipw_ses, plugin_ses, dml_naive_ses, ssac_estimates, ssac_ses = run_simulations_sb(N, T, m, n_covariates, switch_period, n_trees, max_depth, alpha_val, sd_error, eps)
end

# Create histogram of estimates
colors_sb = ColorBrewer.palette("Set1", 9)
psi_star_sb = mean(true_gte)
switchback_lower = minimum(switchback_estimates)-psi_star_sb
switchback_upper = maximum(switchback_estimates)-psi_star_sb
dml_switchback_lower = minimum(dml_switchback_estimates)-psi_star_sb
dml_switchback_upper = maximum(dml_switchback_estimates)-psi_star_sb
bin_lower = min(switchback_lower, dml_switchback_lower)
bin_upper = max(switchback_upper, dml_switchback_upper)
incr = (bin_upper - bin_lower) / n_bins
bins = bin_lower:incr:bin_upper

histogram(dml_switchback_estimates .- psi_star_sb, 
          alpha=opacity, 
          label=L"\psi(W_{1:T}; \hat \eta)",
          normalize=true,
          bins=bins,
          legendfontsize=LEGEND_SIZE,
          tickfontsize=TICK_SIZE,
          legend=:outertop,
          legend_columns=2,
          foreground_color_legend = nothing,
          color=colors_sb[1],
          size=sz,
          left_margin=10mm, 
          bottom_margin=margin,
          yticks=([], []),
          legend_markerstrokecolor = :transparent)
histogram!(switchback_estimates .- psi_star_sb, 
         alpha=0.5,
         label=L"\psi^{\mathrm{SB}}(W_{1:T})", 
         normalize=true,
         bins=bins,
         legendfontsize=LEGEND_SIZE,
         tickfontsize=TICK_SIZE,
         color=colors_sb[6])
vline!([0], 
       linestyle=:dash, 
       color=:grey, 
       label=nothing,
       linewidth=3)
xlabel!("Bias")
ylabel!("Density")

savefig("switchback.svg");

# Create histogram of estimates
pct = 0.01
psi_star_sb = mean(true_gte)
naive_sb_lower = quantile(naive_sb_estimates, pct) - psi_star_sb
naive_sb_upper = quantile(naive_sb_estimates, 1-pct) - psi_star_sb
plugin_sb_lower = quantile(plugin_sb_estimates, pct) - psi_star_sb
plugin_sb_upper = quantile(plugin_sb_estimates, 1-pct) - psi_star_sb
dml_sb_lower = quantile(dml_switchback_estimates, pct) - psi_star_sb
dml_sb_upper = quantile(dml_switchback_estimates, 1-pct) - psi_star_sb
ssac_lower = quantile(ssac_estimates, pct) - psi_star_sb
ssac_upper = quantile(ssac_estimates, 1-pct) - psi_star_sb
bin_lower = min(naive_sb_lower,plugin_sb_lower,dml_sb_lower,ssac_lower)
bin_upper = max(naive_sb_upper,plugin_sb_upper,dml_sb_upper,ssac_upper)
incr = (bin_upper - bin_lower) / n_bins
bins = bin_lower:incr:bin_upper

plot(legendfontsize=LEGEND_SIZE,
       tickfontsize=TICK_SIZE,
       legend=:outertop,
       foreground_color_legend = nothing,
       legend_columns=3,
       size=sz,
       left_margin=10mm, 
       bottom_margin=margin,
       yticks=([], []), legend_markerstrokecolor=:transparent)

ests_sb = [dml_switchback_estimates, naive_sb_estimates, plugin_sb_estimates, dml_naive_estimates_sb, ssac_estimates]
biases_sb = [ests_sb[i] .- psi_star_sb for i in 1:length(ests_sb)]
labels_sb = [L"\psi(W_{1:T}; \hat \eta)", L"\psi^{\mathrm{HT}}(W_{1:T}; \hat \eta)", L"\psi^{\mathrm{pi}}(W_{1:T}; \hat \eta)", L"\psi^{\mathrm{DML-N}}(W_{1:T}; \eta')", L"\psi^{\mathrm{SSAC}}(W_{1:T}; \hat \eta)"]

for i in 1:length(biases_sb)
    histogram!(biases_sb[i],
               color=colors_sb[i],
               alpha=opacity,
               label=labels_sb[i],
               normalize=true,
               bins=bins)
end

vline!([0], 
       linestyle=:dash, 
       color=:grey, 
       label=nothing,
       linewidth=3)

xlabel!("Bias")
ylabel!("Density")

savefig("switchback_vs_naive.svg");

N = 10
T = 1000
m = 5
n_covariates = 1
switch_period = 10
sd_error = 10

alpha_val = 2
eps = 0.1

n_trees = 100
max_depth = 10


sample_sizes = Int.(round.(exp10.(range(2, 4, length=10))))



if use_saved
    df = CSV.read("switchback_coverage_and_width_results.csv", DataFrame)
    dml_avg_bias_n_sb = df.dml_avg_bias_n_sb
    naive_avg_bias_n_sb = df.naive_avg_bias_n_sb
    plugin_avg_bias_n_sb = df.plugin_avg_bias_n_sb
    dml_naive_avg_bias_n_sb = df.dml_naive_avg_bias_n_sb
    switchback_avg_bias_n_sb = df.switchback_avg_bias_n_sb
    ssac_avg_bias_n_sb = df.ssac_avg_bias_n_sb

    dml_coverages_n_sb = df.dml_coverages_n_sb
    naive_coverages_n_sb = df.naive_coverages_n_sb
    plugin_coverages_n_sb = df.plugin_coverages_n_sb
    dml_naive_coverages_n_sb = df.dml_naive_coverages_n_sb
    switchback_coverages_n_sb = df.switchback_coverages_n_sb
    ssac_coverages_n_sb = df.ssac_coverages_n_sb

    dml_widths_n_sb = df.dml_widths_n_sb    
    naive_widths_n_sb = df.naive_widths_n_sb
    plugin_widths_n_sb = df.plugin_widths_n_sb
    dml_naive_widths_n_sb = df.dml_naive_widths_n_sb
    switchback_widths_n_sb = df.switchback_widths_n_sb
    ssac_widths_n_sb = df.ssac_widths_n_sb

    dml_sds_n_sb = df.dml_sds_n_sb
    naive_sds_n_sb = df.naive_sds_n_sb
    plugin_sds_n_sb = df.plugin_sds_n_sb
    dml_naive_sds_n_sb = df.dml_naive_sds_n_sb
    switchback_sds_n_sb = df.switchback_sds_n_sb
    ssac_sds_n_sb = df.ssac_sds_n_sb

    dml_width_sds_n_sb = df.dml_width_sds_n_sb
    naive_width_sds_n_sb = df.naive_width_sds_n_sb
    plugin_width_sds_n_sb = df.plugin_width_sds_n_sb
    dml_naive_width_sds_n_sb = df.dml_naive_width_sds_n_sb
    switchback_width_sds_n_sb = df.switchback_width_sds_n_sb
    ssac_width_sds_n_sb = df.ssac_width_sds_n_sb

    dml_std_bias_n_sb = df.dml_std_bias_n_sb
    naive_std_bias_n_sb = df.naive_std_bias_n_sb
    plugin_std_bias_n_sb = df.plugin_std_bias_n_sb
    dml_naive_std_bias_n_sb = df.dml_naive_std_bias_n_sb
    switchback_std_bias_n_sb = df.switchback_std_bias_n_sb
    ssac_std_bias_n_sb = df.ssac_std_bias_n_sb
    
else
    dml_avg_bias_n_sb, naive_avg_bias_n_sb, plugin_avg_bias_n_sb, dml_naive_avg_bias_n_sb, switchback_avg_bias_n_sb, ssac_avg_bias_n_sb, dml_coverages_n_sb, naive_coverages_n_sb, plugin_coverages_n_sb, dml_naive_coverages_n_sb, switchback_coverages_n_sb, ssac_coverages_n_sb, dml_widths_n_sb, naive_widths_n_sb, plugin_widths_n_sb, dml_naive_widths_n_sb, switchback_widths_n_sb, ssac_widths_n_sb, dml_sds_n_sb, naive_sds_n_sb, plugin_sds_n_sb, dml_naive_sds_n_sb, switchback_sds_n_sb, ssac_sds_n_sb, dml_width_sds_n_sb, naive_width_sds_n_sb, plugin_width_sds_n_sb, dml_naive_width_sds_n_sb, switchback_width_sds_n_sb, ssac_width_sds_n_sb = run_coverage_and_width_simulations_sb(N, sample_sizes, m, n_covariates, switch_period, n_trees, max_depth, alpha_val, sd_error, eps)
    results_df = DataFrame(
        dml_avg_bias_n_sb = dml_avg_bias_n_sb,
        naive_avg_bias_n_sb = naive_avg_bias_n_sb,
        plugin_avg_bias_n_sb = plugin_avg_bias_n_sb,
        dml_naive_avg_bias_n_sb = dml_naive_avg_bias_n_sb,
        switchback_avg_bias_n_sb = switchback_avg_bias_n_sb,
        ssac_avg_bias_n_sb = ssac_avg_bias_n_sb,

        dml_coverages_n_sb = dml_coverages_n_sb,
        naive_coverages_n_sb = naive_coverages_n_sb,
        plugin_coverages_n_sb = plugin_coverages_n_sb,
        dml_naive_coverages_n_sb = dml_naive_coverages_n_sb,
        switchback_coverages_n_sb = switchback_coverages_n_sb,
        ssac_coverages_n_sb = ssac_coverages_n_sb,

        dml_widths_n_sb = dml_widths_n_sb,
        naive_widths_n_sb = naive_widths_n_sb,
        plugin_widths_n_sb = plugin_widths_n_sb,
        dml_naive_widths_n_sb = dml_naive_widths_n_sb,
        switchback_widths_n_sb = switchback_widths_n_sb,
        ssac_widths_n_sb = ssac_widths_n_sb,

        dml_sds_n_sb = dml_sds_n_sb,
        naive_sds_n_sb = naive_sds_n_sb,
        plugin_sds_n_sb = plugin_sds_n_sb,
        dml_naive_sds_n_sb = dml_naive_sds_n_sb,
        switchback_sds_n_sb = switchback_sds_n_sb,
        ssac_sds_n_sb = ssac_sds_n_sb,

        dml_width_sds_n_sb = dml_width_sds_n_sb,
        naive_width_sds_n_sb = naive_width_sds_n_sb,
        plugin_width_sds_n_sb = plugin_width_sds_n_sb,
        dml_naive_width_sds_n_sb = dml_naive_width_sds_n_sb,
        switchback_width_sds_n_sb = switchback_width_sds_n_sb,
        ssac_width_sds_n_sb = ssac_width_sds_n_sb,

        dml_std_bias_n_sb = dml_std_bias_n_sb,
        naive_std_bias_n_sb = naive_std_bias_n_sb,
        plugin_std_bias_n_sb = plugin_std_bias_n_sb,
        dml_naive_std_bias_n_sb = dml_naive_std_bias_n_sb,
        switchback_width_sds_n_sb = switchback_width_sds_n_sb,
        ssac_width_sds_n_sb = ssac_width_sds_n_sb
    )
    CSV.write("switchback_coverage_and_width_results.csv", results_df)
        
end

plot(legend=:outertop,
     legend_columns=3,
     legendfontsize=LEGEND_SIZE,
     tickfontsize=TICK_SIZE,
     xlabel=L"$T$",
     ylabel="Coverage", 
     foreground_color_legend = nothing,
     xscale=:log10,
     margin=margin,
     size=(600, 400))

sb_coverages = [dml_coverages_n_sb, naive_coverages_n_sb, plugin_coverages_n_sb, dml_naive_coverages_n_sb, ssac_coverages_n_sb, switchback_coverages_n_sb]
sb_sds = [dml_sds_n_sb, naive_sds_n_sb, plugin_sds_n_sb, dml_naive_sds_n_sb, ssac_sds_n_sb, switchback_sds_n_sb]
labels = ["DML4SSI", "HT", "Plug-in", "DML-N", "SSAC", "SB"]

for i in 1:length(sb_coverages)
    plot!(sample_sizes, sb_coverages[i],
          ribbon=sb_sds[i],
          fillalpha=0.3,
          label=labels[i],
          linewidth=2,
          color=colors_sb[i])
end

plot!(sample_sizes, fill(0.95, length(sample_sizes)),
      label=nothing,
      linewidth=2,
      linestyle=:dash,
      color=:grey)

xticks!(10 .^ [2,2.5,3,3.5,4])

      
savefig("sb_coverage_vs_sample_size.svg")

plot(legend=:outertop,
     legend_columns=3,
     legendfontsize=LEGEND_SIZE,
     tickfontsize=TICK_SIZE,
     xlabel=L"$T$",
     ylabel="Width", 
     foreground_color_legend = nothing,
     xscale=:log10,
     margin=margin,
     size=(600, 400))

sb_widths = [dml_widths_n_sb, naive_widths_n_sb, plugin_widths_n_sb, dml_naive_widths_n_sb, ssac_widths_n_sb, switchback_widths_n_sb]
sb_width_sds = [dml_width_sds_n_sb, naive_width_sds_n_sb, plugin_width_sds_n_sb, dml_naive_width_sds_n_sb, ssac_width_sds_n_sb, switchback_width_sds_n_sb]
labels = ["DML4SSI", "HT", "Plug-in", "DML-N", "SSAC", "SB"]

for i in 1:length(sb_widths)
    plot!(sample_sizes, sb_widths[i],
          ribbon=sb_width_sds[i],
          fillalpha=0.3,
          label=labels[i],
          linewidth=2,
          color=colors_sb[i])
end



xticks!(10 .^ [2,2.5,3,3.5,4])
savefig("sb_width_vs_sample_size.svg")
