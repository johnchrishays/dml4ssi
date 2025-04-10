# Simulations for "Double Machine Learning for Causal Inference under Shared-State Interference"

This repository contains the code for the simulations in the paper "Double Machine Learning for Causal Inference under Shared-State Interference" by [Chris Hays](https://johnchrishays.com), and [Manish Raghavan](https://mraghavan.github.io).

## Setup

In the directory, run
```
julia
julia> ]
pkg> activate .
pkg> instantiate
```
to install dependencies.

## Generating figures

Run
```
julia --project=. plots_ade.jl
```
and
```
julia --project=. plots_sb.jl
```
to reproduce the figures in the paper.



