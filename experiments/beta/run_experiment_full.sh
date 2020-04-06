#!/bin/sh
dataset=$1
# first do the easier
trcont=0.00
/compass/home/skvara/julia-1.3.1/bin/julia ../run_experiment_beta.jl $dataset full_beta_contaminated-$trcont $trcont 
trcont=0.01
/compass/home/skvara/julia-1.3.1/bin/julia ../run_experiment_beta.jl $dataset full_beta_contaminated-$trcont $trcont
trcont=0.05
/compass/home/skvara/julia-1.3.1/bin/julia ../run_experiment_beta.jl $dataset full_beta_contaminated-$trcont $trcont
