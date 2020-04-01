#!/bin/sh
dataset=$1
trcont=0.00
/compass/home/skvara/julia-1.3.1/bin/julia ../run_experiment_beta.jl $dataset umap_beta_contaminated-$trcont $trcont --umap
