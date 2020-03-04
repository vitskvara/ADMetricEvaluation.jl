#!/bin/sh
dataset=$1
trcont=0.00
/compass/home/skvara/julia-1.3.1/bin/julia ../run_experiment_f1.jl $dataset umap_discriminability_contaminated-${trcont}_pre $trcont --umap --discriminability --nexperiments=50 --models kNN OCSVM LOF IF
