#!/bin/sh
dataset="yeast"
trcont=0.00
julia ../run_experiment_f1.jl $dataset full_ci_contaminated-$trcont $trcont --umap --mc-volume-iters=1000 --models kNN LOF OCSVM IF
