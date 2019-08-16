#!/bin/sh
dataset=$1
trcont=0.00
julia ../run_experiment_f1.jl $dataset umap_f1_contaminated-$trcont $trcont --umap --mc-volume-iters=10 --models kNN LOF OCSVM
julia ../run_experiment_f1.jl $dataset umap_f1_contaminated-$trcont $trcont --umap --models IF
