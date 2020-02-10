#!/bin/sh
dataset=$1
trcont=0.00
julia ../run_experiment_f1.jl $dataset umap_bootstrapping_contaminated-$trcont $trcont --umap --robust-measures --mc-volume-iters=1000 --models kNN OCSVM LOF IF
