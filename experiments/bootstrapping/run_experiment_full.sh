#!/bin/sh
dataset=$1
# first do the easier
trcont=0.00
julia ../run_experiment_f1.jl $dataset full_bootstrapping_contaminated-$trcont $trcont --mc-volume-iters=1000 --models kNN LOF OCSVM IF --robust-measures
trcont=0.01
julia ../run_experiment_f1.jl $dataset full_bootstrapping_contaminated-$trcont $trcont --mc-volume-iters=1000 --models kNN LOF OCSVM IF --robust-measures
trcont=0.05
julia ../run_experiment_f1.jl $dataset full_bootstrapping_contaminated-$trcont $trcont --mc-volume-iters=1000 --models kNN LOF OCSVM IF --robust-measures
