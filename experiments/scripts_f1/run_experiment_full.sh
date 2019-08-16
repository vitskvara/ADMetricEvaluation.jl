#!/bin/sh
dataset=$1
# first do the easier
trcont=0.00
julia ../run_experiment_f1.jl $dataset full_f1_contaminated-$trcont $trcont --mc-volume-iters=1 --models kNN LOF OCSVM
trcont=0.01
julia ../run_experiment_f1.jl $dataset full_f1_contaminated-$trcont $trcont --mc-volume-iters=1 --models kNN LOF OCSVM
trcont=0.05
julia ../run_experiment_f1.jl $dataset full_f1_contaminated-$trcont $trcont --mc-volume-iters=1 --models kNN LOF OCSVM
# then do IF that need MC volume computation and takes a long time
trcont=0.00
julia ../run_experiment_f1.jl $dataset full_f1_contaminated-$trcont $trcont --models IF
trcont=0.01
julia ../run_experiment_f1.jl $dataset full_f1_contaminated-$trcont $trcont --models IF
trcont=0.05
julia ../run_experiment_f1.jl $dataset full_f1_contaminated-$trcont $trcont --models IF
