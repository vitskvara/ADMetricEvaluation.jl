#!/bin/sh
dataset=$1
# first do the easier
trcont=0.00
/compass/home/skvara/julia-1.3.1/bin/julia ../run_experiment_f1.jl $dataset full_discriminability_contaminated-${trcont}_pre $trcont --discriminability --nexperiments=50 --models kNN LOF OCSVM IF
trcont=0.01
/compass/home/skvara/julia-1.3.1/bin/julia ../run_experiment_f1.jl $dataset full_discriminability_contaminated-${trcont}_pre $trcont --discriminability --nexperiments=50 --models kNN LOF OCSVM IF
trcont=0.05
/compass/home/skvara/julia-1.3.1/bin/julia ../run_experiment_f1.jl $dataset full_discriminability_contaminated-${trcont}_pre $trcont --discriminability --nexperiments=50 --models kNN LOF OCSVM IF
