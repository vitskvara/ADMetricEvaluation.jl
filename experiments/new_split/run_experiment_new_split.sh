#!/bin/sh

# cat data_list.txt | xargs -n 1 -P 48  julia ./run_one_beta_alternative.sh

dataset=$1
normal_class=$2
anomalous_class=$3
/compass/home/skvara/julia-1.3.1/bin/julia run_experiment_new_split.jl $dataset $normal_class $anomalous_class run1