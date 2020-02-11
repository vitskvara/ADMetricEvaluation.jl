# Experiments

Run a single experiment via the `run_experiment_f1.jl` script. An example run is 

```bash
julia run_experiment_f1.jl abalone path/to/save 0.01 --models IF kNN
```

You can invoke help on the command line arguments by running `julia run_experiment_f1.jl --help`. 

Running multiple experiments in parallel:

`cat umap_list | xargs -n 1 -P 32 ./run_experiment_umap.sh`

# Evaluation

To produce ranking of models based on individual measures, run script `eval_paper/rank_tables.jl`.

To compute correlation between measures, run `eval_paper/correlation_tables.jl`.

To compare relative losses in model performance based on the base selection measure, run `eval_paper/measure_comaprison_tables.jl`.


To compare how resistant a measure is to dicrepancy in test and train data set, run `eval_paper/multiclass_comparison.jl`.