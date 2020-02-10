# Experiments

Run a single experiment via the `run_experiment_f1.jl` script. An example run is 

```bash
julia run_experiment_f1.jl abalone path/to/save 0.01 --models IF kNN
```

You can invoke help on the command line arguments by running `julia run_experiment_f1.jl --help`. 

Running multiple experiments in parallel:

`cat umap_list | xargs -n 1 -P 32 ./run_experiment_umap.sh`

# Evaluation

