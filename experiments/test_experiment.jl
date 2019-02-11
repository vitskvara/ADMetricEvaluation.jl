# run experiment on the full dataset
import ADMetricEvaluation
ADME = ADMetricEvaluation
using Plots
plotly()
include("models.jl")

host = gethostname()
#master path where data will be stored
outpath = "test"

mkpath(outpath)

dataset = ARGS[1]

# settings
n_experiments = 1
p = 0.8
mc_volume_iters = 10
mc_volume_repeats = 10

# models
models = [kNN_model]
model_names = ["kNN"]
param_struct = [
				([[1, 3, 9], [:kappa, :gamma]], [:k,:metric])
			 ]

@time res = ADME.run_experiment(dataset, models, model_names, param_struct, outpath;
	n_experiments = n_experiments, p = p, contamination=0.05, mc_volume_iters = mc_volume_iters, 
	mc_volume_repeats = mc_volume_repeats, standardize=true)

#@time res = ADME.run_umap_experiment(dataset, models, model_names, param_struct, outpath;
#	n_experiments = n_experiments, p = p, contamination=0.05, mc_volume_iters = mc_volume_iters, 
#	mc_volume_repeats = mc_volume_repeats, standardize=true)
println(res)