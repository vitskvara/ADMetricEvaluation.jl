# run experiment on the full dataset
import ADMetricEvaluation
ADME = ADMetricEvaluation
using Plots
plotly()
include("models.jl")

host = gethostname()
#master path where data will be stored
if host == "vit"
	outpath = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/data_contaminated-0.01"
elseif host == "axolotl.utia.cas.cz"
	outpath = "/home/skvara/work/anomaly_detection/data/metric_evaluation/full_data_contaminated-0.01"
elseif host == "soroban-node-03"
	outpath = "/compass/home/skvara/anomaly_detection/data/metric_evaluation/full_data_contaminated-0.01"
end

mkpath(outpath)

dataset = ARGS[1]

# settings
n_experiments = 10
p = 0.8
mc_volume_iters = 10000
mc_volume_repeats = 10

# models
models = [kNN_model, LOF_model, OCSVM_model, IF_model]
if length(ARGS)>1
	model_names = ARGS[2:end]
else
	model_names = ["kNN", "LOF", "OCSVM", "IF"]
end
param_struct = [
				([[1, 3, 5, 7, 9, 13, 21, 31, 51], [:gamma, :kappa, :delta]], [:k,:metric]),
			 	([[10, 20, 50]], [:num_neighbors]),
			 	([[0.01 0.05 0.1 0.5 1. 5. 10. 50. 100.]], [:gamma]),
			 	([[50 100 200]], [:num_estimators]),
			 ]

@time res = ADME.run_experiment(dataset, models, model_names, param_struct, outpath;
	n_experiments = n_experiments, p = p, contamination=0.01, mc_volume_iters = mc_volume_iters, 
	mc_volume_repeats = mc_volume_repeats, standardize=true)
