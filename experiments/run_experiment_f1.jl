# run experiment on the full dataset
import ADMetricEvaluation
ADME = ADMetricEvaluation
using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "dataset"
		required = true
        help = "dataset"
    "outpath"
    	required = true
    	help = "where to save the run"
    "train-contamination"
    	required = true
    	arg_type = Float64
    	help = "training contamination rate"
    "--models"
    	required = false
    	default = [""]
    	help = "one or more of [kNN, OCSVM, IF, LOF]"
    	nargs = '+'
    "--test-contamination"
    	arg_type = Float64
    	default = -1.0
    	help = "testing contamination rate"
    "--umap"
    	action = :store_true
    	help = "use the umap version of data"
    "--nexperiments"
    	arg_type = Int
    	default = 10
    	help = "number of different seeds for train/test split"
    "--mc-volume-iters"
    	arg_type = Int
    	default = 10000
    	help = "number of MC samples for the enclosed volume computation"
end
parsed_args = parse_args(ARGS, s)
dataset = parsed_args["dataset"]
outpath = parsed_args["outpath"]
tr_contamination = parsed_args["train-contamination"]
model_names = parsed_args["models"]
tst_contamination = (parsed_args["test-contamination"] == -1) ? nothing : parsed_args["test-contamination"]

#using Plots
#plotly()
include("models.jl")


# setup the path
host = gethostname()
if host == "vit-ThinkPad-E470"
	global outpath = joinpath("/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/",outpath)
elseif host == "axolotl.utia.cas.cz"
	global outpath = joinpath("/home/skvara/work/anomaly_detection/data/metric_evaluation/",outpath)
elseif host == "soroban-node-03"
	global outpath = joinpath("/compass/home/skvara/anomaly_detection/data/metric_evaluation/", outpath)
end
mkpath(outpath)
println("Output is going to be saved to "*outpath)

# settings
n_experiments = parsed_args["nexperiments"]
p = 0.8
mc_volume_iters = parsed_args["mc-volume-iters"]
mc_volume_repeats = 10

# models
models = [kNN_model, LOF_model, OCSVM_model, IF_model]
if model_names == [""]
	global model_names = ["kNN", "LOF", "OCSVM", "IF"]
end
param_struct = [
				([[1, 3, 5, 7, 9, 13, 21, 31, 51], [:gamma, :kappa, :delta]], [:k,:metric]),
			 	([[10, 20, 50]], [:num_neighbors]),
			 	([[0.01 0.05 0.1 0.5 1. 5. 10. 50. 100.]], [:gamma]),
			 	([[50 100 200]], [:num_estimators]),
			 ]
if parsed_args["umap"]
	global experiment_fun = ADME.run_umap_experiment
else	
	global experiment_fun = ADME.run_experiment
end
@time res = experiment_fun(
	dataset, models, model_names, param_struct, outpath;
	n_experiments = n_experiments, 
	p = p, 
	contamination=tr_contamination, 
	mc_volume_iters = mc_volume_iters, 
	mc_volume_repeats = mc_volume_repeats, 
	standardize=true,
	test_contamination = tst_contamination 
	)
