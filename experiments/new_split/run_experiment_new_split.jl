# run experiment on the full dataset
import ADMetricEvaluation
ADME = ADMetricEvaluation
using ArgParse, UCI

s = ArgParseSettings()
@add_arg_table! s begin
    "dataset"
		required = true
        help = "dataset"
    "normal-class"
        required = true
        help = "normal class label"
    "anomalous-class"
        required = true
        help = "anomalous class label"
    "outpath"
    	required = true
    	help = "where to save the run"
    "--models"
    	required = false
    	default = [""]
    	help = "one or more of [kNN, OCSVM, IF, LOF]"
    	nargs = '+'
    "--max-fpr"
        arg_type = Float64
        default = 0.1
        help = "the maximum admissible fpr"
    "--test"
        action = :store_true
        help = "test run"
end
parsed_args = parse_args(ARGS, s)
dataset = parsed_args["dataset"]
normal_class = parsed_args["normal-class"]
anomalous_class = parsed_args["anomalous-class"]
outpath = parsed_args["outpath"]
model_names = parsed_args["models"]
max_fpr = parsed_args["max-fpr"]
testrun = parsed_args["test"]

include("../models.jl")

# setup the path
host = gethostname()
if host == "vit-ThinkPad-E470"
	global outpath = joinpath("/home/vit/vyzkum/measure_evaluation/new_split/",outpath)
elseif host == "axolotl.utia.cas.cz"
	global outpath = joinpath("/home/skvara/work/anomaly_detection/data/metric_evaluation/new_split/",outpath)
elseif occursin("soroban",host)
	global outpath = joinpath("/compass/home/skvara/anomaly_detection/data/metric_evaluation/new_split/", outpath)
end
mkpath(outpath)
println("Output is going to be saved to "*outpath)

# models
default_model_names = ["kNN", "LOF", "OCSVM", "IF"]
models = [kNN_model, LOF_model, OCSVM_model, IF_model]
param_struct = [
                ([[1, 3, 5, 7, 9, 13, 21, 31, 51], [:gamma, :kappa, :delta]], [:k,:metric]),
                ([[10, 20, 50]], [:num_neighbors]),
                ([[0.01 0.05 0.1 0.5 1. 5. 10. 50. 100.]], [:gamma]),
                ([[50 100 200]], [:num_estimators]),
             ]
if model_names == [""]
	global model_names = default_model_names
end

# now select the functions, names and params of the actually used models
modelis = map(x->x in model_names, default_model_names)
model_names = Array(default_model_names[modelis])
models = Array(models[modelis])
param_struct = Array(param_struct[modelis])

# settings
p = 0.6
contamination = 0.0
standardize = true
test_contamination = nothing
data_path = ""
fprs = collect(range(0.01, max_fpr, step=0.01))
throw_errs = false

# test run
err_warns = false
if testrun
    dataset = "iris"
    normal_class = "setosa"
    anomalous_class = "versicolor"
    models = [kNN_model]
    model_names = ["kNN"]
    param_struct = [
                ([[1], [:gamma]], [:k,:metric])
             ]
    nexperiments = 2
    outpath = abspath(".")
    err_warns = true
    throw_errs = true
end


@time res = ADME.run_new_split_experiment(
    dataset, normal_class, anomalous_class, models, model_names, 
    param_struct, outpath; 
	p = p, 
    fprs = fprs,
	contamination=contamination, 
	standardize=standardize,
	test_contamination = test_contamination,
    throw_errs = throw_errs,
    err_warns = err_warns
	)
