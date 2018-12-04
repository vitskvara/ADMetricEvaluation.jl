using ArgParse
using UCI
using EvalCurves
using ADMetricEvaluation
using DataFrames
using PyPlot
include("models.jl")
use_args = false

function _parse_args()
	# argument parsing
	s = ArgParseSettings(description = "Produce a surface plot of anomaly scores and other stuff for a given model, settings and data.")

	@add_arg_table s begin
		"dataset"
			help = "dataset"
			required = true
	    "model" 
	    	help = "model, one of kNN, IF, LOF or OCSVM"
	    	required = true
	    "params"
	    	help = "parameter settings of a model"
	    	nargs = '*'
	    	required = true
	    "--seed"
	    	help = "data split seed"
	    	default = 1
		"--subclass"
			help = "dataset subclass - index or string"
			default = 1
	  end

	parsed_args = parse_args(s) # the result is a Dict{String,Any}
end

if use_args
	parsed_args = _parse_args()
	dataset = parsed_args["dataset"]
	seed = parsed_args["seed"]
	if typeof(seed) != Int64
		seed = Int(Meta.parse(seed))
	end
	model = parsed_args["model"]
	global subclass = parsed_args["subclass"]
	try
		global subclass = Int(Meta.parse(subclass))
	catch
		nothing
	end
	params = parsed_args["params"]
	params = map(Meta.parse, params)
else
	# specify your own inputs
	dataset = "three-rings-1"
	seed = 1
	model = "kNN"
	subclass = 1
	params = [5, :gamma]
end

# data acquisition
p = 0.8
standardize = true
if dataset in ["three-gaussians-1", "three-rings-1", "two-bananas", "two-moons",
    "two-rings-2", "three-gaussians-2", "three-rings-2", "two-gaussians", "two-rings-1"]
    data = UCI.get_synthetic_data(dataset)
else
	data = UCI.get_umap_data(dataset, subclass)
end
X_tr, y_tr, X_tst, y_tst = UCI.split_data(data, p; seed = seed, standardize=standardize)

# construct and train the model
models = Dict(
	"kNN" => kNN_model,
	"LOF" => LOF_model,
	"OCSVM" => OCSVM_model,
	"IF" => IF_model
)
m = models[model](params...)
ScikitLearn.fit!(m, Array(transpose(X_tr)))
score_fun(X) = -ScikitLearn.decision_function(m, Array(transpose(X))) 

# now compute scores and other metrics
scores = score_fun(X_tst)
fprvec, tprvec = EvalCurves.roccurve(scores, y_tst)

df = DataFrame() # auc, weighted auc, auc@5, auc@1, precision@k, tpr@fpr, vol@fpr
df[:auc] = EvalCurves.auc(fprvec, tprvec)
df[:auc_weighted] = EvalCurves.auc(fprvec, tprvec, "1/x")
df[:auc_at_5] = EvalCurves.auc_at_p(fprvec,tprvec,0.05; normalize = true)
df[:prec_at_5] = ADMetricEvaluation.precision_at_p(score_fun, X_tst, y_tst, 0.05)
df[:tpr_at_5] = EvalCurves.tpr_at_fpr(fprvec, tprvec, 0.05)
# now get the threshold and volume
X = hcat(X_tr, X_tst)
bounds = EvalCurves.estimate_bounds(X)
fpr = 0.05
mc_volume_iters = 1000
mc_volume_repeats = 1
threshold = EvalCurves.threshold_at_fpr(scores, y_tst, fpr; warn = false)
df[:threshold] = threshold
df[:vol_at_5] = 1-EvalCurves.volume_at_fpr(threshold, bounds, score_fun, mc_volume_iters)

print(df)

# get data for contours
gridsize = 10
plotbounds=bounds.*1.1
_x = range(plotbounds[1][1], length=gridsize, stop=plotbounds[1][2])
_y = range(plotbounds[2][1], length=gridsize, stop=plotbounds[2][2])
_z = fill(0.0, gridsize, gridsize)
_l = Int.((threshold .- _z).<1e-6)
for i in 1:gridsize
	for j in 1:gridsize
		_z[j,i] = score_fun([_x[i], _y[j]])[1]
	end
end

# do the plots
_cmap = "gray"
_s = 10
figure(figsize=(10,5))
subplot(1,2,1)
suptitle("$dataset, $model, $params\n
	AUC=$(df[:auc]), AUCw=$(df[:auc_weighted]), AUC5=$(df[:auc_at_5]), PREC5=$(df[:prec_at_5]), TPR5=$(df[:tpr_at_5]), VOL5=$(df[:vol_at_5])")

contourf(_x,_y,_l,1,cmap=_cmap)
scatter(X_tr[1,:], X_tr[2,:], label="training data", s=_s)
xlim(plotbounds[1])
ylim(plotbounds[2])
title("training data + 0/1 volumes")
legend()

subplot(1,2,2)
contourf(_x,_y,_z,50,cmap=_cmap)
scatter(X_tst[1,y_tst.==0], X_tst[2,y_tst.==0], label="negative", s=_s)
scatter(X_tst[1,y_tst.==1], X_tst[2,y_tst.==1], label="positive", s=_s)
xlim(plotbounds[1])
ylim(plotbounds[2])
title("testing data + anomaly score contours")
legend()
tight_layout()

show()
