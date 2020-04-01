# do the following beta fitting experiment
# for a single model:
# - train
# - compute scores on test
# - choose an fpr value
# - get a distribution of thresholds@fpr by doing random subsplits
# - show the histograms
# - then fit the thresholds by a beta distribution
# maybe choose the parameters such that E(beta) = fpr that we are interested in
# E(beta(a,b)) = a/(a+b) = fpr
# or such that mode(beta) = (a-1)/(a+b-2) pro a,b > 1
# also we want it such that the variance of the beta distribution is larger for smaller datasets
# - then we do some evaluation on a validation dataset

# first get the data
using UCI, ADMetricEvaluation, PyPlot, DataFrames, EvalCurves, Statistics, StatsBase, SpecialFunctions
ADME = ADMetricEvaluation
include("../models.jl")
savepath = "/home/vit/vyzkum/measure_evaluation/beta"

score_fun(m,X) = -ScikitLearn.decision_function(m, Array(transpose(X)))

# create the model, train it and get the thresholds from validation data
model_functions = [kNN_model, LOF_model, OCSVM_model, IF_model]
model_strings = ["kNN", "LOF", "OCSVM", "IF"]
param_struct = [
                ([[1, 3, 5, 7, 9, 13, 21, 31, 51], [:gamma, :kappa, :delta]], [:k,:metric]),
                ([[10, 20, 50]], [:num_neighbors]),
                ([[0.01 0.05 0.1 0.5 1. 5. 10. 50. 100.]], [:gamma]),
                ([[50 100 200]], [:num_estimators])
                ]
# create the models             ]
models = [model_functions[1](3, :kappa), model_functions[2](10), 
	model_functions[3](1), model_functions[4](2)]
#imodel, iparams = 1, 2
#params = [x[iparams] for x in param_struct[imodel][1]]
imodel = 1
model = models[imodel]

dataset = "wine"
data, _, _ = UCI.get_loda_data(dataset)
seed = 1
tr_contamination = 0.0
tst_contamination = nothing
train_x, train_y, val_test_x, val_test_y = UCI.split_data(data, 0.6, tr_contamination; 
    test_contamination = tst_contamination, 
    seed = seed, 
    difficulty = [:easy, :medium], 
    standardize=true)

val_x, val_y, test_x, test_y = UCI.split_val_test(val_test_x, val_test_y)

# now resample the validation set scores and get different thresholds and fpr values
nsamples = 50
d = 0.5 # how many samples are to be resampled
fpr = 0.01

# train it
ScikitLearn.fit!(model, Array(transpose(train_x)))
score_fun(X) = -ScikitLearn.decision_function(model, Array(transpose(X))) 
val_scores = score_fun(val_x)

# select fpr value and get auc@fpr
roc = roccurve(val_scores, val_y)
auroc = auc(roc...)
auc_at = auc_at_p(roc..., fpr, normalize=true)
tpr_at = tpr_at_fpr(roc..., fpr)
fprs = EvalCurves.fpr_distribution(val_scores, val_y, fpr, 100)
α, β = EvalCurves.estimate_beta_params(fprs)
betax = 0:0.002:1.0
betay = exp.(EvalCurves.beta_logpdf.(betax, α, β))
# centered weights by pdf
dx = roc[1][2:end] - roc[1][1:end-1]
dx0 = unique(dx[dx .!= 0])[1]
w = exp.(EvalCurves.beta_logpdf.(roc[1][1:end-1].+dx0, α, β))
wauroc = auc(roc..., w)

figure(figsize=(4,10))
subplot(311)
title("fprs@$(fpr) histogram")
hist(fprs, 101, density=false)
subplot(312)
title("beta($(round(α, digits=2)), $(round(β, digits=2)))")
hist(fprs, density=true)
plot(betax, betay)
subplot(313)
rauc = round(auroc, digits=2)
rwauc = round(wauroc, digits=2)
title("AUC=$(rauc), beta weighted AUC=$(rwauc)")
plot(roc...)

# do it for all the models
function plot_model_results(models, nsamples, fpr, dataset)
	figure(figsize=(12,10))
	n1 = sum(val_y)
	n0 = length(val_y) - n1
	suptitle("$dataset, n0=$(n0), n1=$(n1), fpr=$(fpr)")
	for (i,model) in enumerate(models)
		for _ in 1:10 # do this multiple times - some models tend to not converge
			try
				ScikitLearn.fit!(model, Array(transpose(train_x)))
				val_scores = score_fun(model, val_x)

				roc = roccurve(val_scores, val_y)
				auroc = auc(roc...)
				auc_at = auc_at_p(roc..., fpr, normalize=true)
				tpr_at = tpr_at_fpr(roc..., fpr)
				fprs = EvalCurves.fpr_distribution(val_scores, val_y, fpr, nsamples)
				α, β = EvalCurves.estimate_beta_params(fprs)
				betax = 0:0.002:1.0
				betay = exp.(EvalCurves.beta_logpdf.(betax, α, β))
				# centered weights by pdf
				dx = roc[1][2:end] - roc[1][1:end-1]
				dx0 = unique(dx[dx .!= 0])[1]
				w = exp.(EvalCurves.beta_logpdf.(roc[1][1:end-1].+dx0, α, β))
				wauroc = auc(roc..., w)

				subplot(3,4,i)
				title("$(model_strings[i])\nfprs@$(fpr) histogram")
				hist(fprs, 101, density=false)
				subplot(3,4,4+i)
				title("beta($(round(α, digits=2)), $(round(β, digits=2)))")
				hist(fprs, density=true)
				plot(betax, betay)
				ylim([-1,80])
				subplot(3,4,8+i)
				rauc = round(auroc, digits=2)
				rwauc = round(wauroc, digits=2)
				title("AUC=$(rauc)\nbeta weighted AUC=$(rwauc)")
				plot(roc...)
				break
			catch e
				continue
			end
		end
	end
	tight_layout(rect=[0,0.03,1,0.97])
end

nsamples = 100
fpr = 0.01
dataset = "statlog-segment"

data, _, _ = UCI.get_loda_data(dataset);
seed = 1;
tr_contamination = 0.0
tst_contamination = nothing
train_x, train_y, val_test_x, val_test_y = UCI.split_data(data, 0.6, tr_contamination; 
    test_contamination = tst_contamination, 
    seed = seed, 
    difficulty = [:easy, :medium], 
    standardize=true);
val_x, val_y, test_x, test_y = UCI.split_val_test(val_test_x, val_test_y);

plot_model_results(models, nsamples, fpr, dataset)
savefig(joinpath(savepath, "$(dataset)_$(fpr).png"))