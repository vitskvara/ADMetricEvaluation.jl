using Statistics, DataFrames, CSV
using UCI, ADMetricEvaluation, EvalCurves
using JLD2, FileIO, Distributions
using PyPlot
include("../models.jl")
ADME = ADMetricEvaluation
outpath = "/home/vit/vyzkum/measure_evaluation/t-test"

# two sample t test
welch_test_statistic(μ1::Real, μ2::Real, σ1::Real, σ2::Real, n1::Int, n2::Int) = 
	(μ1 - μ2)/sqrt(σ1^2/n1 + σ2^2/n2)
welch_df(σ1::Real, σ2::Real, n1::Int, n2::Int) = 
	floor(Int, (σ1^2/n1 + σ2^2/n2)^2/(σ1^4/(n1^2*(n1-1)) + σ2^4/(n2^2*(n2-1))))
critt(α::Real, df::Int) = quantile(TDist(df), 1-α)
welch_pval(t::Real, df::Int) = 1-cdf(TDist(df), t) # onesided version
# for the two sided test we just multiply this by a factor of 2

# how do i pickup the right models and dataset?
#  - i can just try some randomly
#  - i can explore the stored results
#		- find a dataset and a set of two models where one is better at 
#		  fpr@0.01 and the the aother at fpr@0.05
base_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation"
data_path = joinpath(base_path, "full_data_contaminated-0.01/")
full_df = ADME.collect_all_data(data_path; aggreg_f = mean)
filter!(r->r[:model] in ["OCSVM", "kNN"],full_df)
function find_2_best(df, metrics)
	dfs = []
	colnames = vcat([:dataset, :model, :params], metrics)
	for dataset in unique(df[!,:dataset])
		subdf = filter(r->r[:dataset] == dataset && !isnan(r[metrics[1]]) && !isnan(r[metrics[2]]), df)
		for metric in metrics
			row = subdf[argmax(subdf[metric]),colnames]
			push!(dfs, row)
		end
	end
	vcat(map(x->DataFrame(x),dfs)...)
end
metrics = [:auc_at_1_mean, :auc_at_5_mean]
best_auc_df = find_2_best(full_df, metrics)
datasets = unique(full_df[!,:dataset])
m1_vals = []
m2_vals = []
for dataset in datasets
	subdf = filter(r->r[:dataset]==dataset,best_auc_df)
	push!(m1_vals, subdf[1,metrics[1]] - subdf[2,metrics[1]])
	push!(m2_vals, subdf[1,metrics[2]] - subdf[2,metrics[2]])
end
diff_df = DataFrame(
	:dataset => datasets,
	metrics[1] => m1_vals,
	metrics[2] => m2_vals,
	:max_diff => m1_vals - m2_vals
	)
max_diff_dataset = diff_df[argmax(diff_df[!,:max_diff]),:dataset]
filter(r->r[:dataset]==max_diff_dataset, best_auc_df)
# 
#│ Row │ dataset              │ model  │ params            │ auc_at_1_mean │ auc_at_5_mean │
#│     │ String               │ String │ String            │ Float64       │ Float64       │
#├─────┼──────────────────────┼────────┼───────────────────┼───────────────┼───────────────┤
#│ 1   │ statlog-satimage-1-3 │ OCSVM  │ gamma=0.5         │ 0.113533      │ 0.285851      │
#│ 2   │ statlog-satimage-1-3 │ kNN    │ metric=kappa k=21 │ 0.08976       │ 0.628003      │
# get data
base_dataset = "statlog-satimage"
sub = "1-3"
dataset_fname = base_dataset*"-"*sub
data, normal_labels, anomaly_labels = UCI.get_data(base_dataset)
subdatasets = UCI.create_multiclass(data, normal_labels, anomaly_labels)
data = filter(x->x[2]==sub,subdatasets)[1][1]
# construct models
model_names = [:OCSVM, :kNN]
model_params = [[0.5], [21, :kappa]]
models = map(x->eval(Symbol(String(x[1])*"_model"))(x[2]...), 
	zip(model_names, model_params))
connect_minus(a,b) = string(a)*"-"*string(b)
connect_und(a,b) = string(a)*"_"*string(b)
model_fname = reduce(connect_und, map(x->reduce(connect_minus,vcat(x[1],x[2])),
	zip(model_names, model_params)))
# for a selected dataset and two models, plot the welch statistic vs changing fpr and 
# also vs changing conamination rate
# what do i need to do:
# 1) decide how to get the sds 
# - either do a crossfold valdiation
# - or do a bootstrapping on the level of testing data
# 2) for every test/train split 
# - get an aroc
# - compute pauc, tpr, auc (maybe also precision) and their sds for different fpr values
# - then use these for the welch statistic
# can we use the present data?
# - no, since it only contains fpr in [0.01, 0.05, 0.1]

score_fun(model, X) = -ScikitLearn.decision_function(model, Array(transpose(X))) 

function get_roc(model, X_tr, X_tst, y_tst)
	ScikitLearn.fit!(model, Array(transpose(X_tr)))
	scores = score_fun(model, X_tst)
	EvalCurves.roccurve(scores, y_tst)
end

function get_scores(roc, fprs, measure)
	map(fpr -> measure(roc...,fpr), fprs)
end

fprs = collect(0.01:0.01:0.3)
ks = collect(1:50)
fname = joinpath(outpath, dataset_fname*"_"*model_fname*".jld2")
if isfile(fname)
	measure_vals = load(fname)["measure_vals"]
else
	measure_vals = Dict{Symbol, Any}(
		:fpr => fprs,
		:k => ks
		)
	for (model, model_name) in zip(models, model_names)
		auc_at_p = []
		tpr_at_p = []
		rocs = []
		for k in ks
			X_tr, y_tr, X_tst, y_tst = UCI.split_data(data, 0.8, 0.01, seed=k, standardize=true)
			roc = get_roc(model, X_tr, X_tst, y_tst)
			push!(auc_at_p, get_scores(roc, fprs, 
				(x,y,z)->EvalCurves.auc_at_p(x,y,z,normalize=true)))
			push!(tpr_at_p, get_scores(roc, fprs, EvalCurves.tpr_at_fpr))
			push!(rocs, roc)
		end
		measure_vals[model_name] = Dict(
			:auc_at_p => hcat(auc_at_p...),
			:tpr_at_p => hcat(tpr_at_p...),
			:roc => rocs
			)
	end
	save(fname, "measure_vals", measure_vals)
end

# plot the histograms for different fprs
figure(figsize=(8,8))
for 

end

# plot the welch test statistics over fprs
n = ks[end]
pauc_means = map(m->mean(measure_vals[m][:auc_at_p],dims=2),model_names)
pauc_stds = map(m->std(measure_vals[m][:auc_at_p],dims=2),model_names)
pauc_welch = map(x->welch_test_statistic.(x[1], x[2], x[3], x[4], n, n), 
	zip(pauc_means[1], pauc_means[2], pauc_stds[1], pauc_stds[2]))

tpr_means = map(m->mean(measure_vals[m][:tpr_at_p],dims=2),model_names)
tpr_stds = map(m->std(measure_vals[m][:tpr_at_p],dims=2),model_names)
tpr_welch = map(x->welch_test_statistic.(x[1], x[2], x[3], x[4], n, n), 
	zip(tpr_means[1], tpr_means[2], tpr_stds[1], tpr_stds[2]))

α = 0.05
pauc_dfs = map(x->welch_df.(x[1], x[2], n, n), 
	zip(pauc_stds[1], pauc_stds[2]))
pauc_critvals = critt.(α/2, pauc_dfs)
pauc_pvals = welch_pval.(abs.(pauc_welch), pauc_dfs)*2

tpr_dfs = map(x->welch_df.(x[1], x[2], n, n), 
	zip(tpr_stds[1], tpr_stds[2]))
tpr_critvals = critt.(α/2, tpr_dfs)
tpr_pvals = welch_pval.(abs.(tpr_welch), tpr_dfs)*2

figure()
plot(fprs,abs.(tpr_welch), label="TPR@FPR")
plot(fprs,abs.(pauc_welch), label="AUC@FPR")
plot(fprs, tpr_critvals, label="TPR@FPR critval")
plot(fprs, pauc_critvals, label="AUC@FPR critval")
xlabel("FPR")
ylabel("Welch statistic")
legend()
title(base_dataset*"-"*sub)
savefig(joinpath(outpath, base_dataset*"-"*sub*".png"))

# just to be sure we have everything right
N1 = 15
X1 = 20.8
s1 = sqrt(7.9)
N2 = 15
X2 = 23
s2 = sqrt(3.8)

t = welch_test_statistic(X1, s1, X2, s2, N1, N2)
nu = welch_df(s1, s2, N1, N2)
p = welch_pval(abs(t), nu)*2

N1 = 10
X1 = 20.6
s1 = sqrt(9.0)
N2 = 20
X2 = 22.1
s2 = sqrt(0.9)

t = welch_test_statistic(X1, s1, X2, s2, N1, N2)
nu = welch_df(s1, s2, N1, N2)
p = welch_pval(abs(t), nu)*2

N1 = 10
X1 = 20.6
s1 = sqrt(9.0)
N2 = 20
X2 = 22.1
s2 = sqrt(0.9)

t = welch_test_statistic(X1, s1, X2, s2, N1, N2)
nu = welch_df(s1, s2, N1, N2)
p = welch_pval(abs(t), nu)*2


###
dataset = "breast-cancer-wisconsin"
subdf = filter(r->occursin(dataset,r[:dataset]),full_df)
seldf = vcat(DataFrame(sort(subdf, :auc_mean)[1,:]), 
	DataFrame(sort(subdf, :auc_mean)[end-1:end,:]))
