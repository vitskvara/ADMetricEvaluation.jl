using ADMetricEvaluation
using UCI
using DataFrames, PyPlot, CSV, Statistics
using EvalCurves
using EvalCurves: beta_auc, localized_auc, f1_at_fpr
ADME = ADMetricEvaluation
include("../models.jl")

umap_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_beta_contaminated-0.00"
full_path_0 = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_beta_contaminated-0.00"

row_measures = [:auc, :auc_weighted, :auc_at_1, :tpr_at_1, :bauc_at_1, :lauc_at_1,
		:auc_at_5, :tpr_at_5, :bauc_at_5, :lauc_at_5]
column_measures = [:auc_at_1, :auc_at_5, :prec_at_1, :prec_at_5,
		:tpr_at_1, :tpr_at_5]

function exact_match(needle, haystack)
	n = length(needle)
	haystack[end-n+1:end] == needle
end
function compute_means(df, fprs)
	nr = size(df,1)
	df[!,:mean] = zeros(nr)
	for row in eachrow(df)
		cm = row[:measure]
		ms = filter(x->x!=cm, column_measures)
		row[:mean] = mean(row[ms])
	end
	for fpr in fprs
		col_head = Symbol("mean_at_$fpr")
		df[!,col_head] = zeros(nr)
		for row in eachrow(df) 
			cm = row[:measure]
			ms = filter(y->exact_match("_$fpr", string(y)), filter(x->x!=cm, column_measures))
			row[col_head] = mean(row[ms])
		end
	end
	return df
end

function filtered_rel_measure_loss(data_path, row_measures, column_measures, target_measure, fprs)
	alldf_val = ADME.collect_fold_averages(data_path; data_type = "validation")
	alldf_test = ADME.collect_fold_averages(data_path; data_type = "test")

	# now collect the best results
	measure_dict_val = Dict(zip(row_measures, 
		map(x->ADME.collect_rows_model_is_parameter(alldf_val,alldf_test,x,column_measures),row_measures)))
	# these are as if selected by the column measures
	measure_dict_test = Dict(zip(column_measures, 
		map(x->ADME.collect_rows_model_is_parameter(alldf_val,alldf_test,x,column_measures),column_measures)))

	# we have to throw away some datasets where the results of bauc are either all NANs or they are
	# mostly nans
	datasets = unique(alldf_val[!,:dataset])
	somenan_datasets = filter(x->any(isnan.(filter(r->r[:dataset]==x, alldf_val)[!,target_measure])), 
		datasets)

	results = ADME.compare_eval_test_measures(data_path, row_measures, column_measures;
		filtered_datasets = somenan_datasets)
	return compute_means(results[3], fprs)
end

fprs = [1,5]
target_measure = :bauc_at_5
umap_res_filtered = filtered_rel_measure_loss(umap_path, row_measures, column_measures, target_measure, fprs)
full_res_filtered = filtered_rel_measure_loss(full_path_0, row_measures, column_measures, target_measure, fprs)

#############
results_umap = ADME.compare_eval_test_measures(umap_path, row_measures, column_measures)
results_full_0 = ADME.compare_eval_test_measures(full_path_0, row_measures, column_measures)

umap_df = results_umap[3]
full_df = results_full_0[3]

# collect the means over validation and test results
data_path = full_path_0
alldf_val = ADME.collect_fold_averages(data_path; data_type = "validation")
alldf_test = ADME.collect_fold_averages(data_path; data_type = "test")

# now collect the best results
measure_dict_val = Dict(zip(row_measures, 
	map(x->ADME.collect_rows_model_is_parameter(alldf_val,alldf_test,x,column_measures),row_measures)))
# these are as if selected by the column measures
measure_dict_test = Dict(zip(column_measures, 
	map(x->ADME.collect_rows_model_is_parameter(alldf_val,alldf_test,x,column_measures),column_measures)))

# we have to throw away some datasets where the results of bauc are either all NANs or they are
# mostly nans
measure = :bauc_at_5
datasets = unique(alldf_val[!,:dataset])
allnan_datasets = filter(x->all(isnan.(filter(r->r[:dataset]==x, alldf_val)[!,measure])), datasets)
somenan_datasets = filter(x->any(isnan.(filter(r->r[:dataset]==x, alldf_val)[!,measure])), datasets)

somenan_datasets = filter(x->sum(isnan.(filter(r->r[:dataset]==x, alldf_val)[!,measure])) > length(filter(r->r[:dataset]==x, alldf_val)[!,measure])/2, datasets)

filtered_datasets = allnan_datasets
filtered_datasets = somenan_datasets

alldf_val_filtered = filter(r->!(r[:dataset] in filtered_datasets), alldf_val)
alldf_tst_filtered = filter(r->!(r[:dataset] in filtered_datasets), alldf_tst)

measure_dict_val_filtered = Dict(zip(row_measures, 
	map(x->ADME.collect_rows_model_is_parameter(alldf_val_filtered,alldf_test,x,column_measures),row_measures)))
# these are as if selected by the column measures
measure_dict_test_filtered = Dict(zip(column_measures, 
	map(x->ADME.collect_rows_model_is_parameter(alldf_val_filtered,alldf_test,x,column_measures),column_measures)))


results_umap_filtered = ADME.compare_eval_test_measures(umap_path, row_measures, column_measures;
	filtered_datasets = filtered_datasets)
results_full_0_filtered = ADME.compare_eval_test_measures(full_path_0, row_measures, column_measures;
	filtered_datasets = filtered_datasets)

umap_df_filtered = results_umap_filtered[3]
full_df_filtered = results_full_0_filtered[3]

##### check the histograms
function plot_hist(x, args...; kwargs...)
	_x = x[.!isnan.(x)]
	if length(_x) > 0
		hist(_x, args...; kwargs...)
	end
end
datasets = unique(alldf_val[!,:dataset])

figure(figsize=(10,10))
for (i,dataset) in enumerate(datasets[33:48])
	_subdf = filter(r->r[:dataset]==dataset, alldf_val)
	subplot(4,4,i)
	title(dataset)
	plot_hist(_subdf[!,:auc_at_5], 30, alpha=0.3, label="AUC@5")
	plot_hist(_subdf[!,:bauc_at_5], 30, alpha=0.3, label="bAUC@5")
end
legend()
tight_layout()
#savefig("/home/vit/vyzkum/measure_evaluation/beta/bauc_vs_auc_histograms.png")
# it actually seems that bauc does not have much smaller variance

##### now find the best and worst results for bauc_at_5
row_measure = :bauc_at_5
col_measure = :tpr_at_5
row_measure2 = :auc_at_5

# this computes the mean loss by dataset
x1 = measure_dict_val[row_measure][!,col_measure]
x2 = measure_dict_val[col_measure][!,col_measure]
dx = x2-x1

x1 = measure_dict_val[row_measure2][!,col_measure]
x2 = measure_dict_val[col_measure][!,col_measure]
dx2 = x2-x1

title("mean measure loss over datasets")
hist(dx, 100,alpha=0.3, label="bAUC@5")
hist(dx2, 30,alpha=0.3, label="AUC@5")
legend()
savefig("/home/vit/vyzkum/measure_evaluation/beta/bauc_vs_auc_loss_histograms.png")

x1f = measure_dict_val_filtered[row_measure][!,col_measure]
x2f = measure_dict_val_filtered[col_measure][!,col_measure]
dxf = x2f-x1f

x1f = measure_dict_val_filtered[row_measure2][!,col_measure]
x2f = measure_dict_val_filtered[col_measure][!,col_measure]
dx2f = x2f-x1f

title("mean measure loss over datasets")
hist(dxf, 100,alpha=0.3, label="bAUC@5")
hist(dx2f, 30,alpha=0.3, label="AUC@5")
legend()
savefig("/home/vit/vyzkum/measure_evaluation/beta/bauc_vs_auc_loss_histograms_filtered.png")


sortis = sortperm(dx)
sortis = sortis[.!isnan.(dx[sortis])]
maxi = sortis[end]
mini = sortis[1]
i = sortis[end-1]

measure_dict_val[row_measure][i,:]


dataset = "libras-1-8"
subdf_val = filter(r->r[:dataset]==dataset, alldf_val)


# now do the same hsitogram for the filtered data
data_path = umap_path
alldf_val = ADME.collect_fold_averages(data_path, row_measures; data_type = "validation")
alldf_test = ADME.collect_fold_averages(data_path, column_measures; data_type = "test")

# now collect the best results
measure_dict_val = Dict(zip(row_measures, 
	map(x->ADME.collect_rows_model_is_parameter(alldf_val,alldf_test,x,column_measures),row_measures)))
# these are as if selected by the column measures
measure_dict_test = Dict(zip(column_measures, 
	map(x->ADME.collect_rows_model_is_parameter(alldf_val,alldf_test,x,column_measures),column_measures)))



figure()
scatter(alldf_val[!,:bauc_at_5], alldf_test[!,:tpr_at_5],s=2,label="bAUC@5")
scatter(alldf_val[!,:auc_at_5], alldf_test[!,:tpr_at_5],s=2, label="AUC@5")
ylabel("TPR@5")
legend()
savefig("/home/vit/vyzkum/measure_evaluation/beta/bauc_scatter.png")




# here replicate the experimental results for a given dataset, model and hyparparams
seed = 1
dataset = "libras"
subdataset = "libras-1-8"
data_path = UCI.get_umap_datapath()
p = 0.6
contamination = 0.0
standardize = true
nsamples = 1000

# this one is the only one that covnerges
# kNN    │ metric=kappa k=13

modelf = OCSVM_model
params = [50]
model = modelf(params...)
raw_data = UCI.get_data(dataset, path=data_path)
multiclass_data = UCI.create_multiclass(raw_data...)
data = filter(x->occursin(x[2], subdataset),multiclass_data)[1][1]

X_tr, y_tr, X_val_tst, y_val_tst = UCI.split_data(data, p, contamination;
	seed = seed, standardize=standardize)
X_val, y_val, X_tst, y_tst = UCI.split_val_test(X_val_tst, y_val_tst);

ScikitLearn.fit!(model, Array(transpose(X_tr)))
score_fun(X) = -ScikitLearn.decision_function(model, Array(transpose(X)))

X = copy(X_val)
y = copy(y_val)

scores = score_fun(X)
fprvec, tprvec = roccurve(scores, y)


measures = DataFrame()
measures[!,:auc] = [auc(fprvec, tprvec)]
measures[!,:auc_weighted] = [auc(fprvec, tprvec, "1/x")]

# now the rest
throw_errs = true
fpr = 0.05
sfpr = "$(round(Int,100*fpr))"
measures[!,Symbol("auc_at_$sfpr")] = [auc_at_p(fprvec,tprvec,fpr; normalize = true)]
measures[!,Symbol("tpr_at_$sfpr")] = [tpr_at_fpr(fprvec,tprvec,fpr)]
measures[!,Symbol("prec_at_$sfpr")] = [ADME.mean_precision_at_p(score_fun, X, y, fpr)]
measures[!,Symbol("f1_at_$sfpr")] = [f1_at_fpr(scores, y, fpr, warns=false)]
measures[!,Symbol("bauc_at_$sfpr")] = [beta_auc(scores, y, fpr, nsamples; d=0.5, warns=true)]
measures[!,Symbol("lauc_at_$sfpr")] = [localized_auc(scores, y, fpr, nsamples; d=0.5, normalize=true, 
	warns=true)]



# why is there no lower fpr than NaN?
# first sample fprs and get parameters of the beta distribution
fprs = EvalCurves.fpr_distribution(scores, y, fpr, nsamples, 0.5, warns=true)

thresholds = map(i->EvalCurves.threshold_at_fpr_sample(scores, y, 0.05, 0.5; warns=true), 1:nsamples)
fprs = map(x->EvalCurves.fpr_at_threshold(scores, y, x), thresholds)

fprs = fprs[.!isnan.(fprs)]
(length(fprs) == 0) ? (return NaN) : nothing
α, β = EvalCurves.estimate_beta_params(fprs)

beta_auc(scores, y, 0.00, nsamples; d=0.5, warns=true)