using PaperUtils
using ADMetricEvaluation
using DataFrames
using Statistics
ADME = ADMetricEvaluation

using CSV

umap_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_beta_contaminated-0.00"
full_path_0 = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_beta_contaminated-0.00"

row_measures = [:auc, :auc_weighted, :bauc_at_1, :bauc_at_5, :bauc_at_10, :auc_at_1, :auc_at_5, 
		:prec_at_1, :prec_at_5, :tpr_at_1, :tpr_at_5, :f1_at_1, :f1_at_5,
		:lauc_at_1, :lauc_at_5]
column_measures = [:auc_at_1, :auc_at_5, :auc_at_10, :prec_at_1, :prec_at_5, :prec_at_10,
		:tpr_at_1, :tpr_at_5, :tpr_at_10]

results_umap = ADME.compare_eval_test_measures(umap_path, row_measures, column_measures)
results_full_0 = ADME.compare_eval_test_measures(full_path_0, row_measures, column_measures)


rel_mean_umap = results_umap[3]
rel_mean_full = results_full_0[3]

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

fprs = [1, 5, 10]
rel_mean_umap = compute_means(rel_mean_umap, fprs)
rel_mean_full = compute_means(rel_mean_full, fprs)

showall(rel_mean_umap)
showall(rel_mean_full)
