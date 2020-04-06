using PaperUtils
using ADMetricEvaluation
using DataFrames
using Statistics
ADME = ADMetricEvaluation

using CSV


umap_path = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/umap_beta_contaminated-0.00"
full_path_0 = "/home/vit/vyzkum/anomaly_detection/data/metric_evaluation/full_beta_contaminated-0.00"

row_measures = [:auc, :auc_weighted, :auc_at_1, :auc_at_5, :prec_at_1, :prec_at_5, 
		:tpr_at_1, :tpr_at_5, :f1_at_1, :f1_at_5, :bauc_at_1, :bauc_at_5,
		:lauc_at_1, :lauc_at_5]
column_measures = [:auc_at_1, :auc_at_5, :auc_at_10, :prec_at_1, :prec_at_5, :prec_at_10,
		:tpr_at_1, :tpr_at_5, :tpr_at_10]

#mean_diff, sd_diff, rel_mean_diff, rel_sd_diff = 
#	compare_eval_test_measures(data_path, row_measures, col_measures)

rel_mean_diff

data_path = umap_path
data_path = full_path


process_

results_umap = compare_eval_test_measures(umap_path, row_measures, column_measures)
results_full_0 = compare_eval_test_measures(full_path_0, row_measures, column_measures)

figure()
scatter(alldf_val[!,:bauc_at_5], alldf_test[!,:tpr_at_5],s=2,label="bAUC@5")
scatter(alldf_val[!,:auc_at_5], alldf_test[!,:tpr_at_5],s=2, label="AUC@5")
ylabel("TPR@5")
legend()
savefig("/home/vit/vyzkum/measure_evaluation/beta/bauc_scatter.png")
